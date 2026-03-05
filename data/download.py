"""Download DFSG-compliant training corpora.

Sources:
  - Wikipedia English (CC-BY-SA) via HuggingFace datasets
  - Project Gutenberg (Public Domain) via HuggingFace datasets
  - Stack Exchange (CC-BY-SA) via archive.org data dumps

Can either import pre-built JSONL files from ~/chat_hamner_v2/data/pretrain/
(if available) or download fresh from the original sources.

All sources are verified DFSG-clean. No Common Crawl derivatives.
No AI-generated content.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

CHAT_HAMNER_PRETRAIN = Path.home() / "chat_hamner_v2" / "data" / "pretrain"

# Mapping: our source name -> (chat_hamner filename, license, description)
IMPORTABLE_SOURCES = {
    "wikipedia": ("wikipedia.jsonl", "CC-BY-SA-4.0", "Wikipedia English"),
    "gutenberg": ("gutenberg.jsonl", "Public Domain", "Project Gutenberg"),
    "stackexchange": ("stackexchange.jsonl", "CC-BY-SA-4.0", "Stack Exchange"),
}


def _write_shard(out_path: Path, batch: list[str], file_idx: int, prefix: str) -> Path:
    """Write a batch of texts as a shard file."""
    shard_path = out_path / f"{prefix}_{file_idx:05d}.txt"
    shard_path.write_text("\n\n".join(batch), encoding="utf-8")
    return shard_path


def _write_manifest(out_path: Path, source: str, license_id: str, token_count: int, num_shards: int, **extra):
    """Write a manifest.json for a downloaded source."""
    meta = {
        "source": source,
        "license": license_id,
        "approx_tokens": token_count,
        "num_shards": num_shards,
        **extra,
    }
    (out_path / "manifest.json").write_text(json.dumps(meta, indent=2))


def import_from_chat_hamner(source: str, output_dir: Path, shard_size: int = 10000) -> bool:
    """Import pre-built JSONL from chat_hamner_v2 if available.

    The JSONL format is one JSON object per line with keys: text, source, license.
    We split into shards of plain text for our pipeline.

    Returns True if import succeeded, False if source not available.
    """
    filename, license_id, description = IMPORTABLE_SOURCES[source]
    src_path = CHAT_HAMNER_PRETRAIN / filename

    if not src_path.exists():
        return False

    out_path = output_dir / source
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = out_path / "manifest.json"
    if manifest.exists():
        print(f"{description} already imported, skipping. Remove {manifest} to re-import.")
        return True

    print(f"Importing {description} from {src_path}...")
    token_count = 0
    file_idx = 0
    batch = []

    with open(src_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=description, unit="docs"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("text", "").strip()
            if len(text) < 200:
                continue

            batch.append(text)
            token_count += len(text) // 4

            if len(batch) >= shard_size:
                _write_shard(out_path, batch, file_idx, source[:4])
                batch = []
                file_idx += 1

    if batch:
        _write_shard(out_path, batch, file_idx, source[:4])
        file_idx += 1

    _write_manifest(
        out_path, source, license_id, token_count, file_idx,
        imported_from=str(src_path),
    )
    print(f"{description}: ~{token_count:,} tokens in {file_idx} shards")
    return True


# --- Fresh download functions (fallback if chat_hamner_v2 not available) ---

def download_wikipedia(output_dir: Path, target_tokens: int = 2_000_000_000):
    """Download Wikipedia English articles via HuggingFace datasets."""
    out_path = output_dir / "wikipedia"
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = out_path / "manifest.json"
    if manifest.exists():
        print(f"Wikipedia already downloaded, skipping. Remove {manifest} to re-download.")
        return

    print("Downloading Wikipedia English articles from HuggingFace...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    token_count = 0
    file_idx = 0
    batch = []

    for article in tqdm(ds, desc="Wikipedia articles", unit="articles"):
        text = article["text"].strip()
        if len(text) < 200:
            continue
        batch.append(text)
        token_count += len(text) // 4

        if len(batch) >= 10000:
            _write_shard(out_path, batch, file_idx, "wiki")
            batch = []
            file_idx += 1

        if token_count >= target_tokens:
            break

    if batch:
        _write_shard(out_path, batch, file_idx, "wiki")
        file_idx += 1

    _write_manifest(out_path, "wikipedia", "CC-BY-SA-4.0", token_count, file_idx,
                    version="20231101.en")
    print(f"Wikipedia: ~{token_count:,} tokens in {file_idx} shards")


def download_gutenberg(output_dir: Path, target_tokens: int = 1_000_000_000):
    """Download Project Gutenberg texts via HuggingFace datasets."""
    out_path = output_dir / "gutenberg"
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = out_path / "manifest.json"
    if manifest.exists():
        print(f"Gutenberg already downloaded, skipping. Remove {manifest} to re-download.")
        return

    print("Downloading Project Gutenberg texts from HuggingFace...")
    ds = load_dataset("manu/project_gutenberg", split="en", streaming=True)

    token_count = 0
    file_idx = 0
    batch = []

    for book in tqdm(ds, desc="Gutenberg books", unit="books"):
        text = book.get("text", "").strip()
        if len(text) < 1000:
            continue
        text = _strip_gutenberg_boilerplate(text)
        if len(text) < 500:
            continue
        batch.append(text)
        token_count += len(text) // 4

        if len(batch) >= 100:
            _write_shard(out_path, batch, file_idx, "gute")
            batch = []
            file_idx += 1

        if token_count >= target_tokens:
            break

    if batch:
        _write_shard(out_path, batch, file_idx, "gute")
        file_idx += 1

    _write_manifest(out_path, "gutenberg", "Public Domain", token_count, file_idx)
    print(f"Gutenberg: ~{token_count:,} tokens in {file_idx} shards")


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1:]
            break

    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


def download_stackexchange(output_dir: Path, target_tokens: int = 500_000_000):
    """Download Stack Exchange posts via archive.org 7z dumps.

    This mirrors the approach in chat_hamner_v2/build_dataset.py:
    downloads .7z archives from archive.org/download/stackexchange/
    and extracts Posts.xml from each.
    """
    out_path = output_dir / "stackexchange"
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = out_path / "manifest.json"
    if manifest.exists():
        print(f"StackExchange already downloaded, skipping. Remove {manifest} to re-download.")
        return

    print("ERROR: Fresh StackExchange download requires 7z extraction.")
    print("       Import from chat_hamner_v2 instead, or run:")
    print("       python ~/chat_hamner_v2/build_dataset.py --source stackexchange")
    print("       then re-run this script.")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Download training corpora")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["wikipedia", "gutenberg", "stackexchange"],
        choices=["wikipedia", "gutenberg", "stackexchange"],
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Download fresh instead of importing from chat_hamner_v2",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fresh_downloaders = {
        "wikipedia": download_wikipedia,
        "gutenberg": download_gutenberg,
        "stackexchange": download_stackexchange,
    }

    for source in args.sources:
        if not args.fresh and source in IMPORTABLE_SOURCES:
            if import_from_chat_hamner(source, args.output_dir):
                continue
            print(f"  chat_hamner_v2 data not found for {source}, downloading fresh...")

        fresh_downloaders[source](args.output_dir)

    print("\nAll sources ready.")
    print("Next step: python data/clean.py")


if __name__ == "__main__":
    main()
