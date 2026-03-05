"""Verify all data sources are DFSG-compliant.

Reads manifest.json files produced by download.py and checks
that every source has an acceptable license.
"""

import json
import sys
from pathlib import Path

DFSG_LICENSES = {
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "CC-BY-4.0",
    "CC0-1.0",
    "Public Domain",
    "MIT",
    "Apache-2.0",
    "GPL-2.0",
    "GPL-3.0",
}


def verify(data_dir: Path) -> bool:
    manifests = list(data_dir.rglob("manifest.json"))
    if not manifests:
        print(f"No manifest.json files found in {data_dir}")
        return False

    all_ok = True
    for manifest_path in manifests:
        meta = json.loads(manifest_path.read_text())
        source = meta.get("source", "unknown")
        license_id = meta.get("license", "UNKNOWN")

        if license_id in DFSG_LICENSES:
            print(f"  OK: {source} — {license_id}")
        else:
            print(f"  FAIL: {source} — {license_id} (not in DFSG allow-list)")
            all_ok = False

    return all_ok


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    print(f"Verifying licenses in {data_dir}...")
    if verify(data_dir):
        print("\nAll sources DFSG-compliant.")
    else:
        print("\nWARNING: Non-DFSG sources detected!")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
