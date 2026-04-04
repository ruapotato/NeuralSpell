#!/usr/bin/env python3
"""GECToR-style tagger v2: uses the tokenizer vocabulary directly.

Instead of a fixed edit vocabulary (REPLACE_word), the model predicts:
  - Tag 0: KEEP (don't change)
  - Tag 1: DELETE (remove this token)
  - Tags 2-32001: Replace with token ID (from 32K BPE vocabulary)

No vocabulary gap — if the tokenizer can encode it, the model can predict it.
No edit vocab build step needed.

Usage:
    PYTHONPATH=. python training/train_tagger_v2.py \
        --data-dir data/processed \
        --tokenizer tokenizer/tokenizer.model \
        --checkpoint-dir checkpoints/tagger_v2
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import sentencepiece as spm

from model.tagger import SpellTagger
from corruption.engine import CorruptionEngine
from training.scheduler import WSDScheduler

# Tags: 0=KEEP, 1=DELETE, 2..32001=replace with token_id (0..31999)
KEEP_TAG = 0
DELETE_TAG = 1
REPLACE_OFFSET = 2  # tag = token_id + REPLACE_OFFSET
NUM_TAGS = 32002    # KEEP + DELETE + 32000 token IDs

TRAIN_CONFIG = {
    "lr": 3e-4,
    "warmup_steps": 2000,
    "total_steps": 100000,
    "batch_size": 64,
    "gradient_accumulation_steps": 2,  # effective 128
    "seq_length": 256,
    "weight_decay": 0.01,
    "adam_betas": (0.9, 0.98),
    "fp16": True,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "log_interval": 100,
    "save_interval": 10000,
    "eval_interval": 5000,
    "corruption_rate_min": 0.02,
    "corruption_rate_max": 0.25,
    "identity_rate": 0.20,
    "keep_loss_weight": 0.3,
}

MODEL_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 512,
    "num_layers": 6,
    "num_heads": 8,
    "intermediate_size": 2048,
    "max_seq_length": 256,
    "num_tags": NUM_TAGS,
    "dropout": 0.1,
}

PAD_ID = 0


class TaggerV2Dataset(IterableDataset):
    """Dataset for v2 tagger: tags are KEEP/DELETE/token_id.

    For each (corrupted, clean) pair, aligns at the subword token level:
    - If corrupted token == clean token at same position: KEEP
    - If clean has fewer tokens: DELETE
    - If tokens differ: REPLACE with clean token id
    """

    def __init__(self, data_dir, tokenizer_path, corruption_engine,
                 max_seq_length=256, corruption_rate_min=0.02,
                 corruption_rate_max=0.25, identity_rate=0.20, seed=42):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.corruption_rate_min = corruption_rate_min
        self.corruption_rate_max = corruption_rate_max
        self.identity_rate = identity_rate
        self.seed = seed

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(tokenizer_path))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed)
        shards = sorted(self.data_dir.glob("clean_*.txt"))
        rng.shuffle(shards)

        worker_seed = self.seed
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]
            worker_seed = self.seed + worker_info.id
            rng = random.Random(worker_seed)

        engine = CorruptionEngine(seed=worker_seed)

        for shard in shards:
            buffer = []
            with open(shard) as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line.split()) < 4:
                        continue
                    buffer.append(line)
                    if len(buffer) >= 5000:
                        rng.shuffle(buffer)
                        yield from self._process_lines(buffer, rng, engine)
                        buffer = []
            if buffer:
                rng.shuffle(buffer)
                yield from self._process_lines(buffer, rng, engine)

    def _process_lines(self, lines, rng, engine):
        for line in lines:
            if rng.random() < self.identity_rate:
                corrupted = line
            else:
                rate = rng.uniform(self.corruption_rate_min, self.corruption_rate_max)
                corrupted = engine.corrupt_sentence(line, rate)

            corrupt_ids = self.sp.Encode(corrupted)
            clean_ids = self.sp.Encode(line)

            if len(corrupt_ids) < 3:
                continue

            # Align at token level using DP
            tags = self._align_tokens(corrupt_ids, clean_ids)
            if tags is None:
                continue

            # Truncate and pad
            max_len = self.max_seq_length
            corrupt_ids = corrupt_ids[:max_len]
            tags = tags[:max_len]
            seq_len = len(corrupt_ids)
            pad_len = max_len - seq_len

            yield {
                "input_ids": torch.tensor(corrupt_ids + [PAD_ID] * pad_len, dtype=torch.long),
                "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
                "tag_ids": torch.tensor(tags + [-100] * pad_len, dtype=torch.long),
            }

    def _align_tokens(self, corrupt_ids, clean_ids):
        """Align corrupted and clean token sequences, return tag per corrupt token.

        Uses DP alignment. For each corrupted token:
        - Matched with same clean token: KEEP (tag 0)
        - Matched with different clean token: REPLACE (tag = clean_id + 2)
        - No match in clean: DELETE (tag 1)
        """
        m, n = len(corrupt_ids), len(clean_ids)
        if m == 0:
            return None

        # Fast path: same length, compare directly
        if m == n:
            tags = []
            for c, t in zip(corrupt_ids, clean_ids):
                if c == t:
                    tags.append(KEEP_TAG)
                else:
                    tags.append(t + REPLACE_OFFSET)
            return tags

        # DP alignment for different lengths
        INF = float('inf')
        dp = [[INF] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = i  # delete all
        for j in range(1, n + 1):
            dp[0][j] = j  # insert all (but we can't tag these)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if corrupt_ids[i-1] == clean_ids[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # match
                else:
                    dp[i][j] = dp[i-1][j-1] + 1  # replace
                dp[i][j] = min(dp[i][j], dp[i-1][j] + 1)  # delete from corrupt
                dp[i][j] = min(dp[i][j], dp[i][j-1] + 1)  # skip clean (insert)

        # Backtrack
        tags = [KEEP_TAG] * m
        i, j = m, n
        while i > 0 and j > 0:
            if corrupt_ids[i-1] == clean_ids[j-1] and dp[i][j] == dp[i-1][j-1]:
                tags[i-1] = KEEP_TAG
                i -= 1; j -= 1
            elif dp[i][j] == dp[i-1][j-1] + 1:
                tags[i-1] = clean_ids[j-1] + REPLACE_OFFSET
                i -= 1; j -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                tags[i-1] = DELETE_TAG
                i -= 1
            else:
                j -= 1  # insertion in clean, skip
        while i > 0:
            tags[i-1] = DELETE_TAG
            i -= 1

        return tags


def get_gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TRAIN_CONFIG
    mcfg = MODEL_CONFIG
    accum_steps = cfg["gradient_accumulation_steps"]

    model = SpellTagger(**mcfg).to(device)
    print(f"Tagger v2: {model.count_parameters():,} params, {NUM_TAGS} tags (KEEP+DELETE+32K vocab)")

    if cfg["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        betas=cfg["adam_betas"], weight_decay=cfg["weight_decay"],
    )
    scheduler = WSDScheduler(
        optimizer, peak_lr=cfg["lr"],
        warmup_steps=cfg["warmup_steps"], total_steps=cfg["total_steps"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"])

    # Class weights: downweight KEEP
    class_weights = torch.ones(NUM_TAGS, device=device)
    class_weights[KEEP_TAG] = cfg["keep_loss_weight"]

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt.get("step", 0)
        for _ in range(ckpt.get("scheduler_step", step)):
            scheduler.step()
        print(f"Resumed from step {step}")

    compiled = False
    if hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)
        compiled = True

    engine = CorruptionEngine(seed=42)
    dataset = TaggerV2Dataset(
        data_dir=args.data_dir, tokenizer_path=args.tokenizer,
        corruption_engine=engine, max_seq_length=mcfg["max_seq_length"],
        corruption_rate_min=cfg["corruption_rate_min"],
        corruption_rate_max=cfg["corruption_rate_max"],
        identity_rate=cfg["identity_rate"],
    )
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_dir / "metrics.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "lr", "tokens_per_sec", "elapsed_sec",
                         "accuracy", "keep_acc", "edit_acc", "gpu_mem_gb"])

    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({"model": mcfg, "train": cfg}, f, indent=2)

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_keep_correct = 0
    running_keep_total = 0
    running_edit_correct = 0
    running_edit_total = 0
    model.train()
    start_time = time.time()
    tokens_seen = 0
    micro_step = 0

    print(f"\nStarting Tagger v2 Training (from step {step})")
    print(f"  Batch: {cfg['batch_size']} x {accum_steps} = {cfg['batch_size'] * accum_steps}")
    print(f"  Steps: {cfg['total_steps']:,}")
    print(f"  KEEP weight: {cfg['keep_loss_weight']}, Identity: {cfg['identity_rate']}")
    print(f"  Compiled: {compiled}")
    print()

    for batch in dataloader:
        if step >= cfg["total_steps"]:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tag_ids = batch["tag_ids"].to(device)
        batch_tokens = attention_mask.sum().item()

        with torch.amp.autocast("cuda", enabled=cfg["fp16"]):
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, NUM_TAGS), tag_ids.view(-1),
                weight=class_weights, ignore_index=-100,
            )
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = tag_ids != -100
            running_correct += (preds[mask] == tag_ids[mask]).sum().item()
            running_total += mask.sum().item()
            keep_mask = mask & (tag_ids == KEEP_TAG)
            edit_mask = mask & (tag_ids != KEEP_TAG)
            running_keep_correct += (preds[keep_mask] == tag_ids[keep_mask]).sum().item()
            running_keep_total += keep_mask.sum().item()
            running_edit_correct += (preds[edit_mask] == tag_ids[edit_mask]).sum().item()
            running_edit_total += edit_mask.sum().item()

        running_loss += loss.item() * accum_steps
        tokens_seen += batch_tokens
        micro_step += 1

        if micro_step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr = scheduler.step()
            step += 1

            if step % cfg["log_interval"] == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / (cfg["log_interval"] * accum_steps)
                acc = running_correct / running_total if running_total > 0 else 0
                keep_acc = running_keep_correct / running_keep_total if running_keep_total > 0 else 0
                edit_acc = running_edit_correct / running_edit_total if running_edit_total > 0 else 0
                tps = tokens_seen / elapsed if elapsed > 0 else 0
                gpu_mem = get_gpu_memory_gb()

                print(
                    f"Step {step:>7d}/{cfg['total_steps']} | "
                    f"Loss: {avg_loss:.4f} | Acc: {acc:.3f} | "
                    f"KEEP: {keep_acc:.3f} | EDIT: {edit_acc:.3f} | "
                    f"LR: {lr:.2e} | tok/s: {tps:,.0f} | GPU: {gpu_mem:.1f}GB"
                )
                csv_writer.writerow([
                    step, f"{avg_loss:.6f}", f"{lr:.2e}", f"{tps:.0f}", f"{elapsed:.1f}",
                    f"{acc:.4f}", f"{keep_acc:.4f}", f"{edit_acc:.4f}", f"{gpu_mem:.2f}",
                ])
                csv_file.flush()
                running_loss = running_correct = running_total = 0
                running_keep_correct = running_keep_total = 0
                running_edit_correct = running_edit_total = 0

            if step % cfg["save_interval"] == 0:
                ckpt_path = checkpoint_dir / f"step_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_step": scheduler._step,
                    "model_config": mcfg,
                }, ckpt_path)
                print(f"Saved: {ckpt_path}")

    final_path = checkpoint_dir / "best.pt"
    torch.save({
        "step": step,
        "model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
        "model_config": mcfg,
    }, final_path)
    elapsed = time.time() - start_time
    print(f"\nDone. {step} steps in {elapsed/3600:.1f}h")
    csv_file.close()


def main():
    parser = argparse.ArgumentParser(description="Train tagger v2 (32K vocab tags)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/tagger_v2"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
