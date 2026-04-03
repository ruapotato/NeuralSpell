#!/usr/bin/env python3
"""Train GECToR-style sequence tagger for spell/grammar correction.

Lean, fast training: 50M param encoder + tagging head.
Predicts per-token edit operations from a fixed vocabulary.

Usage:
    PYTHONPATH=. python training/train_tagger.py \
        --data-dir data/processed \
        --tokenizer tokenizer/tokenizer.model \
        --edit-vocab checkpoints/tagger/edit_vocab.json \
        --checkpoint-dir checkpoints/tagger
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import sentencepiece as spm

from model.tagger import SpellTagger, load_edit_vocab, apply_edits
from corruption.engine import CorruptionEngine
from training.build_edit_vocab import align_and_extract_edits
from training.scheduler import WSDScheduler

TRAIN_CONFIG = {
    "lr": 3e-4,
    "warmup_steps": 2000,
    "total_steps": 100000,
    "batch_size": 128,
    "gradient_accumulation_steps": 1,
    "seq_length": 256,
    "weight_decay": 0.01,
    "adam_betas": (0.9, 0.98),
    "fp16": True,
    "gradient_checkpointing": False,  # small model, not needed
    "max_grad_norm": 1.0,
    "log_interval": 100,
    "save_interval": 10000,
    "eval_interval": 5000,
    "corruption_rate_min": 0.02,
    "corruption_rate_max": 0.25,
    "identity_rate": 0.20,  # 20% clean->clean for restraint
    "keep_loss_weight": 0.3,  # downweight KEEP in loss (class balance)
}

MODEL_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 512,
    "num_layers": 6,
    "num_heads": 8,
    "intermediate_size": 2048,
    "max_seq_length": 256,
    "dropout": 0.1,
}

PAD_ID = 0


class TaggerDataset(IterableDataset):
    """Dataset for sequence tagger training.

    For each sentence:
    1. Generate corrupted version
    2. Tokenize both with SentencePiece
    3. Align tokens and extract edit tags
    4. Yield (input_ids, attention_mask, tag_ids)
    """

    def __init__(self, data_dir, tokenizer_path, edit_vocab, corruption_engine,
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
        self.tag2id = edit_vocab["tag2id"]
        self.keep_id = self.tag2id.get("$KEEP", 0)
        self.num_tags = len(edit_vocab["vocab"])

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

            # Word-level alignment and edit extraction
            edits = align_and_extract_edits(corrupted, line)
            if not edits:
                continue

            # Convert to token-level: tokenize each word, assign the word's
            # edit tag to its first subword token, KEEP to the rest
            input_ids = []
            tag_ids = []
            for word, tag in edits:
                if not word:
                    continue
                tokens = self.sp.Encode(word)
                if not tokens:
                    continue
                # First token gets the edit tag
                tag_id = self.tag2id.get(tag, self.keep_id)
                input_ids.append(tokens[0])
                tag_ids.append(tag_id)
                # Continuation tokens get KEEP
                for t in tokens[1:]:
                    input_ids.append(t)
                    tag_ids.append(self.keep_id)

            if len(input_ids) < 3:
                continue

            # Truncate and pad
            max_len = self.max_seq_length
            input_ids = input_ids[:max_len]
            tag_ids = tag_ids[:max_len]
            seq_len = len(input_ids)
            pad_len = max_len - seq_len

            yield {
                "input_ids": torch.tensor(input_ids + [PAD_ID] * pad_len, dtype=torch.long),
                "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
                "tag_ids": torch.tensor(tag_ids + [-100] * pad_len, dtype=torch.long),
            }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TRAIN_CONFIG
    mcfg = MODEL_CONFIG

    # Load edit vocab
    edit_vocab = load_edit_vocab(args.edit_vocab)
    num_tags = len(edit_vocab["vocab"])
    keep_id = edit_vocab["tag2id"]["$KEEP"]
    print(f"Edit vocabulary: {num_tags} tags")

    mcfg["num_tags"] = num_tags
    model = SpellTagger(**mcfg).to(device)
    print(f"Tagger model: {model.count_parameters():,} params")

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

    # Class weights: downweight KEEP since it's 80%+ of tokens
    class_weights = torch.ones(num_tags, device=device)
    class_weights[keep_id] = cfg["keep_loss_weight"]

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

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))

    engine = CorruptionEngine(seed=42)
    dataset = TaggerDataset(
        data_dir=args.data_dir, tokenizer_path=args.tokenizer,
        edit_vocab=edit_vocab, corruption_engine=engine,
        max_seq_length=mcfg["max_seq_length"],
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
    accum_steps = cfg["gradient_accumulation_steps"]
    micro_step = 0

    print(f"\nStarting Tagger Training (from step {step})")
    print(f"  Model: {model.count_parameters():,} params, {num_tags} edit tags")
    print(f"  Batch: {cfg['batch_size']} x {accum_steps} = {cfg['batch_size'] * accum_steps}")
    print(f"  Steps: {cfg['total_steps']:,}")
    print(f"  KEEP loss weight: {cfg['keep_loss_weight']}")
    print(f"  Identity rate: {cfg['identity_rate']}")
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
                logits.view(-1, num_tags), tag_ids.view(-1),
                weight=class_weights, ignore_index=-100,
            )
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        # Track accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = tag_ids != -100
            running_correct += (preds[mask] == tag_ids[mask]).sum().item()
            running_total += mask.sum().item()
            # KEEP vs non-KEEP accuracy
            keep_mask = mask & (tag_ids == keep_id)
            edit_mask = mask & (tag_ids != keep_id)
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
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

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
                running_loss = 0.0
                running_correct = 0
                running_total = 0
                running_keep_correct = 0
                running_keep_total = 0
                running_edit_correct = 0
                running_edit_total = 0

            if step % cfg["save_interval"] == 0:
                ckpt_path = checkpoint_dir / f"step_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_step": scheduler._step,
                    "model_config": mcfg,
                    "train_config": cfg,
                }, ckpt_path)
                print(f"Saved: {ckpt_path}")

    # Final save
    final_path = checkpoint_dir / "best.pt"
    torch.save({
        "step": step,
        "model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
        "model_config": mcfg,
    }, final_path)
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed/3600:.1f}h")
    csv_file.close()


def main():
    parser = argparse.ArgumentParser(description="Train GECToR-style tagger")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--edit-vocab", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/tagger"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
