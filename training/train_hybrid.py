"""Hybrid training: aspell-augmented encoder-decoder.

The encoder sees BOTH the corrupted text AND aspell's correction attempt,
separated by [SEP]. The model learns to:
  1. Trust aspell when it's right (most typos)
  2. Override aspell when it's wrong (homophones, context errors)
  3. Generate its own fix when aspell has no suggestion

This is a much easier learning problem than generating corrections from
scratch — the model focuses on context-dependent selection rather than
memorizing a dictionary.

Model: 60M params (512 hidden, 6+6 layers, 8 heads)
Input: "corrupted text [SEP] aspell corrected text"
Output: clean text

Usage:
    PYTHONPATH=. python training/train_hybrid.py \
        --data-dir data/processed \
        --tokenizer tokenizer/tokenizer.model \
        --checkpoint-dir checkpoints/hybrid
"""

import argparse
import csv
import json
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import sentencepiece as spm

from model.architecture import NeuralSpellModel
from corruption.engine import CorruptionEngine
from training.scheduler import WSDScheduler

# ─── Model config: 60M params ───────────────────────────────────

MODEL_CONFIG = {
    "hidden_size": 512,
    "encoder_layers": 6,
    "decoder_layers": 6,
    "num_heads": 8,
    "intermediate_size": 2048,
    "max_seq_length": 512,  # larger to fit corrupted + aspell
    "vocab_size": 32000,
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "lr": 3e-4,
    "warmup_steps": 2000,
    "total_steps": 100000,
    "batch_size": 32,
    "gradient_accumulation_steps": 4,  # effective batch = 128
    "weight_decay": 0.01,
    "adam_betas": (0.9, 0.98),
    "fp16": True,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "log_interval": 100,
    "save_interval": 10000,
    "eval_interval": 5000,
    # Realistic error distribution
    "corruption_rate_min": 0.02,
    "corruption_rate_max": 0.25,
    "identity_rate": 0.15,
}

BOS_ID = 2
EOS_ID = 3
PAD_ID = 0


# ─── Aspell integration ─────────────────────────────────────────

def aspell_correct_sentence(sentence: str) -> str:
    """Run aspell on a sentence, return corrected version."""
    try:
        result = subprocess.run(
            ["aspell", "-a", "--lang=en"],
            input=sentence, capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return sentence

    words = sentence.split()
    corrected = list(words)
    suggestions = {}

    for line in result.stdout.split("\n"):
        if line.startswith("&"):
            parts = line.split()
            original = parts[1]
            sugg = line.split(": ", 1)
            if len(sugg) > 1:
                suggestions[original] = sugg[1].split(",")[0].strip()

    for i, w in enumerate(corrected):
        clean = re.sub(r"[^\w'-]", "", w)
        if clean in suggestions:
            corrected[i] = w.replace(clean, suggestions[clean])

    return " ".join(corrected)


# ─── Dataset ─────────────────────────────────────────────────────

class HybridDataset(IterableDataset):
    """Dataset that feeds corrupted + aspell suggestion to encoder.

    Encoder input: "corrupted text [SEP] aspell corrected text"
    Decoder target: clean text

    The model learns when to trust aspell vs override with context.
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer_path: Path,
        corruption_engine: CorruptionEngine,
        max_seq_length: int = 512,
        corruption_rate_min: float = 0.02,
        corruption_rate_max: float = 0.25,
        identity_rate: float = 0.15,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.corruption_rate_min = corruption_rate_min
        self.corruption_rate_max = corruption_rate_max
        self.identity_rate = identity_rate
        self.seed = seed

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(tokenizer_path))
        self.sep_id = self.sp.PieceToId("[SEP]")

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
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
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
            clean_ids = self.sp.Encode(line)
            if len(clean_ids) < 5:
                continue

            # Identity pair or corrupted pair
            if rng.random() < self.identity_rate:
                corrupted = line
            else:
                rate = rng.uniform(self.corruption_rate_min, self.corruption_rate_max)
                corrupted = engine.corrupt_sentence(line, rate)

            # Run aspell on corrupted text
            aspell_fixed = aspell_correct_sentence(corrupted)

            # Encode: "corrupted [SEP] aspell_fixed"
            corrupt_ids = self.sp.Encode(corrupted)
            aspell_ids = self.sp.Encode(aspell_fixed)

            # Budget: half for each, with SEP in between
            max_half = (self.max_seq_length - 1) // 2  # -1 for SEP
            corrupt_ids = corrupt_ids[:max_half]
            aspell_ids = aspell_ids[:max_half]
            encoder_ids = corrupt_ids + [self.sep_id] + aspell_ids

            yield self._build_sample(encoder_ids, clean_ids)

    def _build_sample(self, encoder_ids, clean_ids):
        max_len = self.max_seq_length

        encoder_ids = encoder_ids[:max_len]
        enc_len = len(encoder_ids)
        enc_pad = max_len - enc_len
        encoder_input_ids = encoder_ids + [PAD_ID] * enc_pad
        encoder_attention_mask = [1] * enc_len + [0] * enc_pad

        dec_input = [BOS_ID] + clean_ids[:max_len - 1]
        labels = clean_ids[:max_len - 1] + [EOS_ID]

        dec_len = len(dec_input)
        dec_pad = max_len - dec_len
        decoder_input_ids = dec_input + [PAD_ID] * dec_pad
        decoder_attention_mask = [1] * dec_len + [0] * dec_pad
        label_ids = labels + [-100] * dec_pad

        return {
            "encoder_input_ids": torch.tensor(encoder_input_ids, dtype=torch.long),
            "encoder_attention_mask": torch.tensor(encoder_attention_mask, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(decoder_attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


# ─── Training ────────────────────────────────────────────────────

class TrainingLogger:
    def __init__(self, log_dir, resume=False):
        log_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        self.csv_file = open(log_dir / "metrics.csv", mode, newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not resume:
            self.csv_writer.writerow([
                "step", "loss", "lr", "tokens_per_sec", "elapsed_sec",
                "token_accuracy", "gpu_mem_gb", "grad_norm",
            ])
        self.samples_file = open(log_dir / "samples.log", mode)

    def log(self, step, loss, lr, tps, elapsed, token_acc, gpu_mem, grad_norm):
        self.csv_writer.writerow([
            step, f"{loss:.6f}", f"{lr:.2e}", f"{tps:.0f}", f"{elapsed:.1f}",
            f"{token_acc:.4f}", f"{gpu_mem:.2f}", f"{grad_norm:.4f}",
        ])
        self.csv_file.flush()

    def log_samples(self, step, samples):
        lines = [f"\n{'='*60}", f"Step {step} — Correction Samples", f"{'='*60}"]
        for s in samples:
            lines.append(f"\n  Corrupted: {s['corrupted']}")
            lines.append(f"  Aspell:    {s['aspell']}")
            lines.append(f"  Generated: {s['generated']}")
            lines.append(f"  Original:  {s['original']}")
        text = "\n".join(lines)
        print(text)
        self.samples_file.write(text + "\n")
        self.samples_file.flush()

    def close(self):
        self.csv_file.close()
        self.samples_file.close()


def get_gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def strip_compile_prefix(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


@torch.no_grad()
def generate_samples(model, batch, sp, device, n=3):
    model.eval()
    samples = []

    enc_ids = batch["encoder_input_ids"][:n].to(device)
    enc_mask = batch["encoder_attention_mask"][:n].to(device)
    labels = batch["labels"][:n]

    sep_id = sp.PieceToId("[SEP]")

    with torch.amp.autocast("cuda"):
        encoder_output = model.encode(enc_ids, enc_mask)

    for i in range(enc_ids.size(0)):
        enc_len = enc_mask[i].sum().item()
        enc_token_ids = enc_ids[i][:enc_len].tolist()

        # Split encoder input at [SEP]
        if sep_id in enc_token_ids:
            sep_pos = enc_token_ids.index(sep_id)
            corrupted_text = sp.Decode(enc_token_ids[:sep_pos])
            aspell_text = sp.Decode(enc_token_ids[sep_pos + 1:])
        else:
            corrupted_text = sp.Decode(enc_token_ids)
            aspell_text = "(no aspell)"

        label_ids = [t for t in labels[i].tolist() if t != -100 and t != EOS_ID]
        original_text = sp.Decode(label_ids)

        # Greedy decode
        generated = [BOS_ID]
        enc_out_i = encoder_output[i:i+1]
        enc_mask_i = enc_mask[i:i+1]
        for _ in range(128):
            dec_input = torch.tensor([generated], dtype=torch.long, device=device)
            dec_mask = torch.ones_like(dec_input)
            with torch.amp.autocast("cuda"):
                logits = model.decode(dec_input, dec_mask, enc_out_i, enc_mask_i)
            tok = logits[0, -1, :].argmax().item()
            if tok == EOS_ID:
                break
            generated.append(tok)
        generated_text = sp.Decode(generated[1:])

        samples.append({
            "corrupted": corrupted_text[:300],
            "aspell": aspell_text[:300],
            "generated": generated_text[:300],
            "original": original_text[:300],
        })

    model.train()
    return samples


def save_checkpoint(path, step, model, optimizer, scaler, scheduler_step, loss):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model_state_dict": strip_compile_prefix(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_step": scheduler_step,
        "loss": loss,
        "model_config": MODEL_CONFIG,
        "train_config": TRAIN_CONFIG,
    }, tmp)
    tmp.rename(path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TRAIN_CONFIG
    mcfg = MODEL_CONFIG
    accum_steps = cfg["gradient_accumulation_steps"]

    model = NeuralSpellModel(**mcfg).to(device)
    print(f"Hybrid model: {model.count_parameters():,} params")

    if mcfg.get("gradient_checkpointing", cfg.get("gradient_checkpointing")):
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

    # Resume
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
    dataset = HybridDataset(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        corruption_engine=engine,
        max_seq_length=mcfg["max_seq_length"],
        corruption_rate_min=cfg["corruption_rate_min"],
        corruption_rate_max=cfg["corruption_rate_max"],
        identity_rate=cfg["identity_rate"],
    )
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(checkpoint_dir / "logs", resume=(step > 0))

    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({"model": mcfg, "train": cfg}, f, indent=2)

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_grad_norm = 0.0
    model.train()
    start_time = time.time()
    tokens_seen = 0
    micro_step = 0

    print(f"\nStarting Hybrid Training (from step {step})")
    print(f"  Model: {model.count_parameters():,} params")
    print(f"  Batch: {cfg['batch_size']} x {accum_steps} = {cfg['batch_size'] * accum_steps} effective")
    print(f"  Steps: {cfg['total_steps']:,}")
    print(f"  Corruption: {cfg['corruption_rate_min']}-{cfg['corruption_rate_max']}, identity: {cfg['identity_rate']}")
    print(f"  FP16: {cfg['fp16']}, Compiled: {compiled}")
    print()

    for batch in dataloader:
        if step >= cfg["total_steps"]:
            break

        enc_ids = batch["encoder_input_ids"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)
        dec_ids = batch["decoder_input_ids"].to(device)
        dec_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_tokens = dec_mask.sum().item()

        with torch.amp.autocast("cuda", enabled=cfg["fp16"]):
            logits = model(enc_ids, enc_mask, dec_ids, dec_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
            )
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            running_correct += (preds[mask] == labels[mask]).sum().item()
            running_total += mask.sum().item()

        running_loss += loss.item() * accum_steps
        tokens_seen += batch_tokens
        micro_step += 1

        if micro_step % accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"]).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr = scheduler.step()
            running_grad_norm += grad_norm
            step += 1

            if step % cfg["log_interval"] == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / (cfg["log_interval"] * accum_steps)
                token_acc = running_correct / running_total if running_total > 0 else 0
                tps = tokens_seen / elapsed if elapsed > 0 else 0
                gpu_mem = get_gpu_memory_gb()
                avg_gn = running_grad_norm / cfg["log_interval"]
                eta_h = (cfg["total_steps"] - step) * (elapsed / max(step, 1)) / 3600

                print(
                    f"Step {step:>7d}/{cfg['total_steps']} | "
                    f"Loss: {avg_loss:.4f} | Acc: {token_acc:.3f} | "
                    f"LR: {lr:.2e} | tok/s: {tps:,.0f} | "
                    f"GPU: {gpu_mem:.1f}GB | ETA: {eta_h:.1f}h"
                )
                logger.log(step, avg_loss, lr, tps, elapsed, token_acc, gpu_mem, avg_gn)
                running_loss = 0.0
                running_correct = 0
                running_total = 0
                running_grad_norm = 0.0

            if step % cfg["eval_interval"] == 0:
                samples = generate_samples(model, batch, sp, device, n=3)
                if samples:
                    logger.log_samples(step, samples)

            if step % cfg["save_interval"] == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{step}.pt",
                    step, model, optimizer, scaler, scheduler._step, loss.item() * accum_steps,
                )
                print(f"Saved checkpoint: step_{step}.pt")

    # Final save
    save_checkpoint(
        checkpoint_dir / "best.pt",
        step, model, optimizer, scaler, scheduler._step, 0.0,
    )
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed/3600:.1f}h")
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid aspell+NN training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/hybrid"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
