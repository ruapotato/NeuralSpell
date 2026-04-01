"""Phase 1: Denoising pretraining for encoder-decoder spell correction.

Trains the model to reconstruct clean text from corrupted input using
the corruption engine. No token-length filtering — encoder and decoder
operate on independent sequences.

Checkpoint/resume is designed for crash resilience on long runs:
saves model, optimizer, scaler, scheduler step, and RNG states.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sentencepiece as spm

from model.architecture import NeuralSpellModel
from corruption.engine import CorruptionEngine
from training.dataset import DenoisingDataset, BOS_ID, EOS_ID
from training.scheduler import WSDScheduler

PRETRAIN_CONFIG = {
    "lr": 3e-4,
    "warmup_steps": 5000,
    "total_steps": 170000,
    "batch_size": 32,
    "gradient_accumulation_steps": 4,  # effective batch = 128
    "seq_length": 256,
    "weight_decay": 0.01,
    "adam_betas": (0.9, 0.98),
    "fp16": True,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "log_interval": 100,
    "save_interval": 10000,
    "eval_interval": 5000,
    "corruption_rate_min": 0.10,
    "corruption_rate_max": 0.20,
    "identity_rate": 0.10,
}


class TrainingLogger:
    """Log metrics and correction samples to CSV, text, and TensorBoard."""

    def __init__(self, log_dir: Path, resume: bool = False):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "metrics.csv"
        self.samples_path = log_dir / "samples.log"

        mode = "a" if resume else "w"
        self.csv_file = open(self.csv_path, mode, newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not resume or not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            self.csv_writer.writerow([
                "step", "loss", "lr", "tokens_per_sec", "elapsed_sec",
                "token_accuracy", "gpu_mem_gb", "grad_norm",
            ])
        self.samples_file = open(self.samples_path, mode)

        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
        except ImportError:
            print("TensorBoard not available — install tensorboard for live graphs")

    def log(self, step: int, loss: float, lr: float, tps: float, elapsed: float,
            token_acc: float, gpu_mem_gb: float, grad_norm: float):
        self.csv_writer.writerow([
            step, f"{loss:.6f}", f"{lr:.2e}", f"{tps:.0f}", f"{elapsed:.1f}",
            f"{token_acc:.4f}", f"{gpu_mem_gb:.2f}", f"{grad_norm:.4f}",
        ])
        self.csv_file.flush()
        if self.tb_writer:
            self.tb_writer.add_scalar("pretrain/loss", loss, step)
            self.tb_writer.add_scalar("pretrain/lr", lr, step)
            self.tb_writer.add_scalar("pretrain/token_accuracy", token_acc, step)
            self.tb_writer.add_scalar("pretrain/tokens_per_sec", tps, step)
            self.tb_writer.add_scalar("pretrain/gpu_mem_gb", gpu_mem_gb, step)
            self.tb_writer.add_scalar("pretrain/grad_norm", grad_norm, step)

    def log_samples(self, step: int, samples: list[dict]):
        lines = [f"\n{'='*60}", f"Step {step} — Correction Samples", f"{'='*60}"]
        tb_text = ""
        for i, s in enumerate(samples):
            lines.append(f"\n  Corrupted: {s['corrupted']}")
            lines.append(f"  Generated: {s['generated']}")
            lines.append(f"  Original:  {s['original']}")
            tb_text += f"**Sample {i+1}**  \n"
            tb_text += f"Corrupted: `{s['corrupted']}`  \n"
            tb_text += f"Generated: `{s['generated']}`  \n"
            tb_text += f"Original: `{s['original']}`  \n\n"

        text = "\n".join(lines)
        print(text)
        self.samples_file.write(text + "\n")
        self.samples_file.flush()
        if self.tb_writer:
            self.tb_writer.add_text("pretrain/samples", tb_text, step)

    def close(self):
        self.csv_file.close()
        self.samples_file.close()
        if self.tb_writer:
            self.tb_writer.close()


def get_gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def strip_compile_prefix(state_dict: dict) -> dict:
    """Remove torch.compile _orig_mod. prefix from state dict keys."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


@torch.no_grad()
def generate_samples(model, batch, sp, device, max_gen_length=128, n=3):
    """Generate correction samples via greedy autoregressive decoding."""
    model.eval()
    samples = []

    enc_ids = batch["encoder_input_ids"][:n].to(device)
    enc_mask = batch["encoder_attention_mask"][:n].to(device)
    labels = batch["labels"][:n]

    with torch.amp.autocast("cuda"):
        encoder_output = model.encode(enc_ids, enc_mask)

    for i in range(enc_ids.size(0)):
        # Decode the encoder input (corrupted text)
        enc_len = enc_mask[i].sum().item()
        enc_token_ids = enc_ids[i][:enc_len].tolist()
        corrupted_text = sp.Decode(enc_token_ids)

        # Decode the label (clean text)
        label_ids = labels[i].tolist()
        label_ids = [t for t in label_ids if t != -100 and t != EOS_ID]
        original_text = sp.Decode(label_ids)

        # Greedy autoregressive generation
        generated = [BOS_ID]
        enc_out_i = encoder_output[i:i+1]
        enc_mask_i = enc_mask[i:i+1]

        for _ in range(max_gen_length):
            dec_input = torch.tensor([generated], dtype=torch.long, device=device)
            dec_mask = torch.ones_like(dec_input)

            with torch.amp.autocast("cuda"):
                logits = model.decode(dec_input, dec_mask, enc_out_i, enc_mask_i)

            next_token = logits[0, -1, :].argmax().item()
            if next_token == EOS_ID:
                break
            generated.append(next_token)

        generated_text = sp.Decode(generated[1:])  # skip BOS

        samples.append({
            "corrupted": corrupted_text[:500],
            "generated": generated_text[:500],
            "original": original_text[:500],
        })

    model.train()
    return samples


def save_checkpoint(
    path: Path, step: int, model, optimizer, scaler, scheduler_step: int,
    loss: float, best_loss: float, rng_state: dict,
):
    """Save a complete, crash-resilient checkpoint."""
    # Write to temp file then rename for atomicity
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model_state_dict": strip_compile_prefix(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_step": scheduler_step,
            "loss": loss,
            "best_loss": best_loss,
            "rng_python": rng_state["python"],
            "rng_numpy": rng_state["numpy"],
            "rng_torch": rng_state["torch"],
            "rng_cuda": rng_state.get("cuda"),
            "config": PRETRAIN_CONFIG,
        },
        tmp_path,
    )
    tmp_path.rename(path)


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_state = state["cuda"]
        if hasattr(cuda_state, "cpu"):
            cuda_state = cuda_state.cpu()
        torch.cuda.set_rng_state(cuda_state)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = PRETRAIN_CONFIG
    accum_steps = cfg["gradient_accumulation_steps"]

    model = NeuralSpellModel().to(device)
    print(f"Parameters: {model.count_parameters():,}")

    if cfg["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # Create optimizer and scheduler BEFORE loading checkpoint or compiling
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=cfg["adam_betas"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = WSDScheduler(
        optimizer,
        peak_lr=cfg["lr"],
        warmup_steps=cfg["warmup_steps"],
        total_steps=cfg["total_steps"],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"])

    # Resume from checkpoint
    step = 0
    best_loss = float("inf")
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        # Restore scheduler position
        sched_step = ckpt.get("scheduler_step", step)
        for _ in range(sched_step):
            scheduler.step()
        # Restore RNG states
        if "rng_torch" in ckpt:
            set_rng_state({
                "python": ckpt["rng_python"],
                "numpy": ckpt["rng_numpy"],
                "torch": ckpt["rng_torch"],
                "cuda": ckpt.get("rng_cuda"),
            })
        print(f"Resumed from step {step} (best_loss={best_loss:.4f})")

    # Compile after loading weights (torch.compile wraps the module)
    compiled = False
    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        compiled = True

    # Tokenizer for sample decoding
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))

    # Dataset and dataloader
    engine = CorruptionEngine(seed=42)
    dataset = DenoisingDataset(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        corruption_engine=engine,
        max_seq_length=cfg["seq_length"],
        corruption_rate_min=cfg["corruption_rate_min"],
        corruption_rate_max=cfg["corruption_rate_max"],
        identity_rate=cfg["identity_rate"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=4,
        pin_memory=True,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(checkpoint_dir / "logs", resume=(step > 0))

    # Save config for reproducibility
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_grad_norm = 0.0
    model.train()
    start_time = time.time()
    tokens_seen = 0
    micro_step = 0

    print(f"\nStarting Phase 1: Denoising Pretraining (from step {step})")
    print(f"  Batch size: {cfg['batch_size']} x {accum_steps} accum = {cfg['batch_size'] * accum_steps} effective")
    print(f"  Total steps: {cfg['total_steps']:,}")
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
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / accum_steps  # scale for accumulation

        scaler.scale(loss).backward()

        # Track token-level accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            running_correct += (preds[mask] == labels[mask]).sum().item()
            running_total += mask.sum().item()

        running_loss += loss.item() * accum_steps  # unscale for logging
        tokens_seen += batch_tokens
        micro_step += 1

        # Optimizer step every accum_steps micro-batches
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
                avg_grad_norm = running_grad_norm / cfg["log_interval"]
                eta_hours = (cfg["total_steps"] - step) / (step / elapsed) / 3600 if step > 0 else 0

                print(
                    f"Step {step:>7d}/{cfg['total_steps']} | "
                    f"Loss: {avg_loss:.4f} | Acc: {token_acc:.3f} | "
                    f"LR: {lr:.2e} | tok/s: {tps:,.0f} | "
                    f"GPU: {gpu_mem:.1f}GB | GradNorm: {avg_grad_norm:.2f} | "
                    f"{elapsed/3600:.1f}h elapsed | ETA: {eta_hours:.1f}h"
                )
                logger.log(step, avg_loss, lr, tps, elapsed, token_acc, gpu_mem, avg_grad_norm)
                running_loss = 0.0
                running_correct = 0
                running_total = 0
                running_grad_norm = 0.0

            if step % cfg["eval_interval"] == 0:
                samples = generate_samples(model, batch, sp, device, n=3)
                if samples:
                    logger.log_samples(step, samples)

            if step % cfg["save_interval"] == 0:
                ckpt_path = checkpoint_dir / f"step_{step}.pt"
                save_checkpoint(
                    ckpt_path, step, model, optimizer, scaler,
                    scheduler._step, loss.item() * accum_steps, best_loss,
                    get_rng_state(),
                )
                print(f"Saved checkpoint: {ckpt_path}")

                # Track best
                recent_loss = running_loss / min(cfg["log_interval"] * accum_steps, 1) if running_loss > 0 else loss.item() * accum_steps
                if recent_loss < best_loss:
                    best_loss = recent_loss
                    best_path = checkpoint_dir / "best.pt"
                    save_checkpoint(
                        best_path, step, model, optimizer, scaler,
                        scheduler._step, recent_loss, best_loss,
                        get_rng_state(),
                    )
                    print(f"New best model (loss={best_loss:.4f})")

    # Save final
    final_path = checkpoint_dir / "best.pt"
    save_checkpoint(
        final_path, step, model, optimizer, scaler,
        scheduler._step, 0.0, best_loss, get_rng_state(),
    )
    elapsed = time.time() - start_time
    print(f"\nPretraining complete. {step} steps in {elapsed/3600:.1f}h")
    print(f"Final model saved to {final_path}")
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Denoising Pretraining")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pretrain"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
