"""Phase 1: Masked Language Model pretraining on clean text."""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.architecture import NeuralSpellModel
from training.dataset import MLMDataset
from training.scheduler import WSDScheduler

PRETRAIN_CONFIG = {
    "lr": 1e-4,
    "warmup_steps": 10000,
    "total_steps": 500000,
    "batch_size": 128,
    "seq_length": 256,
    "mask_prob": 0.15,
    "weight_decay": 0.01,
    "adam_betas": (0.9, 0.999),
    "fp16": True,
    "gradient_checkpointing": True,
    "log_interval": 100,
    "save_interval": 10000,
    "eval_interval": 5000,
}


class TrainingLogger:
    """Log metrics to CSV and optionally TensorBoard."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "loss", "lr", "tokens_per_sec", "elapsed_sec"])

        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
        except ImportError:
            print("TensorBoard not available — install tensorboard for live graphs")

    def log(self, step: int, loss: float, lr: float, tokens_per_sec: float, elapsed: float):
        self.csv_writer.writerow([step, f"{loss:.6f}", f"{lr:.2e}", f"{tokens_per_sec:.0f}", f"{elapsed:.1f}"])
        self.csv_file.flush()

        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", loss, step)
            self.tb_writer.add_scalar("train/lr", lr, step)
            self.tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)

    def close(self):
        self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NeuralSpellModel().to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PRETRAIN_CONFIG["lr"],
        betas=PRETRAIN_CONFIG["adam_betas"],
        weight_decay=PRETRAIN_CONFIG["weight_decay"],
    )

    scheduler = WSDScheduler(
        optimizer,
        peak_lr=PRETRAIN_CONFIG["lr"],
        warmup_steps=PRETRAIN_CONFIG["warmup_steps"],
        total_steps=PRETRAIN_CONFIG["total_steps"],
    )

    dataset = MLMDataset(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        max_seq_length=PRETRAIN_CONFIG["seq_length"],
        mask_prob=PRETRAIN_CONFIG["mask_prob"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=PRETRAIN_CONFIG["batch_size"],
        num_workers=4,
        pin_memory=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=PRETRAIN_CONFIG["fp16"])
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(checkpoint_dir / "logs")

    # Resume from checkpoint if available
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt.get("step", 0)
        # Advance scheduler to correct position
        for _ in range(step):
            scheduler.step()
        print(f"Resumed from step {step}")

    running_loss = 0.0
    model.train()
    start_time = time.time()
    tokens_seen = 0

    print(f"Starting Phase 1: MLM Pretraining (from step {step})")
    for batch in dataloader:
        if step >= PRETRAIN_CONFIG["total_steps"]:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_tokens = attention_mask.sum().item()

        with torch.amp.autocast("cuda", enabled=PRETRAIN_CONFIG["fp16"]):
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr = scheduler.step()

        running_loss += loss.item()
        tokens_seen += batch_tokens
        step += 1

        if step % PRETRAIN_CONFIG["log_interval"] == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / PRETRAIN_CONFIG["log_interval"]
            tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
            print(f"Step {step:>6d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | tok/s: {tokens_per_sec:,.0f} | {elapsed/3600:.1f}h")
            logger.log(step, avg_loss, lr, tokens_per_sec, elapsed)
            running_loss = 0.0

        if step % PRETRAIN_CONFIG["save_interval"] == 0:
            ckpt_path = checkpoint_dir / f"step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final
    final_path = checkpoint_dir / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "step": step}, final_path)
    print(f"Training complete. Final model saved to {final_path}")
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: MLM Pretraining")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pretrain"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
