"""Phase 2: Corruption fine-tuning on synthetic error pairs."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.architecture import NeuralSpellModel
from corruption.engine import CorruptionEngine
from training.dataset import CorrectionDataset
from training.scheduler import WSDScheduler

FINETUNE_CONFIG = {
    "lr": 3e-5,
    "warmup_steps": 1000,
    "total_steps": 200000,
    "batch_size": 64,
    "seq_length": 256,
    "weight_decay": 0.01,
    "fp16": True,
    "log_interval": 100,
    "save_interval": 10000,
}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NeuralSpellModel().to(device)

    # Load pretrained weights
    print(f"Loading pretrained model from {args.pretrained}...")
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded from step {ckpt.get('step', '?')}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_CONFIG["lr"],
        weight_decay=FINETUNE_CONFIG["weight_decay"],
    )

    scheduler = WSDScheduler(
        optimizer,
        peak_lr=FINETUNE_CONFIG["lr"],
        warmup_steps=FINETUNE_CONFIG["warmup_steps"],
        total_steps=FINETUNE_CONFIG["total_steps"],
    )

    engine = CorruptionEngine(seed=42)
    dataset = CorrectionDataset(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        corruption_engine=engine,
        max_seq_length=FINETUNE_CONFIG["seq_length"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=FINETUNE_CONFIG["batch_size"],
        num_workers=4,
        pin_memory=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=FINETUNE_CONFIG["fp16"])
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    running_loss = 0.0
    model.train()

    print("Starting Phase 2: Corruption Fine-tuning")
    for batch in dataloader:
        if step >= FINETUNE_CONFIG["total_steps"]:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=FINETUNE_CONFIG["fp16"]):
            logits = model(input_ids, attention_mask)
            # Loss at ALL positions, not just corrupted ones
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
        step += 1

        if step % FINETUNE_CONFIG["log_interval"] == 0:
            avg_loss = running_loss / FINETUNE_CONFIG["log_interval"]
            print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            running_loss = 0.0

        if step % FINETUNE_CONFIG["save_interval"] == 0:
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

    final_path = checkpoint_dir / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "step": step}, final_path)
    print(f"Fine-tuning complete. Final model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Corruption Fine-tuning")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/finetune"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
