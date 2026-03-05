"""Phase 2: Corruption fine-tuning on synthetic error pairs."""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sentencepiece as spm

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
    "eval_interval": 2000,
}


class TrainingLogger:
    """Log metrics and correction samples."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "metrics.csv"
        self.samples_path = log_dir / "samples.log"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "loss", "lr", "tokens_per_sec", "elapsed_sec", "token_accuracy"])
        self.samples_file = open(self.samples_path, "w")

        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
        except ImportError:
            pass

    def log(self, step: int, loss: float, lr: float, tps: float, elapsed: float, token_acc: float):
        self.csv_writer.writerow([step, f"{loss:.6f}", f"{lr:.2e}", f"{tps:.0f}", f"{elapsed:.1f}", f"{token_acc:.4f}"])
        self.csv_file.flush()
        if self.tb_writer:
            self.tb_writer.add_scalar("finetune/loss", loss, step)
            self.tb_writer.add_scalar("finetune/lr", lr, step)
            self.tb_writer.add_scalar("finetune/token_accuracy", token_acc, step)
            self.tb_writer.add_scalar("finetune/tokens_per_sec", tps, step)

    def log_samples(self, step: int, samples: list[dict]):
        lines = [f"\n{'='*60}", f"Step {step} — Correction Samples", f"{'='*60}"]
        tb_text = ""
        for i, s in enumerate(samples):
            lines.append(f"\n  Corrupted: {s['corrupted']}")
            lines.append(f"  Predicted: {s['predicted']}")
            lines.append(f"  Original:  {s['original']}")
            changed = []
            for j, (c, p, o) in enumerate(zip(s['corrupt_tokens'], s['pred_tokens'], s['clean_tokens'])):
                if c != o:
                    status = "FIXED" if p == o else f"WRONG({p})"
                    changed.append(f"    {c!r} -> {p!r} (wanted {o!r}) [{status}]")
            if changed:
                lines.append(f"  Changes:")
                lines.extend(changed)
            else:
                lines.append(f"  (no corruptions in this sample)")
            tb_text += f"**Sample {i+1}**  \n"
            tb_text += f"Corrupted: `{s['corrupted']}`  \n"
            tb_text += f"Predicted: `{s['predicted']}`  \n"
            tb_text += f"Original: `{s['original']}`  \n\n"

        text = "\n".join(lines)
        print(text)
        self.samples_file.write(text + "\n")
        self.samples_file.flush()
        if self.tb_writer:
            self.tb_writer.add_text("finetune/samples", tb_text, step)

    def close(self):
        self.csv_file.close()
        self.samples_file.close()
        if self.tb_writer:
            self.tb_writer.close()


def generate_correction_samples(model, batch, sp, device, n=3):
    """Generate sample corrections for logging."""
    model.eval()
    samples = []

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds = logits.argmax(dim=-1)

    for i in range(batch["input_ids"].size(0)):
        if len(samples) >= n:
            break
        seq_len = batch["attention_mask"][i].sum().item()
        input_ids = batch["input_ids"][i][:seq_len].tolist()
        label_ids = batch["labels"][i][:seq_len].tolist()
        pred_ids = preds[i][:seq_len].tolist()

        # Skip identity pairs — only show samples with actual corruptions
        if input_ids == label_ids:
            continue

        corrupted_text = sp.Decode(input_ids)[:120]
        predicted_text = sp.Decode(pred_ids)[:120]
        original_text = sp.Decode(label_ids)[:120]

        corrupt_tokens = [sp.IdToPiece(t) for t in input_ids]
        pred_tokens = [sp.IdToPiece(t) for t in pred_ids]
        clean_tokens = [sp.IdToPiece(t) for t in label_ids]

        samples.append({
            "corrupted": corrupted_text,
            "predicted": predicted_text,
            "original": original_text,
            "corrupt_tokens": corrupt_tokens,
            "pred_tokens": pred_tokens,
            "clean_tokens": clean_tokens,
        })

    model.train()
    return samples


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NeuralSpellModel().to(device)

    # Load pretrained weights
    print(f"Loading pretrained model from {args.pretrained}...")
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    # Strip torch.compile _orig_mod. prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Loaded from step {ckpt.get('step', '?')}")

    # Resume from finetune checkpoint (before compile, so keys are clean)
    step = 0
    if args.resume:
        rckpt = torch.load(args.resume, map_location=device, weights_only=True)
        rstate = rckpt["model_state_dict"]
        rstate = {k.replace("_orig_mod.", ""): v for k, v in rstate.items()}
        model.load_state_dict(rstate)
        step = rckpt.get("step", 0)
        print(f"Resumed from step {step}")

    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))

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

    # Restore optimizer state and advance scheduler after optimizer creation
    if args.resume and 'rckpt' in locals():
        if "optimizer_state_dict" in rckpt:
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
        for _ in range(step):
            scheduler.step()

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

    logger = TrainingLogger(checkpoint_dir / "logs")

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    model.train()
    start_time = time.time()
    tokens_seen = 0

    print(f"Starting Phase 2: Corruption Fine-tuning (from step {step})")
    for batch in dataloader:
        if step >= FINETUNE_CONFIG["total_steps"]:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_tokens = attention_mask.sum().item()

        with torch.amp.autocast("cuda", enabled=FINETUNE_CONFIG["fp16"]):
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Track token-level accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            running_correct += (preds[mask] == labels[mask]).sum().item()
            running_total += mask.sum().item()

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

        if step % FINETUNE_CONFIG["log_interval"] == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / FINETUNE_CONFIG["log_interval"]
            token_acc = running_correct / running_total if running_total > 0 else 0
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            print(f"Step {step:>6d} | Loss: {avg_loss:.4f} | Acc: {token_acc:.3f} | LR: {lr:.2e} | tok/s: {tps:,.0f} | {elapsed/3600:.1f}h")
            logger.log(step, avg_loss, lr, tps, elapsed, token_acc)
            running_loss = 0.0
            running_correct = 0
            running_total = 0

        if step % FINETUNE_CONFIG["eval_interval"] == 0:
            samples = generate_correction_samples(model, batch, sp, device, n=3)
            if samples:
                logger.log_samples(step, samples)

        if step % FINETUNE_CONFIG["save_interval"] == 0:
            ckpt_path = checkpoint_dir / f"step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = checkpoint_dir / "best.pt"
    torch.save({"model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}, "step": step}, final_path)
    print(f"Fine-tuning complete. Final model saved to {final_path}")
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Corruption Fine-tuning")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/finetune"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
