"""PyTorch datasets for both training phases.

Phase 1 (MLM): Clean text with random masking
Phase 2 (Correction): Corrupted/clean sentence pairs
"""

import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

import sentencepiece as spm


class MLMDataset(IterableDataset):
    """Masked Language Modeling dataset for Phase 1 pretraining.

    Streams sentences from shard files, tokenizes, and applies
    the standard MLM masking strategy:
      - 15% of tokens selected for prediction
      - Of those: 80% [MASK], 10% random, 10% unchanged

    Supports multi-worker DataLoader by splitting shards across workers.
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer_path: Path,
        max_seq_length: int = 256,
        mask_prob: float = 0.15,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.seed = seed

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(tokenizer_path))
        self.mask_id = self.sp.PieceToId("[MASK]")
        self.vocab_size = self.sp.GetPieceSize()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed)
        shards = sorted(self.data_dir.glob("clean_*.txt"))
        rng.shuffle(shards)

        # Split shards across workers
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]
            rng = random.Random(self.seed + worker_info.id)

        for shard in shards:
            # Stream line by line, collect into buffer for shuffling
            buffer = []
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buffer.append(line)
                    # Shuffle and yield in chunks to limit memory
                    if len(buffer) >= 10000:
                        rng.shuffle(buffer)
                        yield from self._process_lines(buffer, rng)
                        buffer = []

            if buffer:
                rng.shuffle(buffer)
                yield from self._process_lines(buffer, rng)

    def _process_lines(self, lines: list[str], rng: random.Random):
        for line in lines:
            token_ids = self.sp.Encode(line)
            if len(token_ids) < 5:
                continue
            token_ids = token_ids[: self.max_seq_length - 2]

            cls_id = self.sp.PieceToId("[CLS]")
            sep_id = self.sp.PieceToId("[SEP]")
            token_ids = [cls_id] + token_ids + [sep_id]

            input_ids, labels = self._apply_masking(token_ids, rng)

            pad_len = self.max_seq_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [0] * pad_len
            labels = labels + [-100] * pad_len

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    def _apply_masking(
        self, token_ids: list[int], rng: random.Random
    ) -> tuple[list[int], list[int]]:
        """Apply MLM masking: 15% of tokens, 80/10/10 split."""
        input_ids = list(token_ids)
        labels = [-100] * len(token_ids)

        for i in range(1, len(token_ids) - 1):  # skip [CLS] and [SEP]
            if rng.random() < self.mask_prob:
                labels[i] = token_ids[i]
                r = rng.random()
                if r < 0.8:
                    input_ids[i] = self.mask_id
                elif r < 0.9:
                    input_ids[i] = rng.randint(4, self.vocab_size - 1)
                # else: keep original (10%)

        return input_ids, labels


class CorrectionDataset(IterableDataset):
    """Correction dataset for Phase 2 fine-tuning.

    Generates (corrupted, clean) sentence pairs on-the-fly using
    the corruption engine. Loss is computed at ALL positions.
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer_path: Path,
        corruption_engine,
        max_seq_length: int = 256,
        corruption_rate: float = 0.15,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.corruption_rate = corruption_rate
        self.engine = corruption_engine
        self.seed = seed

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(tokenizer_path))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed)
        shards = sorted(self.data_dir.glob("clean_*.txt"))
        rng.shuffle(shards)

        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]
            rng = random.Random(self.seed + worker_info.id)

        for shard in shards:
            buffer = []
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buffer.append(line)
                    if len(buffer) >= 10000:
                        rng.shuffle(buffer)
                        yield from self._process_lines(buffer, rng)
                        buffer = []

            if buffer:
                rng.shuffle(buffer)
                yield from self._process_lines(buffer, rng)

    def _process_lines(self, lines: list[str], rng: random.Random):
        for line in lines:
            rate = rng.uniform(0.05, 0.30)
            corrupted = self.engine.corrupt_sentence(line, rate)

            input_ids = self.sp.Encode(corrupted)
            target_ids = self.sp.Encode(line)

            if len(input_ids) < 5 or len(target_ids) < 5:
                continue

            input_ids = input_ids[: self.max_seq_length]
            target_ids = target_ids[: self.max_seq_length]

            pad_len_in = self.max_seq_length - len(input_ids)
            pad_len_tgt = self.max_seq_length - len(target_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len_in
            input_ids = input_ids + [0] * pad_len_in
            target_ids = target_ids + [-100] * pad_len_tgt

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(target_ids, dtype=torch.long),
            }
