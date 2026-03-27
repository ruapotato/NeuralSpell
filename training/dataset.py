"""PyTorch datasets for encoder-decoder training.

Phase 1 (Denoising): Corrupted text -> clean text at moderate corruption rates
Phase 2 (Correction): Corrupted text -> clean text at higher rates, harder types

Key difference from the old encoder-only datasets: NO token-length filtering.
The encoder and decoder operate on independent sequences, so corruptions that
change tokenization (homophones, phonetic rewrites, missing spaces) are now
fully supported.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

import sentencepiece as spm

from corruption.engine import CorruptionEngine

# SentencePiece special token IDs (from tokenizer training config)
BOS_ID = 2
EOS_ID = 3
PAD_ID = 0


class DenoisingDataset(IterableDataset):
    """Denoising dataset for Phase 1 pretraining.

    Generates (corrupted, clean) pairs using the corruption engine.
    Encoder input: corrupted tokens. Decoder target: clean tokens.
    No length-matching constraint — encoder and decoder are independent.
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer_path: Path,
        corruption_engine: CorruptionEngine,
        max_seq_length: int = 256,
        corruption_rate_min: float = 0.10,
        corruption_rate_max: float = 0.20,
        identity_rate: float = 0.10,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.corruption_rate_min = corruption_rate_min
        self.corruption_rate_max = corruption_rate_max
        self.identity_rate = identity_rate
        self.seed = seed
        self._engine_seed = corruption_engine.seed if hasattr(corruption_engine, 'seed') else seed

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

        # Fresh engine per worker for thread safety
        engine = CorruptionEngine(seed=worker_seed)

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
                        yield from self._process_lines(buffer, rng, engine)
                        buffer = []

            if buffer:
                rng.shuffle(buffer)
                yield from self._process_lines(buffer, rng, engine)

    def _process_lines(self, lines: list[str], rng: random.Random, engine: CorruptionEngine):
        for line in lines:
            clean_ids = self.sp.Encode(line)
            if len(clean_ids) < 5:
                continue

            # Identity pair (teaches passthrough) or corrupted pair
            if rng.random() < self.identity_rate:
                corrupted = line
            else:
                rate = rng.uniform(self.corruption_rate_min, self.corruption_rate_max)
                corrupted = engine.corrupt_sentence(line, rate)

            encoder_ids = self.sp.Encode(corrupted)

            yield self._build_sample(encoder_ids, clean_ids)

    def _build_sample(self, encoder_ids: list[int], clean_ids: list[int]) -> dict:
        max_len = self.max_seq_length

        # Truncate encoder input
        encoder_ids = encoder_ids[:max_len]
        enc_len = len(encoder_ids)
        enc_pad = max_len - enc_len
        encoder_input_ids = encoder_ids + [PAD_ID] * enc_pad
        encoder_attention_mask = [1] * enc_len + [0] * enc_pad

        # Decoder input: [BOS] + clean_tokens (teacher forcing, shifted right)
        # Labels: clean_tokens + [EOS]
        # Both truncated to max_len
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


class CorrectionDataset(IterableDataset):
    """Correction dataset for Phase 2 fine-tuning.

    Same structure as DenoisingDataset but with higher corruption rates
    and all corruption types at full weight (homophones, phonetic, etc).
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer_path: Path,
        corruption_engine: CorruptionEngine,
        max_seq_length: int = 256,
        corruption_rate_min: float = 0.15,
        corruption_rate_max: float = 0.40,
        identity_rate: float = 0.05,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.corruption_rate_min = corruption_rate_min
        self.corruption_rate_max = corruption_rate_max
        self.identity_rate = identity_rate
        self.seed = seed
        self._engine_seed = corruption_engine.seed if hasattr(corruption_engine, 'seed') else seed

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
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buffer.append(line)
                    if len(buffer) >= 10000:
                        rng.shuffle(buffer)
                        yield from self._process_lines(buffer, rng, engine)
                        buffer = []

            if buffer:
                rng.shuffle(buffer)
                yield from self._process_lines(buffer, rng, engine)

    def _process_lines(self, lines: list[str], rng: random.Random, engine: CorruptionEngine):
        for line in lines:
            clean_ids = self.sp.Encode(line)
            if len(clean_ids) < 5:
                continue

            if rng.random() < self.identity_rate:
                corrupted = line
            else:
                rate = rng.uniform(self.corruption_rate_min, self.corruption_rate_max)
                corrupted = engine.corrupt_sentence(line, rate)

            encoder_ids = self.sp.Encode(corrupted)

            yield self._build_sample(encoder_ids, clean_ids)

    def _build_sample(self, encoder_ids: list[int], clean_ids: list[int]) -> dict:
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
