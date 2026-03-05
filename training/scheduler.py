"""Warmup-Stable-Decay (WSD) learning rate schedule."""

import math


class WSDScheduler:
    """Warmup-Stable-Decay learning rate scheduler.

    Three phases:
    1. Linear warmup from 0 to peak_lr over warmup_steps
    2. Constant at peak_lr for the stable phase
    3. Cosine decay from peak_lr to min_lr over decay phase

    The decay phase starts at (total_steps - decay_steps).
    """

    def __init__(
        self,
        optimizer,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        decay_fraction: float = 0.1,
    ):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.decay_steps = int(total_steps * decay_fraction)
        self.stable_end = total_steps - self.decay_steps
        self._step = 0

    def get_lr(self) -> float:
        if self._step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * self._step / max(1, self.warmup_steps)
        elif self._step < self.stable_end:
            # Stable phase
            return self.peak_lr
        else:
            # Cosine decay
            progress = (self._step - self.stable_end) / max(1, self.decay_steps)
            return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

    def step(self):
        self._step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
