"""Evaluation metrics for spell correction."""

from dataclasses import dataclass


@dataclass
class CorrectionMetrics:
    """Metrics for evaluating spell correction quality."""
    total_words: int = 0
    correct_corrections: int = 0    # corrupted word correctly fixed
    missed_errors: int = 0          # corrupted word not fixed
    false_positives: int = 0        # correct word wrongly changed
    correct_passthrough: int = 0    # correct word left alone

    @property
    def precision(self) -> float:
        attempted = self.correct_corrections + self.false_positives
        return self.correct_corrections / attempted if attempted > 0 else 0.0

    @property
    def recall(self) -> float:
        actual_errors = self.correct_corrections + self.missed_errors
        return self.correct_corrections / actual_errors if actual_errors > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        correct_total = self.correct_passthrough + self.false_positives
        return self.false_positives / correct_total if correct_total > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.correct_corrections + self.correct_passthrough) / self.total_words if self.total_words > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Accuracy:    {self.accuracy:.4f}\n"
            f"Precision:   {self.precision:.4f}\n"
            f"Recall:      {self.recall:.4f}\n"
            f"F1:          {self.f1:.4f}\n"
            f"FP Rate:     {self.false_positive_rate:.4f}\n"
            f"Total words: {self.total_words}"
        )
