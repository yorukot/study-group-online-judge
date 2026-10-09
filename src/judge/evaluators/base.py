from abc import ABC, abstractmethod

from datasets import Dataset

from judge.models import JudgeResult


class Evaluator(ABC):
    """Score a model independently of a lab's submission and dataset selection."""

    def validate_runtime(self) -> None:
        """Fail early when the evaluator's execution requirements are unavailable."""

    @abstractmethod
    def evaluate(self, model_id: str, dataset: Dataset) -> JudgeResult:
        """Evaluate a Hugging Face model on the supplied documents."""
