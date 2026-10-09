import sys
import traceback
from abc import abstractmethod
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import torch
from datasets import Dataset

from judge.evaluators import Evaluator
from judge.models import JudgeResult, TestResult
from judge.tasks.base import Task, load_student_function


class ModelEvaluationTask(Task):
    """Let a lab choose documents while an evaluator handles model scoring."""

    evaluator: ClassVar[Evaluator]

    @abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and select this lab's evaluation documents."""

    def validate_submission(
        self, module: ModuleType, model_id: str
    ) -> list[TestResult]:
        """Validate lab-specific requirements before preparing evaluation data."""
        return []

    def evaluate(self, submission: Path) -> JudgeResult:
        # Check before downloading data or importing participant code. Runtime
        # failures are infrastructure errors, not participant test failures.
        self.evaluator.validate_runtime()
        torch.set_num_threads(self.resources.cpus)
        print(f"[{self.id}] loading src/labs/{self.id}.py", flush=True)
        sys.path.insert(0, str(submission / "src"))
        try:
            module = load_student_function(submission, self.id)
            model_id = module.eval_model_id
        except Exception as error:  # noqa: BLE001 - participant failures are test failures
            traceback.print_exc(file=sys.stdout)
            return JudgeResult(
                passed=False,
                tests=[
                    TestResult(
                        name="participant_module",
                        passed=False,
                        message=f"{type(error).__name__}: {error}",
                    )
                ],
            )
        finally:
            sys.path.pop(0)

        if not isinstance(model_id, str) or not model_id.strip():
            return JudgeResult(
                passed=False,
                tests=[
                    TestResult(
                        name="model_id",
                        passed=False,
                        message="eval_model_id must be a nonempty Hugging Face model ID",
                    )
                ],
            )
        try:
            checks = self.validate_submission(module, model_id)
        except ValueError as error:
            return JudgeResult(
                passed=False,
                tests=[
                    TestResult(
                        name="submission_requirements", passed=False, message=str(error)
                    )
                ],
            )
        for check in checks:
            print(f"[{self.id}] {check.name}: {check.message}", flush=True)
        result = self.evaluator.evaluate(model_id, self.load_dataset())
        result.tests.extend(checks)
        return result
