from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset

from judge.models import JudgeResult
from judge.tasks.lab4 import Lab4


def test_lab4_keeps_c4_validation_selection():
    assert Lab4.evaluator.tokenizer_id == "openai-community/gpt2"
    dataset = Dataset.from_dict({"text": [str(i) for i in range(7)]})
    with (
        patch("judge.tasks.lab4.load_dataset", return_value=dataset) as load,
        patch("judge.tasks.lab4.VALIDATION_SAMPLES", 3),
    ):
        selected = Lab4().load_dataset()
    load.assert_called_once_with(
        "allenai/c4",
        "en",
        data_files={"validation": "en/c4-validation.*.json.gz"},
        split="validation",
        verification_mode="no_checks",
    )
    assert selected["text"] == dataset.shuffle(seed=42).select(range(3))["text"]


def test_lab4_delegates_participant_model_to_evaluator():
    dataset = Dataset.from_dict({"text": ["document"]})
    expected = JudgeResult(passed=True, score=2.0)
    with (
        patch(
            "judge.tasks.model.load_student_function",
            return_value=SimpleNamespace(eval_model_id="student/lab4-model"),
        ) as load,
        patch("judge.tasks.model.torch.set_num_threads"),
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch.object(Lab4, "load_dataset", return_value=dataset),
        patch(
            "judge.evaluators.PerplexityEvaluator.evaluate", return_value=expected
        ) as evaluate,
    ):
        result = Lab4().evaluate(Path("submission"))
    assert result is expected
    load.assert_called_once_with(Path("submission"), "lab4")
    evaluate.assert_called_once_with("student/lab4-model", dataset)
