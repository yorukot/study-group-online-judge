import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset

from judge.evaluators import PerplexityEvaluator
from judge.models import JudgeResult
from judge.tasks import TASKS
from judge.tasks.lab4 import Lab4
from judge.tasks.lab5 import Lab5


def test_lab5_selects_tail_after_full_shuffle():
    dataset = Dataset.from_dict({"text": [str(i) for i in range(60_000)]})
    with patch("judge.tasks.lab5.load_dataset", return_value=dataset) as load:
        selected = Lab5().load_dataset()
    load.assert_called_once_with("allenai/dolma3_mix-150B-1025", split="train")
    assert len(selected) == 50_000
    expected = dataset.shuffle(seed=42).select(range(10_000, 60_000))
    assert selected["text"] == expected["text"]
    assert (
        selected["text"]
        != dataset.select(range(10_000, 60_000)).shuffle(seed=42)["text"]
    )
    assert not set(selected["text"]) & set(
        dataset.shuffle(seed=42).select(range(10_000))["text"]
    )


def test_lab5_rejects_insufficient_documents():
    with (
        patch(
            "judge.tasks.lab5.load_dataset",
            return_value=Dataset.from_dict({"text": ["short"]}),
        ),
        pytest.raises(ValueError, match="requires at least 50000 documents"),
    ):
        Lab5().load_dataset()


def test_lab5_is_registered_and_requires_gpu():
    task = TASKS["lab5"]
    assert isinstance(task, Lab5)
    assert task.resources.gpus == 1
    assert task.primary_metric == "score"
    assert task.metric_direction.value == "minimize"
    assert isinstance(task.evaluator, PerplexityEvaluator)
    assert task.evaluator.tokenizer_id == "meta-llama/Llama-3.2-1B"
    assert task.evaluator.batch_size == 512
    assert task.evaluator.dtype == "bfloat16"
    assert task.evaluator.max_length == 8192


def test_lab5_loads_its_own_submission_and_uses_shared_evaluator(tmp_path):
    source = tmp_path / "src" / "labs" / "lab5.py"
    source.parent.mkdir(parents=True)
    source.write_text('eval_model_id = "student/lab5-model"\n')
    dataset = Dataset.from_dict({"text": ["document"]})
    expected = JudgeResult(passed=True, score=2.0)
    with (
        patch("judge.tasks.model.torch.set_num_threads"),
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch.object(Lab5, "load_dataset", return_value=dataset),
        patch(
            "judge.evaluators.PerplexityEvaluator.evaluate", return_value=expected
        ) as evaluate,
    ):
        result = Lab5().evaluate(tmp_path)
    assert result is expected
    evaluate.assert_called_once_with("student/lab5-model", dataset)


@pytest.mark.parametrize("task_type", [Lab4, Lab5])
def test_model_tasks_fail_without_cuda_before_dataset_or_participant_load(task_type):
    with (
        patch("judge.tasks.model.torch.cuda.is_available", return_value=False),
        patch.object(task_type, "load_dataset") as load_data,
        patch("judge.tasks.model.load_student_function") as load_participant,
        pytest.raises(RuntimeError, match="requires a CUDA GPU"),
    ):
        task_type().evaluate(Path("submission"))
    load_data.assert_not_called()
    load_participant.assert_not_called()


@pytest.mark.parametrize("task_type", [Lab4, Lab5])
@pytest.mark.parametrize("model_id", ["", "  ", None, 42])
def test_model_tasks_reject_invalid_model_id_before_dataset_load(task_type, model_id):
    with (
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch("judge.tasks.model.torch.set_num_threads"),
        patch(
            "judge.tasks.model.load_student_function",
            return_value=SimpleNamespace(eval_model_id=model_id),
        ),
        patch.object(task_type, "load_dataset") as load,
    ):
        result = task_type().evaluate(Path("submission"))
    assert not result.passed
    assert result.tests[0].name == "model_id"
    load.assert_not_called()


@pytest.mark.parametrize("task_type", [Lab4, Lab5])
def test_model_tasks_report_participant_failure_and_restore_import_path(task_type):
    original_path = sys.path.copy()
    with (
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch("judge.tasks.model.torch.set_num_threads"),
        patch(
            "judge.tasks.model.load_student_function",
            side_effect=ImportError("bad module"),
        ),
        patch.object(task_type, "load_dataset") as load,
    ):
        result = task_type().evaluate(Path("submission"))
    assert not result.passed
    assert result.tests[0].name == "participant_module"
    assert "bad module" in result.tests[0].message
    assert sys.path == original_path
    load.assert_not_called()
