from datasets import Dataset, load_dataset

from judge.evaluators import PerplexityEvaluator
from judge.models import GradingType, MetricDirection, Resources
from judge.tasks.model import ModelEvaluationTask

DATASET = "allenai/c4"
SUBSET = "en"
SPLIT = "validation"
VALIDATION_SAMPLES = 100_000


class Lab4(ModelEvaluationTask):
    grading_type = GradingType.SCORE
    id = "lab4"
    resources = Resources(cpus=8, memory_gb=32, gpus=1, timeout_seconds=4 * 3600)
    primary_metric = "score"
    metric_direction = MetricDirection.MINIMIZE
    evaluator = PerplexityEvaluator(tokenizer_id="openai-community/gpt2")

    def load_dataset(self) -> Dataset:
        print("[lab4] loading C4 en validation", flush=True)
        # Restrict files so preparation does not download the training split.
        dataset = load_dataset(
            DATASET,
            SUBSET,
            data_files={SPLIT: "en/c4-validation.*.json.gz"},
            split=SPLIT,
            verification_mode="no_checks",
        )
        return dataset.shuffle(seed=42).select(range(VALIDATION_SAMPLES))
