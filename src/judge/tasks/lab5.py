from datasets import Dataset, load_dataset

from judge.evaluators import PerplexityEvaluator
from judge.models import GradingType, MetricDirection, Resources
from judge.tasks.model import ModelEvaluationTask

DATASET = "allenai/dolma3_mix-150B-1025"
SPLIT = "train"
VALIDATION_SAMPLES = 50000
SHUFFLE_SEED = 42


class Lab5(ModelEvaluationTask):
    grading_type = GradingType.SCORE
    id = "lab5"
    resources = Resources(cpus=8, memory_gb=32, gpus=1, timeout_seconds=4 * 3600)
    primary_metric = "score"
    metric_direction = MetricDirection.MINIMIZE
    evaluator = PerplexityEvaluator(
        tokenizer_id="meta-llama/Llama-3.2-1B",
        batch_size=64,
        max_length=8192,
        dtype="bfloat16",
    )

    def load_dataset(self) -> Dataset:
        print("[lab5] loading Dolma 3 mix", flush=True)
        dataset = load_dataset(DATASET, split=SPLIT)
        if len(dataset) < VALIDATION_SAMPLES:
            raise ValueError(
                f"Lab 5 requires at least {VALIDATION_SAMPLES} documents; "
                f"got {len(dataset)}"
            )
        # The held-out tail is defined after a full deterministic shuffle, not
        # a streaming buffer shuffle or a shuffle of the original tail alone.
        dataset = dataset.shuffle(seed=SHUFFLE_SEED)
        return dataset.select(range(len(dataset) - VALIDATION_SAMPLES, len(dataset)))
