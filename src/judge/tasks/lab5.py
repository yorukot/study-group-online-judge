from datasets import Dataset, load_dataset

from judge.evaluators import PerplexityEvaluator
from judge.models import GradingType, MetricDirection, Resources
from judge.tasks.model import ModelEvaluationTask

DATASET = "cerulean-works/study-group-lab5-eval"
SPLIT = "train"
VALIDATION_SAMPLES = 50000


class Lab5(ModelEvaluationTask):
    grading_type = GradingType.SCORE
    id = "lab5"
    resources = Resources(cpus=8, memory_gb=32, gpus=1, timeout_seconds=4 * 3600)
    primary_metric = "score"
    metric_direction = MetricDirection.MINIMIZE
    evaluator = PerplexityEvaluator(
        tokenizer_id="meta-llama/Llama-3.2-1B",
        max_seq_length=8192,
        dtype="bfloat16",
    )

    def load_dataset(self) -> Dataset:
        print("[lab5] loading prepared evaluation dataset", flush=True)
        dataset = load_dataset(DATASET, None, split=SPLIT)
        if len(dataset) != VALIDATION_SAMPLES:
            raise ValueError(
                f"Lab 5 requires exactly {VALIDATION_SAMPLES} documents; "
                f"got {len(dataset)}"
            )
        # Already shuffled and filtered; preserve the published evaluation set.
        return dataset
