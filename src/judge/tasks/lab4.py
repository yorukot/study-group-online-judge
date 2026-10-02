import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge.models import (
    GradingType,
    JudgeResult,
    MetricDirection,
    Resources,
    TestResult,
)
from judge.tasks.base import Task, load_student_function

DATASET = "allenai/c4"
SUBSET = "en"
SPLIT = "validation"
VALIDATION_SAMPLES = 100_000
TOKENIZER_ID = "openai-community/gpt2"
EVALUATION_BATCH_SIZE = 32

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.mps.is_available()
    else "cpu"
)


@torch.inference_mode()
def evaluate_perplexity(batch: dict[str, list], model, tokenizer) -> dict[str, list]:
    inputs = tokenizer(
        batch["text"],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model(**inputs)

    logits = output.logits[:, :-1, :].float().contiguous()
    labels = inputs["input_ids"][:, 1:].contiguous()
    mask = inputs["attention_mask"][:, 1:].bool()

    losses = F.cross_entropy(
        logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
    )

    losses = losses.view_as(labels).masked_fill(~mask, 0)
    loss_sums = losses.sum(dim=1).tolist()
    token_counts = mask.sum(dim=1).tolist()
    document_perplexities = [
        (
            math.exp(loss_sum / token_count)
            if loss_sum / token_count < math.log(sys.float_info.max)
            else math.inf
        )
        if token_count
        else None
        for loss_sum, token_count in zip(loss_sums, token_counts, strict=True)
    ]
    return {
        "loss_sum": loss_sums,
        "token_count": token_counts,
        "document_perplexity": document_perplexities,
    }


class Lab4(Task):
    grading_type = GradingType.SCORE
    id = "lab4"
    resources = Resources(cpus=8, memory_gb=32, gpus=1, timeout_seconds=4 * 3600)
    primary_metric = "score"
    metric_direction = MetricDirection.MINIMIZE

    def evaluate(self, submission: Path) -> JudgeResult:
        torch.set_num_threads(self.resources.cpus)
        print("[lab4] starting C4 en validation evaluation", flush=True)
        # `split` selects the result after preparation; restrict files as well
        # so validation does not download and prepare the entire training set.
        dataset = load_dataset(
            DATASET,
            SUBSET,
            data_files={SPLIT: "en/c4-validation.*.json.gz"},
            split=SPLIT,
            # The published split metadata also expects the excluded train set.
            verification_mode="no_checks",
        )
        dataset = dataset.shuffle(seed=42).select(range(VALIDATION_SAMPLES))

        sys.path.insert(0, str(submission / "src"))
        try:
            print("[lab4] loading src/labs/lab4.py", flush=True)
            model_name = load_student_function(submission, "lab4").eval_model_id
        except Exception as error:  # noqa: BLE001 - participant failures are test failures
            print(
                f"[lab4] participant implementation failed: {type(error).__name__}: {error}",
                flush=True,
            )
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

        if not isinstance(model_name, str) or not model_name.strip():
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

        print(f"[lab4] loading participant model {model_name}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        torch.nn.Module.to(model, device=torch.device(device))
        model.eval()
        print(f"[lab4] loading GPT-2 tokenizer {TOKENIZER_ID}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        total_loss = 0.0
        total_tokens = 0
        document_perplexities = []
        for start in tqdm(
            range(0, len(dataset), EVALUATION_BATCH_SIZE),
            desc="[lab4] evaluating C4 perplexity",
            file=sys.stdout,
            mininterval=5,
        ):
            batch = dataset[start : start + EVALUATION_BATCH_SIZE]
            results = evaluate_perplexity(batch, model, tokenizer)
            total_loss += sum(results["loss_sum"])
            total_tokens += sum(results["token_count"])
            document_perplexities.extend(
                value for value in results["document_perplexity"] if value is not None
            )
        if total_tokens == 0:
            return JudgeResult(
                passed=False,
                tests=[
                    TestResult(
                        name="validation_data",
                        passed=False,
                        message="No validation tokens remained after tokenization",
                    )
                ],
            )
        mean_loss = total_loss / total_tokens
        corpus_perplexity = (
            math.exp(mean_loss)
            if mean_loss < math.log(sys.float_info.max)
            else math.inf
        )
        if not math.isfinite(corpus_perplexity) or not all(
            math.isfinite(value) for value in document_perplexities
        ):
            return JudgeResult(
                passed=False,
                tests=[
                    TestResult(
                        name="perplexity",
                        passed=False,
                        message="Model produced non-finite perplexity",
                    )
                ],
            )
        print(
            f"[lab4] evaluated {len(document_perplexities)}/{len(dataset)} documents "
            f"and {total_tokens} tokens; corpus perplexity={corpus_perplexity:.4f}",
            flush=True,
        )

        return JudgeResult(
            passed=True,
            score=corpus_perplexity,
            metrics={
                "corpus_perplexity": corpus_perplexity,
                "p90_document_perplexity": float(
                    np.percentile(document_perplexities, 90)
                ),
                "p99_document_perplexity": float(
                    np.percentile(document_perplexities, 99)
                ),
                "evaluated_documents": float(len(document_perplexities)),
                "evaluated_tokens": float(total_tokens),
                "skipped_documents": float(len(dataset) - len(document_perplexities)),
            },
        )


if __name__ == "__main__":
    lab4 = Lab4()
    print(lab4.evaluate(Path("")))
