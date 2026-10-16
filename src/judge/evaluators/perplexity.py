import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge.evaluators.base import Evaluator
from judge.models import JudgeResult, TestResult

CUDA_DEVICE = torch.device("cuda:0")
CE_CHUNK_SIZE = 2048
ROLLING_DOCUMENTS = 1000
PROGRESS_INTERVAL_SECONDS = 30


def document_scores(loss_sums, token_counts):
    ppls = [
        safe_exp(loss / count) if count else None
        for loss, count in zip(loss_sums, token_counts, strict=True)
    ]
    return {
        "loss_sum": loss_sums,
        "token_count": token_counts,
        "document_perplexity": ppls,
    }


def safe_exp(value: float) -> float:
    return math.exp(value) if value < math.log(sys.float_info.max) else math.inf


@torch.inference_mode()
def eval_batch_ppl(model, token_ids: list[list[int]]) -> dict[str, list]:
    """Pack whole documents with isolated attention and next-token targets."""
    lengths = [len(ids) for ids in token_ids]
    if not lengths or any(length < 2 for length in lengths):
        raise ValueError("Perplexity requires at least two tokens per document")
    cu_lens = [0, *np.cumsum(lengths).tolist()]
    cu_seqlens = torch.tensor(cu_lens, dtype=torch.int32, device=CUDA_DEVICE)
    input_ids = torch.tensor(
        [token for ids in token_ids for token in ids], device=CUDA_DEVICE
    ).unsqueeze(0)
    position_ids = torch.cat(
        [torch.arange(length, device=CUDA_DEVICE) for length in lengths]
    ).unsqueeze(0)
    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        use_cache=False,
        return_dict=True,
    ).logits
    labels = input_ids.clone()
    # Ignore predictions across document boundaries; retain real EOS tokens.
    labels[:, cu_lens[:-1]] = -100
    flattened_ce = torch.cat(
        [
            F.cross_entropy(
                logits_chunk.float(),
                labels_chunk,
                ignore_index=-100,
                reduction="none",
            )
            for logits_chunk, labels_chunk in zip(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]).split(CE_CHUNK_SIZE),
                labels[:, 1:].reshape(-1).split(CE_CHUNK_SIZE),
                strict=True,
            )
        ]
    )
    # Transfer all document losses together, avoiding a GPU sync per document.
    loss_sums = torch.stack(
        [flattened_ce[start : end - 1].sum() for start, end in pairwise(cu_lens)]
    ).tolist()
    return document_scores(loss_sums, [length - 1 for length in lengths])


@torch.inference_mode()
def evaluate_perplexity(
    batch: dict[str, list | torch.Tensor], model, tokenizer
) -> dict[str, list]:
    inputs = (
        tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        if "text" in batch
        else {k: torch.as_tensor(v) for k, v in batch.items()}
    )
    inputs = {k: v.to(CUDA_DEVICE) for k, v in inputs.items()}
    output = model(**inputs, return_dict=True, use_cache=False)
    logits = output.logits

    labels = inputs["input_ids"]
    # Attention masks distinguish real EOS tokens from padding when IDs coincide.
    mask = inputs.get("attention_mask", labels != tokenizer.pad_token_id).bool()
    valid = mask[:, :-1] & mask[:, 1:] & (labels[:, 1:] != -100)
    truncated_labels = labels[:, 1:].masked_fill(~valid, -100)
    truncated_logits = logits[:, :-1, :]

    flattened_ce = F.cross_entropy(
        truncated_logits.reshape(-1, truncated_logits.shape[-1]).float(),
        truncated_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    ce = flattened_ce.reshape_as(truncated_labels)
    loss_sums = ce.sum(dim=-1).tolist()
    token_counts = valid.sum(dim=-1).tolist()
    return document_scores(loss_sums, token_counts)


@dataclass(frozen=True)
class PerplexityEvaluator(Evaluator):
    """Token-weighted causal-LM perplexity with mandatory CUDA and Inductor."""

    batch_size: int = 32
    tokenizer_id: str | None = None
    dtype: str = "auto"
    # None retains padded evaluation for architectures such as GPT-2.
    # Otherwise this is both the document truncation and packed token budget.
    max_seq_length: int | None = None

    def validate_runtime(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Perplexity evaluation requires a CUDA GPU; CPU/MPS fallback is disabled"
            )
        if self.batch_size < 1:
            raise ValueError("Perplexity batch_size must be positive")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported perplexity dtype")
        if self.max_seq_length is not None and self.max_seq_length < 2:
            raise ValueError("Perplexity max_seq_length must be at least two")

    def evaluation_batches(self, model, tokenizer, dataset):
        if self.max_seq_length is None:
            for batch in tqdm(
                dataset.iter(batch_size=self.batch_size),
                desc="[perplexity] evaluating batches",
                total=ceil(len(dataset) / self.batch_size),
                file=sys.stdout,
                mininterval=5,
            ):
                yield evaluate_perplexity(batch, model, tokenizer)
            return
        tokenized = dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
            ),
            batched=True,
            remove_columns=dataset.column_names,
            desc="[perplexity] tokenizing",
        )
        pack = []
        pack_length = 0
        for batch in tokenized.iter(batch_size=1000):
            for token_ids in batch["input_ids"]:
                if len(token_ids) < 2:
                    continue
                if pack and pack_length + len(token_ids) > self.max_seq_length:
                    yield eval_batch_ppl(model, pack)
                    pack = []
                    pack_length = 0
                pack.append(token_ids)
                pack_length += len(token_ids)
        if pack:
            yield eval_batch_ppl(model, pack)

    @torch.inference_mode()
    def evaluate(self, model_id: str, dataset: Dataset) -> JudgeResult:
        self.validate_runtime()
        print(f"[perplexity] loading model {model_id} on {CUDA_DEVICE}", flush=True)
        model_options = {"dtype": self.dtype}
        if self.max_seq_length is not None:
            # Select the prebuilt Hub kernel explicitly. The automatic FA2
            # fallback can replace a kernel-loading error with a misleading
            # missing-flash_attn ImportError.
            model_options["attn_implementation"] = "kernels-community/flash-attn2"
        model = cast(
            torch.nn.Module,
            AutoModelForCausalLM.from_pretrained(model_id, **model_options),
        )
        model = model.to(CUDA_DEVICE).eval()
        model = torch.compile(model, dynamic=True, fullgraph=True)
        print(
            "[perplexity] torch.compile enabled (inductor, fullgraph, dynamic)",
            flush=True,
        )
        tokenizer_id = self.tokenizer_id or model_id
        print(f"[perplexity] loading tokenizer {tokenizer_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer is None:
            raise ValueError("Perplexity tokenizer could not be loaded")
        if self.max_seq_length is None and tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Perplexity tokenizer requires a pad token or EOS token"
                )
            tokenizer.pad_token_id = tokenizer.eos_token_id

        total_loss = 0.0
        total_tokens = 0
        document_perplexities = []
        recent = deque(maxlen=ROLLING_DOCUMENTS)
        last_log = time.monotonic()

        def log_progress():
            if not total_tokens:
                return
            rolling_loss = sum(loss for loss, _ in recent)
            rolling_tokens = sum(count for _, count in recent)
            print(
                f"[perplexity] scored {len(document_perplexities)}/{len(dataset)} documents; "
                f"tokens={total_tokens}; "
                f"running_corpus_perplexity={safe_exp(total_loss / total_tokens):.4f}; "
                f"rolling_corpus_perplexity={safe_exp(rolling_loss / rolling_tokens):.4f} "
                f"(last {len(recent)} scored documents)",
                flush=True,
            )

        for results in self.evaluation_batches(model, tokenizer, dataset):
            total_loss += sum(results["loss_sum"])
            total_tokens += sum(results["token_count"])
            document_perplexities.extend(
                value for value in results["document_perplexity"] if value is not None
            )
            recent.extend(
                (loss, count)
                for loss, count in zip(
                    results["loss_sum"], results["token_count"], strict=True
                )
                if count
            )
            now = time.monotonic()
            if now - last_log >= PROGRESS_INTERVAL_SECONDS:
                log_progress()
                last_log = now
        log_progress()
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
            f"[perplexity] evaluated {len(document_perplexities)}/{len(dataset)} documents "
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
