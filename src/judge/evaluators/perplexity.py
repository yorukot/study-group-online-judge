import copy
import math
import sys
from dataclasses import dataclass, fields
from itertools import chain
from math import ceil

import numpy as np
import torch
import torch._dynamo.config
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from judge.evaluators.base import Evaluator
from judge.models import JudgeResult, TestResult

CUDA_DEVICE = torch.device("cuda:0")


class CompilationRequiredModel(torch.nn.Module):
    """Reject an eager forward even if compilation was disabled externally."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        if not torch.compiler.is_compiling():
            raise RuntimeError("Perplexity model forward requires torch.compile")
        return self.model(**inputs)


class DecoderReplay(torch.nn.Module):
    """Supply computed decoder states to the causal LM's own output-head code."""

    def __init__(self, output_type, hidden_state_field):
        super().__init__()
        self.output_type = output_type
        self.hidden_state_field = hidden_state_field

    def forward(self, *args, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            raise RuntimeError("Chunked output-head scoring requires decoder states")
        return self.output_type(**{self.hidden_state_field: inputs_embeds})


class OutputHeadLoss(torch.nn.Module):
    """Keep the model's head, bias, projection, scaling, and softcapping intact."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, labels):
        if not torch.compiler.is_compiling():
            raise RuntimeError("Perplexity loss requires torch.compile")
        logits = self.model(
            inputs_embeds=hidden_states.unsqueeze(0),
            use_cache=False,
            return_dict=True,
        ).logits
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels,
            reduction="none",
        )


def score_hidden_states(hidden_states, input_ids, attention_mask, loss_fn, chunk_size):
    """Bound vocabulary-logit memory independently of the decoder batch size."""
    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:].masked_fill(~attention_mask[:, 1:].bool(), -100)
    flat_labels = labels.reshape(-1)
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    losses = torch.empty(
        flat_labels.shape, dtype=torch.float32, device=input_ids.device
    )
    for start in range(0, flat_labels.numel(), chunk_size):
        stop = start + chunk_size
        losses[start:stop] = loss_fn(hidden_states[start:stop], flat_labels[start:stop])
    return document_scores(
        losses.view_as(input_ids).sum(dim=1).tolist(),
        attention_mask[:, 1:].sum(dim=1).tolist(),
    )


def scoring_decoder(model):
    getter = getattr(model, "get_decoder", None)
    decoder = getter() if callable(getter) else model.base_model
    return model.base_model if decoder is model else decoder


def supports_chunked_scoring(model):
    """Use standard Transformers interfaces, without an architecture allowlist."""
    if not isinstance(model, PreTrainedModel):
        return False
    decoder = scoring_decoder(model)
    return (
        decoder is not model
        and any(child is decoder for child in model.modules())
        and isinstance(model.get_output_embeddings(), torch.nn.Module)
    )


def replay_decoder(module, decoder, replay):
    """Copy only decoder ancestors; keep original modules and weights untouched."""
    if module is decoder:
        return replay
    children = {
        name: replay_decoder(child, decoder, replay) if child is not None else None
        for name, child in module._modules.items()
    }
    if all(child is module._modules[name] for name, child in children.items()):
        return module
    clone = copy.copy(module)
    clone._modules = children
    return clone


class ChunkedPerplexityModel:
    """Compile the standard base model and replay its output through the LM head."""

    def __init__(self, model: PreTrainedModel, chunk_size):
        if not supports_chunked_scoring(model):
            raise ValueError(
                "Chunked scoring requires a separate base model and output head"
            )
        self.model = model
        self.decoder = torch.compile(
            CompilationRequiredModel(scoring_decoder(model)),
            backend="inductor",
            fullgraph=True,
            dynamic=True,
        )
        self.loss_fn = None
        self.chunk_size = chunk_size

    def score(self, inputs):
        outputs = self.decoder(**inputs, use_cache=False, return_dict=True)
        hidden_states = outputs[0]
        if hidden_states.device != CUDA_DEVICE:
            raise RuntimeError("Perplexity model returned hidden states outside cuda:0")
        if self.loss_fn is None:
            # Replay the actual decoder, including models exposing it below
            # base_model (via get_decoder). Preserve the complete LM head forward.
            head_model = replay_decoder(
                self.model,
                scoring_decoder(self.model),
                DecoderReplay(type(outputs), fields(outputs)[0].name),
            )
            self.loss_fn = torch.compile(
                OutputHeadLoss(head_model),
                backend="inductor",
                fullgraph=True,
                dynamic=True,
            )
        return score_hidden_states(
            hidden_states,
            inputs["input_ids"],
            inputs["attention_mask"],
            self.loss_fn,
            self.chunk_size,
        )


@torch.inference_mode()
def evaluate_perplexity(
    batch: dict[str, list | torch.Tensor], model, tokenizer, *, max_length: int = 1024
) -> dict[str, list]:
    inputs = (
        tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        if "text" in batch
        else {k: torch.as_tensor(v) for k, v in batch.items()}
    )
    inputs = {k: v.to(CUDA_DEVICE) for k, v in inputs.items()}
    if isinstance(model, ChunkedPerplexityModel):
        return model.score(inputs)
    output = model(**inputs, use_cache=False)
    if output.logits.device != CUDA_DEVICE:
        raise RuntimeError("Perplexity model returned logits outside cuda:0")
    return score_logits(output.logits, inputs["input_ids"], inputs["attention_mask"])


def score_logits(
    logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> dict[str, list]:
    """Compute padding-aware document losses from next-token logits."""
    logits = logits[:, :-1, :].float().contiguous()
    labels = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].bool()

    losses = F.cross_entropy(
        logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
    )

    losses = losses.view_as(labels).masked_fill(~mask, 0)
    loss_sums = losses.sum(dim=1).tolist()
    token_counts = mask.sum(dim=1).tolist()
    return document_scores(loss_sums, token_counts)


def document_scores(loss_sums, token_counts):
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


@dataclass(frozen=True)
class PerplexityEvaluator(Evaluator):
    """Token-weighted causal-LM perplexity with mandatory CUDA and Inductor."""

    batch_size: int = 32
    tokenizer_id: str | None = None
    max_length: int = 1024
    dtype: str = "auto"
    logits_chunk_size: int = 4096

    def validate_runtime(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Perplexity evaluation requires a CUDA GPU; CPU/MPS fallback is disabled"
            )
        if self.batch_size < 1:
            raise ValueError("Perplexity batch_size must be positive")
        if self.max_length < 2:
            raise ValueError("Perplexity max_length must be at least two tokens")
        if self.logits_chunk_size < 1:
            raise ValueError("Perplexity logits_chunk_size must be positive")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported perplexity dtype")

    @torch.inference_mode()
    def evaluate(self, model_id: str, dataset: Dataset) -> JudgeResult:
        self.validate_runtime()
        print(f"[perplexity] loading model {model_id} on {CUDA_DEVICE}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype)
        max_length = self.max_length
        context_length = getattr(model.config, "max_position_embeddings", None)
        if isinstance(context_length, int) and context_length > 0:
            max_length = min(max_length, context_length)
        torch.nn.Module.to(model, device=CUDA_DEVICE)
        model.eval()
        if any(
            tensor.device != CUDA_DEVICE
            for tensor in chain(model.parameters(), model.buffers())
        ):
            raise RuntimeError(
                "All perplexity model parameters and buffers must be on cuda:0"
            )
        print(
            "[perplexity] torch.compile enabled (inductor, fullgraph, dynamic)",
            flush=True,
        )
        if supports_chunked_scoring(model):
            print(
                f"[perplexity] chunked output-head loss ({self.logits_chunk_size} tokens/chunk)",
                flush=True,
            )
            model = ChunkedPerplexityModel(model, self.logits_chunk_size)
        else:
            model = torch.compile(
                CompilationRequiredModel(model),
                backend="inductor",
                fullgraph=True,
                dynamic=True,
            )
        tokenizer_id = self.tokenizer_id or model_id
        print(f"[perplexity] loading tokenizer {tokenizer_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    "Perplexity tokenizer requires a pad token or EOS token"
                )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        total_loss = 0.0
        total_tokens = 0
        document_perplexities = []
        with torch._dynamo.config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=True
        ):
            for batch in tqdm(
                dataset.iter(batch_size=self.batch_size),
                desc="[perplexity] evaluating batches",
                total=ceil(len(dataset) / self.batch_size),
                file=sys.stdout,
                mininterval=5,
            ):
                results = evaluate_perplexity(
                    batch, model, tokenizer, max_length=max_length
                )
                total_loss += sum(results["loss_sum"])
                total_tokens += sum(results["token_count"])
                document_perplexities.extend(
                    value
                    for value in results["document_perplexity"]
                    if value is not None
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
