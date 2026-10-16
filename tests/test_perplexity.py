import math
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
)

from judge.evaluators import PerplexityEvaluator
from judge.evaluators.perplexity import (
    CE_CHUNK_SIZE,
    CUDA_DEVICE,
    eval_batch_ppl,
    evaluate_perplexity,
)


class TokenizerStub:
    def __init__(self, pad_token_id=0, eos_token_id=2, padding_side="right"):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.padding_side = padding_side

    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "return_tensors": "pt",
            "padding": "longest",
            "truncation": True,
        }
        sequences = {"short": [1, 2], "long": [1, 3, 2, 3], "tiny": [1]}
        rows = [sequences[text] for text in texts]
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            padding = [self.pad_token_id] * (width - len(row))
            mask = [1] * len(row)
            zeros = [0] * len(padding)
            ids.append(row + padding if self.padding_side == "right" else padding + row)
            masks.append(mask + zeros if self.padding_side == "right" else zeros + mask)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


def reference_scores(logits, input_ids, attention_mask):
    """Independent scalar next-token NLL oracle, without production CE/reductions."""
    losses, counts, ppls = [], [], []
    for row in range(len(input_ids)):
        terms = []
        for position in range(1, len(input_ids[row])):
            target = int(input_ids[row, position])
            if (
                attention_mask[row, position - 1]
                and attention_mask[row, position]
                and target != -100
            ):
                scores = logits[row, position - 1].double()
                terms.append((torch.logsumexp(scores, dim=0) - scores[target]).item())
        losses.append(sum(terms))
        counts.append(len(terms))
        ppls.append(math.exp(sum(terms) / len(terms)) if terms else None)
    return {"loss_sum": losses, "token_count": counts, "document_perplexity": ppls}


def assert_scores(actual, expected, rel=1e-6):
    assert actual["token_count"] == expected["token_count"]
    assert actual["loss_sum"] == pytest.approx(expected["loss_sum"], rel=rel, abs=1e-6)
    for actual_ppl, expected_ppl in zip(
        actual["document_perplexity"], expected["document_perplexity"], strict=True
    ):
        if expected_ppl is None:
            assert actual_ppl is None
        else:
            assert actual_ppl == pytest.approx(expected_ppl, rel=rel)


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("pad_token_id", [0, 2])
@pytest.mark.parametrize("pretokenized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_batch_matches_reference_with_padding_eos_and_causal_shift(
    padding_side, pad_token_id, pretokenized, dtype
):
    tokenizer = TokenizerStub(pad_token_id=pad_token_id, padding_side=padding_side)
    texts = ["short", "long", "tiny"]
    inputs = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True)
    torch.manual_seed(42)
    logits = torch.randn(3, 4, 4).to(dtype)
    # Even very bad predictions at padding must not affect loss or denominator.
    logits[~inputs["attention_mask"].bool(), :] = 100
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    batch = (
        {k: v.tolist() for k, v in inputs.items()} if pretokenized else {"text": texts}
    )
    with patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")):
        actual = evaluate_perplexity(batch, model, tokenizer)
    assert_scores(actual, reference_scores(logits, **inputs))
    assert actual["token_count"] == [1, 3, 0]
    model.assert_called_once()
    assert model.call_args.kwargs["return_dict"] is True
    assert model.call_args.kwargs["use_cache"] is False
    for key, tensor in inputs.items():
        torch.testing.assert_close(model.call_args.kwargs[key], tensor)


@pytest.mark.parametrize("ids, mask", [([[1]], [[1]]), ([[0, 0]], [[0, 0]])])
def test_documents_without_next_token_targets_are_skipped(ids, mask):
    inputs = {"input_ids": ids, "attention_mask": mask}
    model = MagicMock(
        return_value=SimpleNamespace(logits=torch.zeros(1, len(ids[0]), 4))
    )
    with patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")):
        result = evaluate_perplexity(inputs, model, TokenizerStub())
    assert result == {
        "loss_sum": [0],
        "token_count": [0],
        "document_perplexity": [None],
    }


def test_pretokenized_inputs_without_mask_use_pad_id():
    ids = torch.tensor([[1, 2, 0, 0], [0, 1, 3, 2]])
    logits = torch.randn(2, 4, 4)
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    tokenizer = MagicMock(pad_token_id=0)
    with patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")):
        actual = evaluate_perplexity({"input_ids": ids}, model, tokenizer)
    tokenizer.assert_not_called()
    assert_scores(actual, reference_scores(logits, ids, ids != 0))


@pytest.mark.parametrize("loss", [100.0, 1000.0])
def test_document_exponentiation_uses_python_float_and_handles_overflow(loss):
    logits = torch.tensor([[[0.0, -loss], [0.0, 0.0]]])
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    with patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")):
        actual = evaluate_perplexity({"input_ids": [[0, 1]]}, model, TokenizerStub(2))
    expected = math.exp(loss) if loss == 100 else math.inf
    assert actual["document_perplexity"] == [expected]


@contextmanager
def evaluator_runtime(tokenizer=None, compiled=None, compile_error=None):
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    tokenizer = tokenizer or TokenizerStub(pad_token_id=None)
    compiled = compiled if compiled is not None else MagicMock()
    with (
        patch("judge.evaluators.perplexity.torch.cuda.is_available", return_value=True),
        patch(
            "judge.evaluators.perplexity.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as load_model,
        patch(
            "judge.evaluators.perplexity.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as load_tokenizer,
        patch(
            "judge.evaluators.perplexity.torch.compile",
            return_value=compiled,
            side_effect=compile_error,
        ) as compile_model,
    ):
        yield SimpleNamespace(
            model=model,
            tokenizer=tokenizer,
            compiled=compiled,
            load_model=load_model,
            load_tokenizer=load_tokenizer,
            compile=compile_model,
        )


def test_requires_cuda_before_loading_model():
    with (
        patch(
            "judge.evaluators.perplexity.torch.cuda.is_available", return_value=False
        ),
        patch(
            "judge.evaluators.perplexity.AutoModelForCausalLM.from_pretrained"
        ) as load,
        pytest.raises(RuntimeError, match="requires a CUDA GPU"),
    ):
        PerplexityEvaluator().evaluate("model", Dataset.from_dict({"text": ["short"]}))
    load.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"dtype": "int8"},
        {"max_seq_length": 0},
        {"max_seq_length": 1},
    ],
)
def test_invalid_evaluator_settings(kwargs):
    with (
        patch("judge.evaluators.perplexity.torch.cuda.is_available", return_value=True),
        pytest.raises(ValueError),
    ):
        PerplexityEvaluator(**kwargs).validate_runtime()


def test_runtime_compiles_direct_model_and_aggregates_by_token():
    dataset = Dataset.from_dict({"text": ["short", "long", "tiny"]})
    batches = [
        {
            "loss_sum": [0, 2],
            "token_count": [1, 3],
            "document_perplexity": [1, math.exp(2 / 3)],
        },
        {"loss_sum": [3], "token_count": [1], "document_perplexity": [math.exp(3)]},
    ]
    with (
        evaluator_runtime() as runtime,
        patch(
            "judge.evaluators.perplexity.evaluate_perplexity", side_effect=batches
        ) as score,
    ):
        result = PerplexityEvaluator(
            batch_size=2, tokenizer_id="fixed/tokenizer", dtype="bfloat16"
        ).evaluate("student/model", dataset)
    assert result.passed
    assert result.score == pytest.approx(math.e)
    assert result.metrics["corpus_perplexity"] == result.score
    assert result.metrics["evaluated_tokens"] == 5
    assert result.metrics["evaluated_documents"] == 3
    assert result.metrics["skipped_documents"] == 0
    for percentile in (90, 99):
        assert result.metrics[f"p{percentile}_document_perplexity"] == pytest.approx(
            np.percentile([1, math.exp(2 / 3), math.exp(3)], percentile)
        )
    runtime.load_model.assert_called_once_with("student/model", dtype="bfloat16")
    runtime.load_tokenizer.assert_called_once_with("fixed/tokenizer")
    runtime.model.to.assert_called_once_with(CUDA_DEVICE)
    runtime.model.eval.assert_called_once_with()
    runtime.compile.assert_called_once_with(runtime.model, dynamic=True, fullgraph=True)
    assert runtime.tokenizer.pad_token_id == runtime.tokenizer.eos_token_id
    assert score.call_count == 2
    assert score.call_args_list[0].args[0] == {"text": ["short", "long"]}
    assert score.call_args_list[1].args[0] == {"text": ["tiny"]}
    assert score.call_args.args[1] is runtime.compiled
    assert score.call_args.kwargs == {}


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_end_to_end_score_is_invariant_to_batch_size_and_padding(
    batch_size, padding_side
):
    # Uniform four-class logits give NLL log(4), independent of batching/padding.
    model = MagicMock(
        side_effect=lambda **inputs: SimpleNamespace(
            logits=torch.zeros(*inputs["input_ids"].shape, 4)
        )
    )
    tokenizer = TokenizerStub(pad_token_id=2, padding_side=padding_side)
    with (
        evaluator_runtime(tokenizer, compiled=model) as runtime,
        patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")),
    ):
        result = PerplexityEvaluator(batch_size=batch_size).evaluate(
            "student/model", Dataset.from_dict({"text": ["short", "long", "tiny"]})
        )
    assert result.passed
    assert result.score == pytest.approx(4)
    assert result.metrics["evaluated_tokens"] == 4
    assert result.metrics["evaluated_documents"] == 2
    assert result.metrics["skipped_documents"] == 1
    assert result.metrics["p90_document_perplexity"] == pytest.approx(4)
    assert result.metrics["p99_document_perplexity"] == pytest.approx(4)
    runtime.load_tokenizer.assert_called_once_with("student/model")
    assert tokenizer.pad_token_id == 2
    assert tokenizer.padding_side == padding_side


@pytest.mark.parametrize("texts", [[], ["tiny"]])
def test_empty_validation_tokens_fail_cleanly(texts):
    with (
        evaluator_runtime(
            compiled=MagicMock(
                return_value=SimpleNamespace(logits=torch.zeros(1, 1, 4))
            )
        ),
        patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")),
    ):
        result = PerplexityEvaluator().evaluate(
            "student/model", Dataset.from_dict({"text": texts})
        )
    assert not result.passed
    assert result.tests[0].name == "validation_data"


@pytest.mark.parametrize("value", [math.inf, math.nan, 1000.0])
def test_nonfinite_perplexity_fails(value):
    with (
        evaluator_runtime(),
        patch(
            "judge.evaluators.perplexity.evaluate_perplexity",
            return_value={
                "loss_sum": [value],
                "token_count": [1],
                "document_perplexity": [value],
            },
        ),
    ):
        result = PerplexityEvaluator().evaluate(
            "student/model", Dataset.from_dict({"text": ["short"]})
        )
    assert not result.passed
    assert result.tests[0].name == "perplexity"


def test_missing_pad_and_eos_tokens_fail():
    with (
        evaluator_runtime(TokenizerStub(pad_token_id=None, eos_token_id=None)),
        pytest.raises(ValueError, match="pad token or EOS token"),
    ):
        PerplexityEvaluator().evaluate("model", Dataset.from_dict({"text": ["short"]}))


@pytest.mark.parametrize("lazy", [False, True])
def test_compilation_failure_propagates(lazy):
    failure = RuntimeError("compiler failure")
    compiled = MagicMock(side_effect=failure)
    with (
        evaluator_runtime(compiled=compiled, compile_error=None if lazy else failure),
        patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")),
        pytest.raises(RuntimeError, match="compiler failure"),
    ):
        PerplexityEvaluator().evaluate("model", Dataset.from_dict({"text": ["short"]}))


def tiny_model(architecture):
    if architecture == "gpt2":
        return GPT2LMHeadModel(
            GPT2Config(vocab_size=4, n_positions=32, n_embd=16, n_layer=1, n_head=2)
        )
    if architecture == "gemma2":
        model = Gemma2ForCausalLM(
            Gemma2Config(
                vocab_size=4,
                max_position_embeddings=32,
                hidden_size=16,
                intermediate_size=32,
                head_dim=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                sliding_window=2,
                final_logit_softcapping=0.1,
            )
        )
        model.set_attn_implementation("eager")
        return model
    if architecture == "opt":
        return OPTForCausalLM(
            OPTConfig(
                vocab_size=4,
                max_position_embeddings=32,
                hidden_size=16,
                word_embed_proj_dim=8,
                ffn_dim=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                pad_token_id=0,
            )
        )
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=4,
            max_position_embeddings=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )


@pytest.mark.parametrize("architecture", ["gpt2", "llama", "gemma2", "opt"])
def test_direct_model_fullgraph_scoring_matches_reference_on_cpu(architecture):
    torch.manual_seed(42)
    model = tiny_model(architecture).eval()
    tokenizer = TokenizerStub()
    inputs = tokenizer(
        ["short", "long", "tiny"],
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )
    with torch.inference_mode():
        expected = reference_scores(
            model(**inputs, return_dict=True, use_cache=False).logits, **inputs
        )
        # Exercise Dynamo's fullgraph tracing locally; real Inductor is tested on CUDA below.
        torch._dynamo.reset()
        try:
            compiled = torch.compile(model, backend="eager", fullgraph=True)
            with patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")):
                actual = evaluate_perplexity(inputs, compiled, tokenizer)
            assert_scores(actual, expected)
        finally:
            torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("architecture", ["gpt2", "llama", "gemma2", "opt"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_cuda_inductor_evaluator_matches_reference(architecture, dtype):
    torch.manual_seed(42)
    model = tiny_model(architecture).to(device=CUDA_DEVICE, dtype=dtype).eval()
    tokenizer = TokenizerStub(pad_token_id=2)
    texts = ["short", "long", "tiny"]
    inputs = {
        k: v.to(CUDA_DEVICE)
        for k, v in tokenizer(
            texts, return_tensors="pt", padding="longest", truncation=True
        ).items()
    }
    with torch.inference_mode():
        expected = reference_scores(
            model(**inputs, return_dict=True, use_cache=False).logits, **inputs
        )
    with (
        patch(
            "judge.evaluators.perplexity.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
        patch(
            "judge.evaluators.perplexity.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
    ):
        torch._dynamo.reset()
        try:
            result = PerplexityEvaluator(batch_size=2).evaluate(
                f"test/tiny-{architecture}", Dataset.from_dict({"text": texts})
            )
        finally:
            torch._dynamo.reset()
    assert result.passed
    assert result.score == pytest.approx(
        math.exp(sum(expected["loss_sum"]) / 4), rel=2e-3
    )
    assert result.metrics["evaluated_tokens"] == 4
    assert result.metrics["skipped_documents"] == 1


class PackedTokenizerStub(TokenizerStub):
    def __call__(self, texts, **kwargs):
        assert kwargs["padding"] is False
        assert kwargs["truncation"] is True
        sequences = {"short": [1, 2], "long": [1, 3, 2, 3], "tiny": [1], "empty": []}
        return {
            "input_ids": [sequences[text][: kwargs["max_length"]] for text in texts]
        }


@pytest.mark.parametrize("lengths", [[2, 3, 5], [2049, 2, 9]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_scores_match_independent_documents_across_ce_chunks(lengths, dtype):
    torch.manual_seed(123)
    documents = [torch.randint(0, 4, (length,)).tolist() for length in lengths]
    # EOS is a real target, including directly before a boundary.
    for ids in documents:
        ids[-1] = 2
    logits = torch.randn(1, sum(lengths), 4).to(dtype)
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    expected = {"loss_sum": [], "token_count": [], "document_perplexity": []}
    start = 0
    for ids in documents:
        document = torch.tensor([ids])
        scores = reference_scores(
            logits[:, start : start + len(ids)], document, torch.ones_like(document)
        )
        for key, values in expected.items():
            values.extend(scores[key])
        start += len(ids)
    with (
        patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")),
        patch(
            "judge.evaluators.perplexity.F.cross_entropy", wraps=F.cross_entropy
        ) as ce,
    ):
        actual = eval_batch_ppl(model, documents)
    assert_scores(actual, expected)
    assert max(call.args[0].shape[0] for call in ce.call_args_list) <= CE_CHUNK_SIZE
    assert len(ce.call_args_list) == math.ceil((sum(lengths) - 1) / CE_CHUNK_SIZE)
    inputs = model.call_args.kwargs
    assert inputs["input_ids"].tolist() == [
        [token for ids in documents for token in ids]
    ]
    assert inputs["position_ids"].tolist() == [
        [position for length in lengths for position in range(length)]
    ]
    boundaries = [0, *np.cumsum(lengths).tolist()]
    assert inputs["cu_seq_lens_q"].tolist() == boundaries
    assert inputs["cu_seq_lens_k"].tolist() == boundaries
    assert inputs["cu_seq_lens_q"].dtype == torch.int32
    assert inputs["max_length_q"] == inputs["max_length_k"] == max(lengths)
    assert not inputs["use_cache"]


@pytest.mark.parametrize("documents", [[], [[]], [[1]], [[1, 2], [3]]])
def test_packed_scorer_rejects_empty_or_short_documents(documents):
    model = MagicMock()
    with pytest.raises(ValueError, match="at least two tokens"):
        eval_batch_ppl(model, documents)
    model.assert_not_called()


@pytest.mark.parametrize("budget", [2, 4, 6, 8])
def test_packed_evaluator_tokenizes_truncates_flushes_and_preserves_corpus_score(
    budget, capsys
):
    model = MagicMock(
        side_effect=lambda **inputs: SimpleNamespace(
            logits=torch.zeros(*inputs["input_ids"].shape, 4)
        )
    )
    tokenizer = PackedTokenizerStub(pad_token_id=None, eos_token_id=None)
    dataset = Dataset.from_dict(
        {"text": ["short", "long", "tiny", "empty", "short"], "metadata": [0] * 5}
    )
    with (
        evaluator_runtime(tokenizer, compiled=model) as runtime,
        patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")),
    ):
        result = PerplexityEvaluator(max_seq_length=budget, dtype="bfloat16").evaluate(
            "student/model", dataset
        )
    assert result.passed
    assert result.score == pytest.approx(4)
    assert result.metrics["evaluated_tokens"] == 2 + min(budget, 4) - 1
    assert result.metrics["evaluated_documents"] == 3
    assert result.metrics["skipped_documents"] == 2
    runtime.load_model.assert_called_once_with(
        "student/model",
        dtype="bfloat16",
        attn_implementation="kernels-community/flash-attn2",
    )
    runtime.compile.assert_called_once_with(runtime.model, dynamic=True, fullgraph=True)
    for call in model.call_args_list:
        assert call.kwargs["input_ids"].numel() <= budget
    output = capsys.readouterr().out
    assert "running_corpus_perplexity=4.0000" in output
    assert "rolling_corpus_perplexity=4.0000" in output
    assert "last 3 scored documents" in output


@pytest.mark.parametrize("texts", [[], ["empty", "tiny"]])
def test_packed_evaluator_without_scorable_documents_fails_cleanly(texts):
    with evaluator_runtime(PackedTokenizerStub()) as runtime:
        result = PerplexityEvaluator(max_seq_length=8).evaluate(
            "model", Dataset.from_dict({"text": texts})
        )
    assert not result.passed
    assert result.tests[0].name == "validation_data"
    runtime.compiled.assert_not_called()


def test_rolling_logging_uses_last_1000_documents_and_cumulative_token_weights(capsys):
    # First document falls out of the rolling window. Unequal lengths ensure
    # averaging document PPLs or document losses would give the wrong value.
    batches = [
        {"loss_sum": [10], "token_count": [1], "document_perplexity": [math.exp(10)]},
        {
            "loss_sum": [0, 6] * 500,
            "token_count": [1, 3] * 500,
            "document_perplexity": [1, math.exp(2)] * 500,
        },
    ]
    with (
        evaluator_runtime(),
        patch("judge.evaluators.perplexity.evaluate_perplexity", side_effect=batches),
        patch("judge.evaluators.perplexity.time.monotonic", side_effect=[0, 31, 62]),
    ):
        result = PerplexityEvaluator(batch_size=1000).evaluate(
            "model", Dataset.from_dict({"text": ["short"] * 1001})
        )
    assert result.score == pytest.approx(math.exp(3010 / 2001))
    output = capsys.readouterr().out
    assert (
        f"rolling_corpus_perplexity={math.exp(1.5):.4f} (last 1000 scored documents)"
        in output
    )
    assert f"running_corpus_perplexity={math.exp(3010 / 2001):.4f}" in output
    assert "scored 1/1001 documents" in output


@pytest.mark.parametrize("compiled", [False, True])
def test_llama_packed_flashattention_dispatch_matches_independent_documents_on_cpu(
    compiled, monkeypatch
):
    # Use Transformers' actual FA2/varlen dispatch, replacing only the CUDA
    # kernel with dense CPU attention. This checks kwargs forwarding, RoPE
    # resets, causal masking, and document isolation with real model weights.
    import transformers.modeling_flash_attention_utils as flash_utils

    def reference_varlen(query, key, value, cu_seqlens_q, cu_seqlens_k, **kwargs):
        positions = torch.arange(query.shape[0], device=query.device)
        q_documents = torch.bucketize(positions, cu_seqlens_q[1:-1], right=True)
        k_documents = torch.bucketize(positions, cu_seqlens_k[1:-1], right=True)
        mask = (q_documents[:, None] == k_documents[None, :]) & (
            positions[:, None] >= positions[None, :]
        )
        result = F.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            attn_mask=mask,
            scale=kwargs["softmax_scale"],
        )
        return result.transpose(0, 1)

    monkeypatch.setattr(
        flash_utils,
        "lazy_import_flash_attention",
        lambda *args: ((None, reference_varlen, None, None, None), (dict, dict)),
    )
    torch.manual_seed(42)
    model = tiny_model("llama").to(dtype=torch.bfloat16).eval()
    documents = [[1, 2], [3, 1, 2, 3], [2, 1, 2]]
    expected = {"loss_sum": [], "token_count": [], "document_perplexity": []}
    with torch.inference_mode():
        for ids in documents:
            inputs = torch.tensor([ids])
            logits = model(input_ids=inputs, use_cache=False).logits
            scores = reference_scores(logits, inputs, torch.ones_like(inputs))
            for key, values in expected.items():
                values.extend(scores[key])
    model.config._attn_implementation = "flash_attention_2"
    torch._dynamo.reset()
    try:
        if compiled:
            model = torch.compile(model, backend="eager", dynamic=True, fullgraph=True)
        with patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")):
            actual = eval_batch_ppl(model, documents)
        assert_scores(actual, expected, rel=2e-3)
    finally:
        torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_real_cuda_packed_flashattention_and_inductor_match_independent_documents():
    torch.manual_seed(42)
    model = tiny_model("llama").to(device=CUDA_DEVICE, dtype=torch.bfloat16).eval()
    documents = [[1, 2], [3, 1, 2, 3], [2, 1, 2]]
    expected = {"loss_sum": [], "token_count": [], "document_perplexity": []}
    with torch.inference_mode():
        for ids in documents:
            inputs = torch.tensor([ids], device=CUDA_DEVICE)
            logits = model(input_ids=inputs, use_cache=False).logits
            scores = reference_scores(logits, inputs, torch.ones_like(inputs))
            for key, values in expected.items():
                values.extend(scores[key])
    model.set_attn_implementation("kernels-community/flash-attn2")
    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, dynamic=True, fullgraph=True)
        actual = eval_batch_ppl(compiled, documents)
        assert_scores(actual, expected, rel=2e-3)
    finally:
        torch._dynamo.reset()


def test_explicit_hub_kernel_preserves_real_loading_error(tmp_path, monkeypatch):
    from transformers import modeling_utils

    # Exercise real from_pretrained dispatch: implicit FA2 fallback would replace
    # this error with "flash_attn not installed", hiding the actionable cause.
    tiny_model("llama").save_pretrained(tmp_path)
    requested = []

    def unavailable_binary(implementation, **kwargs):
        requested.append(implementation)
        raise ValueError("No compatible prebuilt kernel for this CUDA runtime")

    monkeypatch.setattr(modeling_utils, "is_kernels_available", lambda: True)
    monkeypatch.setattr(
        modeling_utils, "lazy_import_flash_attention", unavailable_binary
    )
    with (
        patch("judge.evaluators.perplexity.torch.cuda.is_available", return_value=True),
        pytest.raises(ValueError, match="No compatible prebuilt kernel"),
    ):
        PerplexityEvaluator(max_seq_length=8192, dtype="bfloat16").evaluate(
            str(tmp_path), Dataset.from_dict({"text": ["short"]})
        )
    assert requested == ["kernels-community/flash-attn2"]
