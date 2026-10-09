import math
import unittest
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
    CUDA_DEVICE,
    ChunkedPerplexityModel,
    CompilationRequiredModel,
    DecoderReplay,
    OutputHeadLoss,
    evaluate_perplexity,
    score_hidden_states,
    score_logits,
    supports_chunked_scoring,
)


@pytest.fixture(autouse=True)
def isolate_compiled_model_caches():
    # Parametrized architectures/precisions must not exhaust a shared wrapper's
    # Dynamo cache and trigger production recompilation limits in unrelated tests.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class TokenizerStub:
    eos_token: str | None = "<eos>"

    def __init__(self) -> None:
        self.pad_token = None
        self.padding_side = "left"

    def __call__(self, texts, **kwargs):
        assert kwargs["padding"] == "longest"
        sequences = {"short": [1, 2], "long": [1, 3, 2, 3], "tiny": [1]}
        ids = [sequences[text] for text in texts]
        width = max(map(len, ids))
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in ids]),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in ids]
            ),
        }


class PerplexityMathTests(unittest.TestCase):
    def test_per_document_losses_ignore_padding_and_aggregate_by_token(self) -> None:
        logits = torch.zeros((2, 4, 4))
        logits[0, 0, 2] = 2
        logits[0, 1:, 0] = 100  # Padded positions must not affect the score.
        logits[1, 0, 3] = 1
        logits[1, 1, 2] = 1
        logits[1, 2, 3] = 1

        inputs = TokenizerStub()(["short", "long"], padding="longest")
        result = score_logits(logits, **inputs)
        first_loss = F.cross_entropy(logits[0, 0], torch.tensor(2)).item()
        second_loss = sum(
            F.cross_entropy(logits[1, index], torch.tensor(label)).item()
            for index, label in enumerate([3, 2, 3])
        )

        self.assertEqual(result["token_count"], [1, 3])
        self.assertAlmostEqual(result["loss_sum"][0], first_loss, places=5)
        self.assertAlmostEqual(result["loss_sum"][1], second_loss, places=5)
        self.assertAlmostEqual(
            result["document_perplexity"][0], math.exp(first_loss), places=5
        )
        self.assertAlmostEqual(
            result["document_perplexity"][1], math.exp(second_loss / 3), places=5
        )
        self.assertNotEqual(*result["document_perplexity"])

    def test_short_document_has_no_scored_tokens(self) -> None:
        inputs = TokenizerStub()(["tiny"], padding="longest")
        result = score_logits(torch.zeros((1, 1, 4)), **inputs)
        self.assertEqual(result["token_count"], [0])
        self.assertEqual(result["loss_sum"], [0])
        self.assertEqual(result["document_perplexity"], [None])


class PerplexityRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_dict({"text": ["short", "long", "tiny"]})
        self.model = MagicMock()
        self.model.parameters.return_value = [SimpleNamespace(device=CUDA_DEVICE)]
        self.model.buffers.return_value = [SimpleNamespace(device=CUDA_DEVICE)]
        self.tokenizer = TokenizerStub()

    def evaluate_with(self, results, evaluator=None, **compile_kwargs):
        with (
            patch(
                "judge.evaluators.perplexity.torch.cuda.is_available", return_value=True
            ),
            patch(
                "judge.evaluators.perplexity.AutoModelForCausalLM.from_pretrained",
                return_value=self.model,
            ) as load_model,
            patch(
                "judge.evaluators.perplexity.AutoTokenizer.from_pretrained",
                return_value=self.tokenizer,
            ) as load_tokenizer,
            patch("judge.evaluators.perplexity.torch.nn.Module.to") as move,
            patch(
                "judge.evaluators.perplexity.torch.compile", **compile_kwargs
            ) as compile_model,
            patch(
                "judge.evaluators.perplexity.evaluate_perplexity", side_effect=results
            ) as batch,
        ):
            self.load_tokenizer = load_tokenizer
            evaluator = evaluator or PerplexityEvaluator(batch_size=2)
            result = evaluator.evaluate("cerulean/trained-gpt2", self.dataset)
        return result, load_model, move, compile_model, batch

    def test_requires_cuda_before_loading_model(self):
        with (
            patch(
                "judge.evaluators.perplexity.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "judge.evaluators.perplexity.AutoModelForCausalLM.from_pretrained"
            ) as load,
            self.assertRaisesRegex(RuntimeError, "requires a CUDA GPU"),
        ):
            PerplexityEvaluator().evaluate("model", self.dataset)
        load.assert_not_called()

    def test_requires_positive_batch_size(self):
        with (
            patch(
                "judge.evaluators.perplexity.torch.cuda.is_available", return_value=True
            ),
            self.assertRaisesRegex(ValueError, "batch_size"),
        ):
            PerplexityEvaluator(batch_size=0).validate_runtime()

    def test_evaluates_compiled_model_and_preserves_corpus_metrics(self):
        batch_results = [
            {
                "loss_sum": [0.0, 2.0],
                "token_count": [1, 3],
                "document_perplexity": [1.0, math.exp(2 / 3)],
            },
            {
                "loss_sum": [3.0],
                "token_count": [1],
                "document_perplexity": [math.exp(3)],
            },
        ]

        # Assert strict configuration from inside the lazy forward calls.
        def score_batch(*args, **kwargs):
            self.assertFalse(torch._dynamo.config.suppress_errors)
            self.assertTrue(torch._dynamo.config.fail_on_recompile_limit_hit)
            return batch_results.pop(0)

        compiled = MagicMock()
        result, load, move, compile_model, batch = self.evaluate_with(
            score_batch, return_value=compiled
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.score, math.e)
        self.assertEqual(result.metrics["evaluated_tokens"], 5)
        self.assertEqual(result.metrics["evaluated_documents"], 3)
        self.assertAlmostEqual(
            result.metrics["p90_document_perplexity"],
            float(np.percentile([1, math.exp(2 / 3), math.exp(3)], 90)),
        )
        load.assert_called_once_with("cerulean/trained-gpt2", dtype="auto")
        self.load_tokenizer.assert_called_once_with("cerulean/trained-gpt2")
        move.assert_called_once_with(self.model, device=CUDA_DEVICE)
        compile_model.assert_called_once()
        assert isinstance(compile_model.call_args.args[0], CompilationRequiredModel)
        self.assertIs(compile_model.call_args.args[0].model, self.model)
        self.assertEqual(
            compile_model.call_args.kwargs,
            {"backend": "inductor", "fullgraph": True, "dynamic": True},
        )
        self.assertEqual(batch.call_count, 2)
        self.assertIs(batch.call_args.args[1], compiled)
        self.assertEqual(batch.call_args.kwargs, {"max_length": 1024})
        self.model.eval.assert_called_once_with()
        self.assertEqual(self.tokenizer.pad_token, self.tokenizer.eos_token)
        self.assertEqual(self.tokenizer.padding_side, "right")

    def test_fixed_tokenizer_and_custom_length_are_used(self):
        _, _, _, _, batch = self.evaluate_with(
            [{"loss_sum": [1], "token_count": [1], "document_perplexity": [math.e]}],
            evaluator=PerplexityEvaluator(
                tokenizer_id="meta-llama/Llama-3.2-1B", max_length=2048
            ),
        )
        self.load_tokenizer.assert_called_once_with("meta-llama/Llama-3.2-1B")
        self.assertEqual(batch.call_args.kwargs, {"max_length": 2048})

    def test_length_is_capped_by_model_context(self):
        self.model.config.max_position_embeddings = 64
        _, _, _, _, batch = self.evaluate_with(
            [
                {"loss_sum": [1], "token_count": [1], "document_perplexity": [math.e]},
                {"loss_sum": [1], "token_count": [1], "document_perplexity": [math.e]},
            ]
        )
        self.assertEqual(batch.call_args.kwargs, {"max_length": 64})

    def test_existing_padding_token_is_preserved(self):
        self.tokenizer.pad_token = "<pad>"
        self.evaluate_with(
            [
                {"loss_sum": [1], "token_count": [1], "document_perplexity": [math.e]},
                {"loss_sum": [1], "token_count": [1], "document_perplexity": [math.e]},
            ]
        )
        self.assertEqual(self.tokenizer.pad_token, "<pad>")

    def test_missing_pad_and_eos_tokens_fail(self):
        self.tokenizer.eos_token = None
        with self.assertRaisesRegex(ValueError, "pad token or EOS token"):
            self.evaluate_with([])

    def test_model_left_on_cpu_is_rejected_before_compilation(self):
        for member in ("parameters", "buffers"):
            with self.subTest(member=member):
                self.setUp()
                getattr(self.model, member).return_value = [torch.zeros(1)]
                with self.assertRaisesRegex(RuntimeError, "parameters and buffers"):
                    self.evaluate_with([])

    def test_compilation_failure_is_not_suppressed(self):
        with self.assertRaisesRegex(RuntimeError, "compiler failure"):
            self.evaluate_with([], side_effect=RuntimeError("compiler failure"))

    def test_lazy_compilation_failure_is_not_suppressed(self):
        with self.assertRaisesRegex(RuntimeError, "lazy compiler failure"):
            self.evaluate_with(RuntimeError("lazy compiler failure"))

    def test_empty_tokens_fail_cleanly(self):
        result, *_ = self.evaluate_with(
            [
                {
                    "loss_sum": [0, 0],
                    "token_count": [0, 0],
                    "document_perplexity": [None, None],
                },
                {"loss_sum": [0], "token_count": [0], "document_perplexity": [None]},
            ]
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.tests[0].name, "validation_data")

    def test_nonfinite_perplexity_fails(self):
        result, *_ = self.evaluate_with(
            [
                {
                    "loss_sum": [math.inf, 0],
                    "token_count": [1, 1],
                    "document_perplexity": [math.inf, 1],
                },
                {"loss_sum": [0], "token_count": [1], "document_perplexity": [1]},
            ]
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.tests[0].name, "perplexity")

    def test_skips_documents_without_tokens(self):
        result, *_ = self.evaluate_with(
            [
                {
                    "loss_sum": [1, 0],
                    "token_count": [1, 0],
                    "document_perplexity": [math.e, None],
                },
                {"loss_sum": [1], "token_count": [1], "document_perplexity": [math.e]},
            ]
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["skipped_documents"], 1)
        self.assertEqual(result.metrics["evaluated_documents"], 2)

    def test_batch_requires_cuda_inputs_and_outputs_and_disables_cache(self):
        inputs = {key: MagicMock() for key in ("input_ids", "attention_mask")}
        for tensor in inputs.values():
            tensor.to.return_value = tensor
        tokenizer = MagicMock(return_value=inputs)
        model = MagicMock(return_value=SimpleNamespace(logits=torch.zeros(1)))
        with self.assertRaisesRegex(RuntimeError, "logits outside cuda:0"):
            evaluate_perplexity({"text": ["short"]}, model, tokenizer)
        for tensor in inputs.values():
            tensor.to.assert_called_once_with(CUDA_DEVICE)
        model.assert_called_once_with(**inputs, use_cache=False)
        tokenizer.assert_called_once_with(
            ["short"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024,
        )


def tiny_model(architecture):
    if architecture == "gpt2":
        return GPT2LMHeadModel(
            GPT2Config(vocab_size=4, n_positions=1024, n_embd=16, n_layer=1, n_head=2)
        )
    if architecture == "gemma2":
        model = Gemma2ForCausalLM(
            Gemma2Config(
                vocab_size=4,
                max_position_embeddings=1024,
                hidden_size=16,
                intermediate_size=32,
                head_dim=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                sliding_window=2,
                final_logit_softcapping=0.1,
                attn_implementation="eager",
            )
        )
        with torch.no_grad():
            model.get_output_embeddings().weight.mul_(20)
        return model
    if architecture == "opt":
        return OPTForCausalLM(
            OPTConfig(
                vocab_size=4,
                max_position_embeddings=1024,
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
            max_position_embeddings=1024,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )


@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_causal_models_trace_as_full_graphs(architecture):
    model = tiny_model(architecture).eval()
    compiled = torch.compile(
        CompilationRequiredModel(model), backend="eager", fullgraph=True, dynamic=True
    )
    with torch.inference_mode():
        for batch_size, length in [(2, 4), (1, 1), (2, 8)]:
            ids = torch.ones(batch_size, length, dtype=torch.long)
            mask = torch.ones_like(ids)
            actual = compiled(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            expected = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_real_cuda_inductor_perplexity(architecture):
    torch.manual_seed(42)
    model = tiny_model(architecture)
    torch.nn.Module.to(model, device=CUDA_DEVICE)
    model.eval()
    tokenizer = TokenizerStub()
    tokenizer.padding_side = "right"
    dataset = Dataset.from_dict({"text": ["short", "long", "tiny"]})
    with torch.inference_mode():
        baseline = evaluate_perplexity({"text": ["short", "long"]}, model, tokenizer)
    expected = math.exp(sum(baseline["loss_sum"]) / sum(baseline["token_count"]))
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
        result = PerplexityEvaluator(batch_size=2).evaluate(
            f"test/tiny-{architecture}", dataset
        )
    assert result.passed
    assert result.score == pytest.approx(expected, rel=1e-4)
    assert result.metrics["skipped_documents"] == 1


def test_eager_model_forward_is_rejected():
    model = MagicMock()
    with pytest.raises(RuntimeError, match="forward requires torch.compile"):
        CompilationRequiredModel(model)(input_ids=torch.ones(1, 2, dtype=torch.long))
    model.assert_not_called()


def test_disabled_compile_cannot_silently_run_eager():
    model = MagicMock()
    compiled = torch.compile(CompilationRequiredModel(model), disable=True)
    with pytest.raises(RuntimeError, match="forward requires torch.compile"):
        compiled(input_ids=torch.ones(1, 2, dtype=torch.long))
    model.assert_not_called()


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 100])
def test_chunked_loss_matches_full_logits_with_padding_and_document_boundaries(
    chunk_size,
):
    torch.manual_seed(42)
    inputs = TokenizerStub()(["short", "long", "tiny"], padding="longest")
    hidden = torch.randn(3, 4, 8)
    head = torch.nn.Linear(8, 4)
    expected = score_logits(head(hidden), **inputs)

    def loss_fn(h, y):
        return F.cross_entropy(head(h).float(), y, reduction="none")

    actual = score_hidden_states(
        hidden, **inputs, loss_fn=loss_fn, chunk_size=chunk_size
    )
    assert actual["token_count"] == expected["token_count"]
    assert actual["loss_sum"] == pytest.approx(expected["loss_sum"], abs=1e-6)
    assert actual["document_perplexity"][-1] is None
    assert actual["document_perplexity"][:2] == pytest.approx(
        expected["document_perplexity"][:2], rel=1e-6
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"logits_chunk_size": 0},
        {"dtype": "int8"},
    ],
)
def test_invalid_optimized_evaluator_settings(kwargs):
    with (
        patch("judge.evaluators.perplexity.torch.cuda.is_available", return_value=True),
        pytest.raises(ValueError),
    ):
        PerplexityEvaluator(**kwargs).validate_runtime()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("architecture", ["gpt2", "llama", "gemma2", "opt"])
def test_cuda_chunked_scoring_matches_masked_full_logits(dtype, architecture):
    torch.manual_seed(42)
    model = tiny_model(architecture).to(device=CUDA_DEVICE, dtype=dtype).eval()
    inputs = {
        k: v.to(CUDA_DEVICE)
        for k, v in TokenizerStub()(
            ["short", "long", "tiny"], padding="longest"
        ).items()
    }
    with torch.inference_mode():
        reference = torch.compile(
            CompilationRequiredModel(model),
            backend="inductor",
            fullgraph=True,
            dynamic=True,
        )
        expected = score_logits(reference(**inputs, use_cache=False).logits, **inputs)
        original_base = model.base_model
        original_attention = model.config._attn_implementation
        actual = ChunkedPerplexityModel(model, chunk_size=3).score(inputs)
        assert model.base_model is original_base
        assert model.config._attn_implementation == original_attention
    assert actual["token_count"] == expected["token_count"]
    assert actual["loss_sum"] == pytest.approx(expected["loss_sum"], rel=1e-4, abs=1e-6)


def test_output_head_rejects_disabled_compilation():
    compiled = torch.compile(OutputHeadLoss(tiny_model("llama").eval()), disable=True)
    with pytest.raises(RuntimeError, match="loss requires torch.compile"):
        compiled(torch.randn(2, 16), torch.ones(2, dtype=torch.long))


@pytest.mark.parametrize("architecture", ["gpt2", "llama", "gemma2", "opt"])
def test_standard_decoder_head_interfaces_enable_chunking(architecture):
    assert supports_chunked_scoring(tiny_model(architecture))
    assert not supports_chunked_scoring(tiny_model(architecture).base_model)


def test_decoder_replay_requires_hidden_states():
    from transformers.modeling_outputs import BaseModelOutputWithPast

    with pytest.raises(RuntimeError, match="requires decoder states"):
        DecoderReplay(BaseModelOutputWithPast, "last_hidden_state")()


@pytest.mark.parametrize("architecture", ["gpt2", "llama", "gemma2", "opt"])
def test_generic_chunked_scoring_traces_on_cpu_with_native_model_heads(architecture):
    torch.manual_seed(42)
    model = tiny_model(architecture).eval()
    inputs = TokenizerStub()(["short", "long", "tiny"], padding="longest")
    original_compile = torch.compile

    def compile_for_tracing(
        module: torch.nn.Module, *, backend: str, fullgraph: bool, dynamic: bool
    ):
        return original_compile(
            module, backend="eager", fullgraph=fullgraph, dynamic=dynamic
        )

    with torch.inference_mode():
        expected = score_logits(model(**inputs, use_cache=False).logits, **inputs)
        with (
            patch("judge.evaluators.perplexity.CUDA_DEVICE", torch.device("cpu")),
            patch(
                "judge.evaluators.perplexity.torch.compile",
                side_effect=compile_for_tracing,
            ),
        ):
            actual = ChunkedPerplexityModel(model, chunk_size=3).score(inputs)
    assert actual["token_count"] == expected["token_count"]
    assert actual["loss_sum"] == pytest.approx(expected["loss_sum"], rel=1e-4, abs=1e-6)
