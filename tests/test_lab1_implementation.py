"""Offline checks of the student model, independent of the judge's mock tests."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2LMHeadModel

from labs.lab1 import GPT2, VOCAB_SIZE, gpt2_complete


@pytest.fixture
def models(tmp_path: Path):
    # A tiny random checkpoint exercises the same operations without a download.
    # Saving the reference's original layout also checks checkpoint loading.
    with torch.random.fork_rng():
        torch.manual_seed(42)
        config = GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=24,
            n_layer=2,
            n_head=3,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
        )
        reference = GPT2LMHeadModel(config).eval()
        model = GPT2(
            vocab_size=32,
            max_positions=16,
            hidden_size=24,
            num_layers=2,
            num_heads=3,
        ).eval()
    checkpoint = tmp_path / "tiny-gpt2.safetensors"
    save_file(reference.transformer.state_dict(), checkpoint)
    model.load_weights(checkpoint)
    return model, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("use_cache", [False, True])
def test_forward_matches_reference_with_left_padding(models, dtype, use_cache):
    model, reference = (model.to(dtype=dtype) for model in models)
    input_ids = torch.tensor([[0, 0, 4, 5], [6, 7, 8, 9]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
    with torch.inference_mode():
        actual = model(input_ids, mask)
        expected = reference(
            input_ids, attention_mask=mask, position_ids=positions, use_cache=use_cache
        ).logits
    # Fully padded query positions are irrelevant; compare every real position.
    # Small fp16 errors can flip a greedy choice even if allclose would pass.
    # Require exact fp16 agreement for this reference checkpoint and batch.
    tolerance = 0 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(
        actual[mask.bool()], expected[mask.bool()], atol=tolerance, rtol=tolerance
    )


def test_future_tokens_cannot_change_prefix_logits(models):
    model, _ = models
    mask = torch.ones((1, 4), dtype=torch.long)
    with torch.inference_mode():
        first = model(torch.tensor([[1, 2, 3, 4]]), mask)
        changed = model(torch.tensor([[1, 2, 8, 9]]), mask)
    torch.testing.assert_close(first[:, :2], changed[:, :2], atol=0, rtol=0)


def test_padding_does_not_change_real_token_logits(models):
    model, _ = models
    with torch.inference_mode():
        alone = model(torch.tensor([[4, 5]]), torch.ones((1, 2), dtype=torch.long))
        batched = model(
            torch.tensor([[20, 21, 4, 5], [6, 7, 8, 9]]),
            torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
        )
    torch.testing.assert_close(alone[0], batched[0, -2:], atol=1e-6, rtol=1e-5)


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return {
            "short": [1, 2],
            "ends_early": [3, 4, 5],
            "at_limit": [6, 7, 8, 9],
            "over_limit": [10, 11, 12, 13, 14],
            "empty": [],
            "too_long": [1] * 1025,
        }[text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(f"<{token}>" for token in ids if token != self.eos_token_id)


class ScriptedModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        step = len(self.calls)
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        batch_size, width = input_ids.shape
        logits = torch.full((batch_size, width, VOCAB_SIZE), -2.0)
        # Row 0 emits 5 then 6. Row 1 emits EOS immediately. Rows 2 and 3
        # already exceed the generation budget, regardless of their predictions.
        for row, token in enumerate((5 + step, 0, 9, 9)):
            logits[row, -1, token] = 3.0 + step
        return logits


def test_generation_batches_prompts_and_stops_each_row_independently():
    model = ScriptedModel()
    with (
        patch("labs.lab1.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
        patch("labs.lab1._load_model", return_value=model),
    ):
        texts, logits = gpt2_complete(
            ["short", "ends_early", "at_limit", "over_limit"], max_seq_length=4
        )

    assert texts == ["<5><6>", "", "", ""]
    assert logits.shape == (4, 2, VOCAB_SIZE)
    assert logits[0, 0, 5] == 3  # Raw scores, not probabilities or one-hot tokens.
    assert logits[0, 1, 6] == 4
    assert logits[0, 0, 6] == -2
    assert logits[1, 0, 0] == 3  # Preserve the logits that produced EOS.
    assert torch.count_nonzero(logits[1, 1]) == 0
    assert torch.count_nonzero(logits[2:]) == 0
    assert [ids.shape for ids, _ in model.calls] == [(4, 5), (4, 6)]
    assert model.calls[1][1][:, -1].tolist() == [1, 1, 0, 0]


def test_empty_batch_needs_no_download():
    with patch("labs.lab1.AutoTokenizer.from_pretrained") as tokenizer:
        texts, logits = gpt2_complete([])
    assert texts == []
    assert logits.shape == (0, 0, VOCAB_SIZE)
    tokenizer.assert_not_called()


def test_finished_prompts_need_no_model():
    with (
        patch("labs.lab1.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
        patch("labs.lab1._load_model") as load_model,
    ):
        texts, logits = gpt2_complete(["at_limit", "over_limit"], max_seq_length=4)
    assert texts == ["", ""]
    assert logits.shape == (2, 0, VOCAB_SIZE)
    load_model.assert_not_called()


@pytest.mark.parametrize("max_seq_length", [0, -1, 1025])
def test_rejects_invalid_length_limit(max_seq_length):
    with pytest.raises(ValueError, match="max_seq_length"):
        gpt2_complete(["short"], max_seq_length=max_seq_length)


@pytest.mark.parametrize("prompt", ["empty", "too_long"])
def test_rejects_unsupported_prompt_lengths(prompt):
    with (
        patch("labs.lab1.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
        pytest.raises(ValueError, match="prompt|Prompts"),
    ):
        gpt2_complete([prompt])
