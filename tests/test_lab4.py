import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from judge.tasks.lab4 import Lab4, evaluate_perplexity


class TokenizerStub:
    eos_token = "<eos>"

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


class ModelStub:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def __call__(self, **inputs):
        assert inputs["input_ids"].shape[:2] == self.logits.shape[:2]
        return SimpleNamespace(logits=self.logits)


class Lab4Tests(unittest.TestCase):
    def test_per_document_losses_ignore_padding_and_aggregate_by_token(self) -> None:
        logits = torch.zeros((2, 4, 4))
        logits[0, 0, 2] = 2
        logits[0, 1:, 0] = 100  # Padded positions must not affect the score.
        logits[1, 0, 3] = 1
        logits[1, 1, 2] = 1
        logits[1, 2, 3] = 1

        result = evaluate_perplexity(
            {"text": ["short", "long"]}, ModelStub(logits), TokenizerStub()
        )
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
        result = evaluate_perplexity(
            {"text": ["tiny"]},
            ModelStub(torch.zeros((1, 1, 4))),
            TokenizerStub(),
        )
        self.assertEqual(result["token_count"], [0])
        self.assertEqual(result["loss_sum"], [0])
        self.assertEqual(result["document_perplexity"], [None])

    def test_evaluate_uses_participant_model_and_corpus_perplexity(self) -> None:
        dataset = Dataset.from_dict({"text": ["short", "long", "tiny"]})
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        tokenizer = TokenizerStub()
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
        with (
            patch("judge.tasks.lab4.load_dataset", return_value=dataset) as load_data,
            patch("judge.tasks.lab4.VALIDATION_SAMPLES", 3),
            patch("judge.tasks.lab4.EVALUATION_BATCH_SIZE", 2),
            patch(
                "judge.tasks.lab4.load_student_function",
                return_value=SimpleNamespace(eval_model_id="cerulean/trained-gpt2"),
            ),
            patch(
                "judge.tasks.lab4.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as load_model,
            patch(
                "judge.tasks.lab4.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer,
            patch(
                "judge.tasks.lab4.evaluate_perplexity", side_effect=batch_results
            ) as evaluate_batch,
            patch("judge.tasks.lab4.torch.set_num_threads"),
            patch("judge.tasks.lab4.torch.nn.Module.to"),
        ):
            result = Lab4().evaluate(Path("."))

        self.assertTrue(result.passed)
        score = result.score
        assert score is not None
        self.assertAlmostEqual(score, math.e)
        self.assertEqual(result.metrics["evaluated_tokens"], 5)
        self.assertEqual(result.metrics["evaluated_documents"], 3)
        self.assertAlmostEqual(
            result.metrics["p90_document_perplexity"],
            float(np.percentile([1, math.exp(2 / 3), math.exp(3)], 90)),
        )
        load_data.assert_called_once_with(
            "allenai/c4",
            "en",
            data_files={"validation": "en/c4-validation.*.json.gz"},
            split="validation",
            verification_mode="no_checks",
        )
        load_model.assert_called_once_with("cerulean/trained-gpt2")
        load_tokenizer.assert_called_once_with("openai-community/gpt2")
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(evaluate_batch.call_count, 2)
        model.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
