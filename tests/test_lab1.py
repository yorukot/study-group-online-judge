import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from judge.tasks.base import load_student_function
from judge.tasks.lab1 import Lab1


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(12))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return f"prompt-{len(ids)}"


class Lab1Tests(unittest.TestCase):
    def test_loads_student_implementation_from_src_labs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "src" / "labs" / "lab1.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "def gpt2_complete(input, max_seq_length=1024):\n    return input, None\n"
            )

            complete = load_student_function(Path(directory), "lab1").gpt2_complete

            self.assertEqual(complete(["hello"]), (["hello"], None))

    def evaluate_with(self, actual_logits: torch.Tensor):
        expected_logits = torch.zeros((20, 4, 3), dtype=torch.float16)
        completions = ["completion"] * 20

        def student(prompts: list[str], max_seq_length: int):
            self.assertEqual(len(prompts), 20)
            self.assertEqual(set(prompts), {"prompt-6", "prompt-7", "prompt-8"})
            self.assertEqual(max_seq_length, 10)
            return completions, actual_logits

        with (
            patch("judge.tasks.lab1.AutoTokenizer.from_pretrained") as tokenizer,
            patch(
                "judge.tasks.lab1.load_student_function",
                return_value=SimpleNamespace(gpt2_complete=student),
            ),
            patch(
                "judge.tasks.lab1._reference_completion",
                return_value=(completions, expected_logits),
            ),
            patch("judge.tasks.lab1.torch.set_num_threads"),
        ):
            tokenizer.return_value = FakeTokenizer()
            output = StringIO()
            with redirect_stdout(output):
                result = Lab1().evaluate(Path("."))
            self.logs = output.getvalue()
            return result

    def test_passes_all_20_matching_samples(self) -> None:
        result = self.evaluate_with(torch.zeros((20, 4, 3), dtype=torch.float16))

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1)
        self.assertEqual(len(result.tests), 20)
        self.assertIn("running student model", self.logs)
        self.assertIn("validating samples", self.logs)
        self.assertIn("20/20 samples passed", self.logs)

    def test_fails_the_sample_with_wrong_logits(self) -> None:
        logits = torch.zeros((20, 4, 3), dtype=torch.float16)
        logits[2, 0, 0] = 1

        result = self.evaluate_with(logits)

        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(
            [test.name for test in result.tests if not test.passed],
            ["tiny_shakespeare_03"],
        )
        self.assertIn("max_abs_diff=1", result.tests[2].message)
        self.assertIn("tiny_shakespeare_03 failed", self.logs)

    def test_logs_student_exception(self) -> None:
        with (
            patch("judge.tasks.lab1.AutoTokenizer.from_pretrained") as tokenizer,
            patch(
                "judge.tasks.lab1.load_student_function",
                side_effect=RuntimeError("model download failed"),
            ),
            patch("judge.tasks.lab1.torch.set_num_threads"),
        ):
            tokenizer.return_value = FakeTokenizer()
            output = StringIO()
            with redirect_stdout(output):
                result = Lab1().evaluate(Path("."))

        self.assertFalse(result.passed)
        self.assertEqual(result.tests[0].message, "RuntimeError: model download failed")
        self.assertIn("student implementation failed", output.getvalue())
        self.assertIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
