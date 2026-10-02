import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from judge.tasks.base import load_student_function
from judge.tasks.lab2 import Lab2, _prompt, question_key


def row(subject: str, question: str, answer: int = 0) -> dict:
    return {
        "subject": subject,
        "question": question,
        "choices": ["first", "second", "third", "fourth"],
        "answer": answer,
    }


class Lab2Tests(unittest.TestCase):
    def test_key_disambiguates_exact_duplicate_rows(self) -> None:
        question = row("high_school_mathematics", "What is x?")
        payload = {
            "index": 0,
            "subject": question["subject"],
            "question": question["question"],
            "choices": question["choices"],
        }
        expected = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        self.assertEqual(question_key(0, question), expected)
        self.assertNotEqual(question_key(0, question), question_key(1, question))

    def test_prompt_has_four_subject_exemplars_and_blank_answer(self) -> None:
        exemplars = [
            row("high_school_mathematics", f"Example {i}?", i) for i in range(4)
        ]

        prompt = _prompt(row("high_school_mathematics", "Test?"), exemplars)

        self.assertTrue(
            prompt.startswith(
                "The following are multiple choice questions about high school mathematics.\n\n"
            )
        )
        self.assertIn(
            "Example 3?\n(A) first (B) second (C) third (D) fourth\nAnswer: D", prompt
        )
        self.assertTrue(
            prompt.endswith(
                "Test?\n(A) first (B) second (C) third (D) fourth\nAnswer: "
            )
        )

    def test_imports_participant_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "src" / "labs" / "lab2.py"
            source.parent.mkdir(parents=True)
            source.write_text("def mmlu_eval():\n    return {'hash': 'A'}\n")

            self.assertEqual(
                load_student_function(Path(directory), "lab2").mmlu_eval(),
                {"hash": "A"},
            )

    def evaluate_with(self, size: int, wrong: int):
        rows = [row("high_school_mathematics", f"Question {i}?") for i in range(size)]
        exemplars = {"high_school_mathematics": rows[:4]}
        answers = {
            question_key(i, question): "B" if i < wrong else "A"
            for i, question in enumerate(rows)
        }
        expected = {question_key(i, question): "A" for i, question in enumerate(rows)}
        with (
            patch("judge.tasks.lab2._load_data", return_value=(exemplars, rows)),
            patch(
                "judge.tasks.lab2.load_student_function",
                return_value=SimpleNamespace(mmlu_eval=lambda: answers),
            ),
            patch(
                "judge.tasks.lab2._reference_predictions", return_value=expected
            ) as reference,
            patch("judge.tasks.lab2.torch.set_num_threads"),
            redirect_stdout(StringIO()) as output,
        ):
            result = Lab2().evaluate(Path("."))
        reference.assert_called_once_with(rows, exemplars)
        return result, output.getvalue()

    def test_passes_at_exactly_97_percent_and_reports_bounded_detail(self) -> None:
        result, logs = self.evaluate_with(100, 3)

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.97)
        self.assertEqual(result.metrics["samples_passed"], 97)
        self.assertEqual(result.metrics["mmlu_accuracy"], 0.97)
        self.assertEqual(result.metrics["evaluated_questions"], 100)
        self.assertNotIn("matching_predictions", result.metrics)
        self.assertNotIn("reference_agreement", result.metrics)
        self.assertEqual(len(result.tests), 2)
        self.assertIn("row 0", result.tests[1].message or "")
        self.assertIn("verdict: PASS", logs)
        self.assertNotIn("Question 0?", logs)

    def test_fails_below_97_percent_agreement(self) -> None:
        result, logs = self.evaluate_with(100, 4)

        self.assertFalse(result.passed)
        self.assertFalse(result.tests[0].passed)
        self.assertEqual(result.metrics["mismatched_predictions"], 4)
        self.assertNotIn("row 3", result.tests[1].message or "")
        self.assertIn("verdict: FAIL", logs)

    def test_mmlu_accuracy_uses_answer_labels_not_reference_agreement(self) -> None:
        rows = [row("math", f"Question {index}?", index % 2) for index in range(4)]
        predictions = {
            question_key(index, question): "A" for index, question in enumerate(rows)
        }
        with (
            patch("judge.tasks.lab2._load_data", return_value=({"math": rows}, rows)),
            patch(
                "judge.tasks.lab2.load_student_function",
                return_value=SimpleNamespace(mmlu_eval=lambda: predictions),
            ),
            patch("judge.tasks.lab2._reference_predictions", return_value=predictions),
            patch("judge.tasks.lab2.torch.set_num_threads"),
            redirect_stdout(StringIO()) as output,
        ):
            result = Lab2().evaluate(Path("."))

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics["samples_passed"], 4)
        self.assertEqual(result.metrics["mmlu_accuracy"], 0.5)
        self.assertIn("MMLU accuracy: 2/4 (50.00%)", output.getvalue())

    def test_reports_one_result_per_subject_not_per_question(self) -> None:
        rows = [row(f"subject_{index}", f"Question {index}?") for index in range(57)]
        exemplars = {item["subject"]: [item] * 4 for item in rows}
        predictions = {
            question_key(index, item): "A" for index, item in enumerate(rows)
        }
        with (
            patch("judge.tasks.lab2._load_data", return_value=(exemplars, rows)),
            patch(
                "judge.tasks.lab2.load_student_function",
                return_value=SimpleNamespace(mmlu_eval=lambda: predictions),
            ),
            patch("judge.tasks.lab2._reference_predictions", return_value=predictions),
            patch("judge.tasks.lab2.torch.set_num_threads"),
            redirect_stdout(StringIO()),
        ):
            result = Lab2().evaluate(Path("."))

        self.assertTrue(result.passed)
        self.assertEqual(len(result.tests), 58)
        self.assertEqual(result.metrics["evaluated_questions"], 57)

    def test_rejects_missing_keys_before_reference_model_load(self) -> None:
        rows = [row("math", "one"), row("math", "two")]
        with (
            patch("judge.tasks.lab2._load_data", return_value=({"math": rows}, rows)),
            patch(
                "judge.tasks.lab2.load_student_function",
                return_value=SimpleNamespace(mmlu_eval=dict),
            ),
            patch("judge.tasks.lab2._reference_predictions") as reference,
            patch("judge.tasks.lab2.torch.set_num_threads"),
            redirect_stdout(StringIO()),
        ):
            result = Lab2().evaluate(Path("."))

        self.assertFalse(result.passed)
        self.assertEqual(result.tests[0].name, "return_contract")
        self.assertIn("missing=2", result.tests[0].message or "")
        reference.assert_not_called()

    def test_logs_student_error_and_traceback(self) -> None:
        rows = [row("math", "one")]
        with (
            patch("judge.tasks.lab2._load_data", return_value=({"math": rows}, rows)),
            patch(
                "judge.tasks.lab2.load_student_function",
                side_effect=ValueError("broken"),
            ),
            patch("judge.tasks.lab2.torch.set_num_threads"),
            redirect_stdout(StringIO()) as output,
        ):
            result = Lab2().evaluate(Path("."))

        self.assertFalse(result.passed)
        self.assertEqual(result.tests[0].message, "ValueError: broken")
        self.assertIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
