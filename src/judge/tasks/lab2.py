from judge.models import GradingType

"""Full MMLU reference comparison for Lab 2."""

import hashlib
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge.models import JudgeResult, Resources, TestResult
from judge.tasks.base import Task, load_student_function

DATASET_ID = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MODEL_ID = "openai-community/gpt2"
LETTERS = "ABCD"
EXEMPLARS_PER_SUBJECT = 4
MIN_AGREEMENT_PERCENT = 97
REFERENCE_BATCH_SIZE = 16


def question_key(index: int, row: dict[str, Any]) -> str:
    """Identify a test row, including MMLU's exact duplicate questions."""

    value = {
        "index": index,
        "subject": row["subject"],
        "question": row["question"],
        "choices": row["choices"],
    }
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_data() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    print(f"[lab2] loading {DATASET_ID} at {DATASET_REVISION}", flush=True)
    dev = load_dataset(DATASET_ID, "all", split="dev", revision=DATASET_REVISION)
    test = load_dataset(DATASET_ID, "all", split="test", revision=DATASET_REVISION)
    exemplars: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev:
        if len(exemplars[row["subject"]]) < EXEMPLARS_PER_SUBJECT:
            exemplars[row["subject"]].append(row)
    rows = list(test)
    subjects = {row["subject"] for row in rows}
    if not rows or any(
        len(exemplars[subject]) != EXEMPLARS_PER_SUBJECT for subject in subjects
    ):
        raise RuntimeError("MMLU is empty or a subject lacks four dev exemplars")
    print(
        f"[lab2] loaded {len(rows)} test questions across {len(subjects)} subjects",
        flush=True,
    )
    return dict(exemplars), rows


def _format_question(row: dict[str, Any], answer: str | None = None) -> str:
    choices = row["choices"]
    if len(choices) != 4:
        raise ValueError("MMLU question does not have four choices")
    options = " ".join(
        f"({letter}) {choice}" for letter, choice in zip(LETTERS, choices, strict=True)
    )
    return f"{row['question']}\n{options}\nAnswer: {answer or ''}"


def _prompt(row: dict[str, Any], exemplars: list[dict[str, Any]]) -> str:
    subject = row["subject"].replace("_", " ")
    examples = [
        _format_question(example, LETTERS[example["answer"]]) for example in exemplars
    ]
    return (
        f"The following are multiple choice questions about {subject}.\n\n"
        + "\n\n".join([*examples, _format_question(row)])
    )


def _reference_predictions(
    rows: list[dict[str, Any]],
    exemplars: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("Lab 2 requires an available CUDA GPU")
    print(f"[lab2] loading GPT-2 reference tokenizer: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    token_ids = [
        tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS
    ]
    if any(len(ids) != 1 for ids in token_ids):
        raise RuntimeError("An answer letter is not one GPT-2 token")
    print(f"[lab2] loading GPT-2 reference model: {MODEL_ID} (fp16, CUDA)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    torch.nn.Module.to(model, device=torch.device("cuda"))
    model.eval()
    context_length = model.config.n_positions
    print(
        f"[lab2] reference model ready; evaluating all {len(rows)} test questions",
        flush=True,
    )
    predictions: dict[str, str] = {}
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(rows), REFERENCE_BATCH_SIZE),
            desc="[lab2] reference inference",
            file=sys.stdout,
            mininterval=5,
        ):
            batch = range(start, min(start + REFERENCE_BATCH_SIZE, len(rows)))
            prompts = [
                _prompt(rows[index], exemplars[rows[index]["subject"]])
                for index in batch
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            ).to("cuda")
            positions = (encoded["attention_mask"].cumsum(dim=1) - 1).clamp_min(0)
            logits = model(**encoded, position_ids=positions).logits[:, -1, :]
            choices = logits[:, [ids[0] for ids in token_ids]].argmax(dim=-1).tolist()
            for index, choice in zip(batch, choices, strict=True):
                predictions[question_key(index, rows[index])] = LETTERS[choice]
    print("[lab2] reference predictions complete", flush=True)
    return predictions


def _failure(name: str, reason: str) -> JudgeResult:
    print(f"[lab2] {name} failed: {reason}", flush=True)
    return JudgeResult(
        passed=False,
        tests=[TestResult(name=name, passed=False, message=reason)],
    )


class Lab2(Task):
    grading_type = GradingType.PASS_FAIL
    id = "lab2"
    resources = Resources(cpus=8, memory_gb=32, gpus=1, timeout_seconds=4 * 3600)

    def evaluate(self, submission: Path) -> JudgeResult:
        torch.set_num_threads(self.resources.cpus)
        exemplars, rows = _load_data()
        expected_keys = {question_key(index, row) for index, row in enumerate(rows)}
        if len(expected_keys) != len(rows):
            raise RuntimeError("MMLU test row identities are not unique")

        sys.path.insert(0, str(submission / "src"))
        try:
            try:
                print("[lab2] loading src/labs/lab2.py", flush=True)
                evaluate = load_student_function(submission, "lab2").mmlu_eval
                print("[lab2] running participant MMLU evaluation", flush=True)
                predictions = evaluate()
                print("[lab2] participant evaluation finished", flush=True)
            except Exception as error:  # noqa: BLE001 - participant failures are test failures
                print(
                    f"[lab2] participant implementation failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                traceback.print_exc(file=sys.stdout)
                return _failure("student_function", f"{type(error).__name__}: {error}")
        finally:
            sys.path.pop(0)

        if not isinstance(predictions, dict):
            return _failure(
                "return_contract",
                f"Expected dict[str, str], got {type(predictions).__name__}",
            )
        actual_keys = set(predictions)
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        invalid = [
            key
            for key in expected_keys & actual_keys
            if not isinstance(predictions[key], str) or predictions[key] not in LETTERS
        ]
        if missing or extra or invalid:
            return _failure(
                "return_contract",
                f"Expected {len(expected_keys)} exact question keys with A/B/C/D predictions; "
                f"missing={len(missing)} (e.g. {sorted(missing)[:2]}), "
                f"extra={len(extra)} (e.g. {sorted(map(str, extra))[:2]}), "
                f"invalid={len(invalid)} (e.g. {sorted(invalid)[:2]})",
            )

        print(
            f"[lab2] comparing all {len(rows)} participant predictions with GPT-2",
            flush=True,
        )
        reference = _reference_predictions(rows, exemplars)
        by_subject: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
        correct_answers = 0
        for index, row in enumerate(
            tqdm(
                rows,
                desc="[lab2] validating predictions",
                file=sys.stdout,
                mininterval=5,
            )
        ):
            key = question_key(index, row)
            correct_answers += int(predictions[key] == LETTERS[row["answer"]])
            by_subject[row["subject"]].append(
                (index, key, predictions[key], reference[key])
            )

        tests = []
        mismatches = 0
        for subject in sorted(by_subject):
            checked = by_subject[subject]
            differences = [
                (index, key, got, want)
                for index, key, got, want in checked
                if got != want
            ]
            mismatches += len(differences)
            examples = ", ".join(
                f"row {index} ({key[:12]}): got {got}, expected {want}"
                for index, key, got, want in differences[:3]
            )
            matched = len(checked) - len(differences)
            message = f"{matched}/{len(checked)} matched" + (
                f"; examples: {examples}" if examples else ""
            )
            print(f"[lab2] {subject}: {message}", flush=True)
            tests.append(
                TestResult(
                    name=subject,
                    passed=matched * 100 >= MIN_AGREEMENT_PERCENT * len(checked),
                    message=message,
                )
            )

        matched = len(rows) - mismatches
        agreement = matched / len(rows)
        mmlu_accuracy = correct_answers / len(rows)
        passed = matched * 100 >= MIN_AGREEMENT_PERCENT * len(rows)
        summary = (
            f"{matched}/{len(rows)} matched "
            f"({agreement:.2%}); requires at least {MIN_AGREEMENT_PERCENT}% agreement"
        )
        print(f"[lab2] verdict: {'PASS' if passed else 'FAIL'}; {summary}", flush=True)
        print(
            f"[lab2] MMLU accuracy: {correct_answers}/{len(rows)} "
            f"({mmlu_accuracy:.2%})",
            flush=True,
        )
        tests.insert(
            0,
            TestResult(name="overall_agreement", passed=passed, message=summary),
        )
        return JudgeResult(
            passed=passed,
            score=agreement,
            metrics={
                "evaluated_questions": float(len(rows)),
                "samples_passed": float(matched),
                "mismatched_predictions": float(mismatches),
                "mmlu_accuracy": mmlu_accuracy,
            },
            tests=tests,
        )
