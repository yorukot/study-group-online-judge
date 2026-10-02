import sys
import traceback
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge.models import GradingType, JudgeResult, Resources, TestResult
from judge.tasks.base import Task, load_student_function

MODEL_ID = "openai-community/gpt2"
MAX_SEQ_LENGTH = 10
LOGIT_ATOL = 0.15
LOGIT_RTOL = 0.02

# Short lines from karpathy/char-rnn's Tiny Shakespeare input.txt.
TINY_SHAKESPEARE_SAMPLES = (
    "Before we proceed any further, hear me speak.",
    "You are all resolved rather to die than to famish?",
    "First, you know Caius Marcius is chief enemy to the people.",
    "Let us kill him, and we'll have corn at our own price.",
    "No more talking on't; let it be done: away, away!",
    "We are accounted poor citizens, the patricians good.",
    "What authority surfeits on would relieve us: if they",
    "would yield us but the superfluity, while it were",
    "wholesome, we might guess they relieved us humanely;",
    "but they think we are too dear: the leanness that",
    "afflicts us, the object of our misery, is as an",
    "inventory to particularise their abundance; our",
    "sufferance is a gain to them Let us revenge this with",
    "our pikes, ere we become rakes: for the gods know I",
    "speak this in hunger for bread, not in thirst for revenge.",
    "Would you proceed especially against Caius Marcius?",
    "Against him first: he's a very dog to the commonalty.",
    "Consider you what services he has done for his country?",
    "Very well; and could be content to give him good",
    "report fort, but that he pays himself with being proud.",
)


def _reference_completion(
    prompts: list[str], max_seq_length: int
) -> tuple[list[str], torch.Tensor]:
    print(f"[lab1] loading reference tokenizer: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"[lab1] loading reference model: {MODEL_ID} (fp16, CPU)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    model.eval()
    print("[lab1] reference model ready", flush=True)

    prompt_ids = [tokenizer.encode(prompt) for prompt in prompts]
    lengths = [len(ids) for ids in prompt_ids]
    width = max(lengths)
    eos_id = tokenizer.eos_token_id
    input_ids = torch.full((len(prompts), width), eos_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(prompt_ids):
        input_ids[index, -len(ids) :] = torch.tensor(ids)
        attention_mask[index, -len(ids) :] = 1

    generated: list[list[int]] = [[] for _ in prompts]
    finished = torch.tensor([length >= max_seq_length for length in lengths])
    steps: list[torch.Tensor] = []

    with (
        torch.inference_mode(),
        tqdm(
            total=max(0, max_seq_length - min(lengths)),
            desc="[lab1] reference decoding",
            file=sys.stdout,
            mininterval=0,
            miniters=1,
        ) as progress,
    ):
        while not bool(finished.all()):
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
            logits = (
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                .logits[:, -1, :]
                .clone()
            )
            active = ~finished
            logits[~active] = 0
            next_ids = logits.argmax(dim=-1)
            next_ids[~active] = eos_id
            steps.append(logits)

            for index in range(len(prompts)):
                if active[index]:
                    token = int(next_ids[index])
                    generated[index].append(token)
                    lengths[index] += 1
                    if token == eos_id or lengths[index] >= max_seq_length:
                        finished[index] = True

            input_ids = torch.cat((input_ids, next_ids[:, None]), dim=1)
            attention_mask = torch.cat(
                (attention_mask, active[:, None].to(dtype=attention_mask.dtype)), dim=1
            )
            progress.update()

    completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    logits = (
        torch.stack(steps, dim=1)
        if steps
        else torch.empty(
            (len(prompts), 0, model.config.vocab_size), dtype=torch.float16
        )
    )
    return completions, logits


class Lab1(Task):
    grading_type = GradingType.PASS_FAIL
    id = "lab1"
    resources = Resources(cpus=4, memory_gb=8, timeout_seconds=600)

    def evaluate(self, submission: Path) -> JudgeResult:
        torch.set_num_threads(self.resources.cpus)
        print(
            f"[lab1] loading tokenizer and preparing {len(TINY_SHAKESPEARE_SAMPLES)} samples",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        token_ids = [tokenizer.encode(sample) for sample in TINY_SHAKESPEARE_SAMPLES]
        if any(len(ids) < 8 for ids in token_ids):
            raise RuntimeError("A Tiny Shakespeare sample is too short")
        prompts = [
            tokenizer.decode(ids[: 6 + index % 3])
            for index, ids in enumerate(token_ids)
        ]

        sys.path.insert(0, str(submission / "src"))
        try:
            try:
                print(
                    "[lab1] loading student implementation from src/labs/lab1.py",
                    flush=True,
                )
                complete = load_student_function(submission, "lab1").gpt2_complete
                print("[lab1] running student model and batched generation", flush=True)
                completions, logits = complete(prompts, max_seq_length=MAX_SEQ_LENGTH)
                print("[lab1] student generation finished", flush=True)
            except Exception as error:  # noqa: BLE001 - student failures are test failures
                print(
                    f"[lab1] student implementation failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                traceback.print_exc(file=sys.stdout)
                return JudgeResult(
                    passed=False,
                    tests=[
                        TestResult(
                            name="student_function",
                            passed=False,
                            message=f"{type(error).__name__}: {error}",
                        )
                    ],
                )
        finally:
            sys.path.pop(0)

        expected_completions, expected_logits = _reference_completion(
            prompts, MAX_SEQ_LENGTH
        )
        if (
            not isinstance(completions, list)
            or len(completions) != len(prompts)
            or not all(isinstance(text, str) for text in completions)
            or not isinstance(logits, torch.Tensor)
            or logits.shape != expected_logits.shape
        ):
            actual_shape = (
                tuple(logits.shape)
                if isinstance(logits, torch.Tensor)
                else type(logits).__name__
            )
            reason = (
                "Expected (list[str], Tensor) with "
                f"{len(prompts)} completions and logits shaped {tuple(expected_logits.shape)}; "
                f"got completions={type(completions).__name__} "
                f"(length={len(completions) if isinstance(completions, list) else 'n/a'}), "
                f"logits={actual_shape}"
            )
            print(f"[lab1] return contract failed: {reason}", flush=True)
            return JudgeResult(
                passed=False,
                tests=[
                    TestResult(
                        name="return_contract",
                        passed=False,
                        message=reason,
                    )
                ],
            )

        actual = logits.detach().to(device="cpu", dtype=torch.float32)
        expected = expected_logits.to(dtype=torch.float32)
        tests = []
        for index, (actual_text, expected_text) in enumerate(
            tqdm(
                zip(completions, expected_completions, strict=True),
                total=len(prompts),
                desc="[lab1] validating samples",
                file=sys.stdout,
                mininterval=0,
                miniters=1,
            ),
            start=1,
        ):
            logits_match = bool(
                torch.allclose(
                    actual[index - 1],
                    expected[index - 1],
                    atol=LOGIT_ATOL,
                    rtol=LOGIT_RTOL,
                )
            )
            text_match = actual_text == expected_text
            reason = None
            if not logits_match or not text_match:
                max_difference = (
                    (actual[index - 1] - expected[index - 1]).abs().max().item()
                )
                reason = (
                    f"logits_match={logits_match} "
                    f"(max_abs_diff={max_difference:.6g}, atol={LOGIT_ATOL}, rtol={LOGIT_RTOL}); "
                    f"completion_match={text_match} "
                    f"(expected={expected_text!r}, actual={actual_text!r})"
                )
                print(
                    f"[lab1] tiny_shakespeare_{index:02d} failed: {reason}", flush=True
                )
            tests.append(
                TestResult(
                    name=f"tiny_shakespeare_{index:02d}",
                    passed=logits_match and text_match,
                    message=reason,
                )
            )

        passed_count = sum(test.passed for test in tests)
        print(
            f"[lab1] validation complete: {passed_count}/{len(tests)} samples passed",
            flush=True,
        )
        return JudgeResult(
            passed=passed_count == len(tests),
            score=passed_count / len(tests),
            metrics={"samples_passed": float(passed_count)},
            tests=tests,
        )
