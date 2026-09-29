import json
from hashlib import sha256

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoTokenizer

from labs.lab1 import forward, model_name


def format_question(row):
    choices = " ".join(
        f"({letter}) {choice}" for letter, choice in zip("ABCD", row["choices"])
    )
    return f"{row['question']}\n{choices}\nAnswer: "


def get_prompt(row, exemplars):
    subject = row["subject"].replace("_", " ")
    prompt = f"The following are multiple choice questions about {subject}.\n\n"
    for example in exemplars:
        prompt += format_question(example) + "ABCD"[example["answer"]] + "\n\n"
    return prompt + format_question(row)


def mmlu_eval() -> dict[str, str]:
    """Return GPT-2's A/B/C/D prediction for every MMLU test question.

    Load ``cais/mmlu`` at revision
    ``c30699e8356da336a370243923dbaf21066bb9fe``. For each subject, use
    its first four ``dev`` questions as exemplars and evaluate its ``test``
    questions. Format the prompt as specified in the Lab 2 assignment. If a
    prompt exceeds GPT-2's context window, retain its final 1024 tokens.

    Each key is the SHA-256 of a compact UTF-8 JSON object with keys
    ``index``, ``subject``, ``question``, and ``choices`` (sorted keys,
    ``ensure_ascii=False``, compact separators). ``index`` is the zero-based
    row number of the pinned ``all`` test split. This disambiguates repeated
    questions, including 27 identical subject/question/choice rows. Each
    value is one of A/B/C/D, selected from the corresponding next-token
    logits. No question labels should be used to choose a prediction.
    """
    revision = "c30699e8356da336a370243923dbaf21066bb9fe"
    dataset = load_dataset("cais/mmlu", "all", split="test", revision=revision)
    dev = load_dataset("cais/mmlu", "all", split="dev", revision=revision)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    checkpoint = hf_hub_download(model_name, "model.safetensors")
    weights = load_file(checkpoint, device="cuda")

    exemplars = {}
    for example in dev:
        examples = exemplars.setdefault(example["subject"], [])
        if len(examples) < 4:
            examples.append(example)

    answer_token_ids = [
        tokenizer.encode(letter, add_special_tokens=False)[0] for letter in "ABCD"
    ]
    predictions = {}

    with torch.inference_mode():
        for index, row in enumerate(tqdm(dataset, desc="MMLU")):
            prompt = get_prompt(row, exemplars[row["subject"]])
            ids = tokenizer.encode(prompt, add_special_tokens=False)[-1024:]
            token_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            attention_mask = torch.ones_like(token_ids, dtype=torch.bool)

            logits = forward(token_ids, weights, attention_mask)
            choice_logits = logits[0, answer_token_ids]

            payload = {
                "index": index,
                "subject": row["subject"],
                "question": row["question"],
                "choices": row["choices"],
            }
            question_hash = sha256(
                json.dumps(
                    payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            predictions[question_hash] = "ABCD"[int(choice_logits.argmax())]

    return predictions
