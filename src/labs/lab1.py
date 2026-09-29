import math

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer

model_name = "openai-community/gpt2"


def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    """Generate greedy completions with a from-scratch GPT-2 Small implementation.

    Load pretrained GPT-2 Small weights into manually implemented transformer
    blocks. Generate for the entire batch at once, choosing the highest-logit
    token for every unfinished sequence at each step. Stop each sequence at EOS
    or max_seq_length total tokens, including the prompt.

    Return newly generated text for each prompt and a tensor of pre-selection
    logits shaped (batch_size, decoding_steps, 50257). Fill logits with zero
    after a row has finished while other rows continue.
    """

    # Load bunch of file and weight with initial
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model Loaded")

    # Generate Part
    prompt_ids = [tokenizer.encode(prompt) for prompt in input]
    lengths = torch.tensor([len(ids) for ids in prompt_ids], dtype=torch.long)
    print("Tokenizer finished")

    batch_size = len(input)
    finished = lengths >= max_seq_length

    gpt_2_checkpoint = hf_hub_download(model_name, "model.safetensors")
    print("GPT-2 checkpoint downloaded")

    weights = {
        name: tensor.half()
        for name, tensor in load_file(gpt_2_checkpoint, "cpu").items()
    }
    print("Weights loaded")

    eos_id = tokenizer.eos_token_id
    width = int(lengths.max())

    # fill tensor with eos
    input_ids = torch.full((batch_size, width), eos_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(prompt_ids):
        input_ids[row, -len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, -len(ids) :] = 1

    generated: list[list[int]] = [[] for _ in input]
    steps: list[torch.Tensor] = []

    with torch.inference_mode():
        while not bool(finished.all()):
            logits = forward(input_ids, weights, attention_mask)
            active = ~finished
            # we do this since the lab require we fill with 0
            logits[~active] = 0
            steps.append(logits)
            next_ids = logits.argmax(dim=-1)
            next_ids[~active] = eos_id

            for row in range(batch_size):
                if bool(active[row]):
                    generated[row].append(int(next_ids[row]))
            lengths += active.long()
            # check if it was finish or not
            finished = finished | (next_ids == eos_id) | (lengths >= max_seq_length)

            input_ids = torch.cat((input_ids, next_ids[:, None]), dim=1)
            attention_mask = torch.cat((attention_mask, active[:, None].long()), dim=1)

    completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    if not steps:
        return completions, torch.empty(
            (batch_size, 0, 50257), dtype=weights["wte.weight"].dtype
        )
    return completions, torch.stack(steps, dim=1)


def forward(input_ids, weights, attention_mask):
    batch_size, length = input_ids.shape
    positions = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)

    x = weights["wte.weight"][input_ids] + weights["wpe.weight"][positions]

    mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).tril()
    mask = mask[None, None, :, :] & attention_mask[:, None, None, :].bool()

    # use 12 layer
    for layer in range(12):
        prefix = f"h.{layer}"

        normalized = layer_norm(x, weights, prefix + ".ln_1")
        qkv = linear(normalized, weights, prefix + ".attn.c_attn")
        query, key, value = qkv.chunk(3, dim=-1)

        # since we need to use multi head we need to shape it to 64 chunk.
        query = query.reshape(batch_size, length, 12, 64).transpose(1, 2)
        key = key.reshape(batch_size, length, 12, 64).transpose(1, 2)
        value = value.reshape(batch_size, length, 12, 64).transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(64)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = probabilities.masked_fill(~mask, 0.0)
        context = probabilities @ value

        context = context.transpose(1, 2).reshape(batch_size, length, 768)
        x = x + linear(context, weights, prefix + ".attn.c_proj")

        normalized = layer_norm(x, weights, prefix + ".ln_2")
        hidden = linear(normalized, weights, prefix + ".mlp.c_fc")
        # gelu
        hidden = (
            0.5
            * hidden
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden**3)))
        )
        x = x + linear(hidden, weights, prefix + ".mlp.c_proj")

    x = layer_norm(x, weights, "ln_f")

    logits = x[:, -1, :] @ weights["wte.weight"].T
    return logits


def linear(x, weights, name):
    weight = weights[name + ".weight"]
    bias = weights[name + ".bias"]
    output_shape = (*x.shape[:-1], weight.shape[1])
    return torch.addmm(bias, x.reshape(-1, x.shape[-1]), weight).reshape(output_shape)


def layer_norm(x, weights, name):
    return nn.functional.layer_norm(
        x,
        # 768 since we just have 768 dimensions vector
        normalized_shape=(768,),
        weight=weights[name + ".weight"],
        bias=weights[name + ".bias"],
        eps=1e-5,
    )
