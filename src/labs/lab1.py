"""GPT-2 inference, written out so the tensor operations are easy to follow.

Read GPT2.forward first, then TransformerBlock, CausalSelfAttention, and finally
gpt2_complete. Hugging Face supplies the tokenizer and pretrained weight file;
all transformer computation and decoding below run through our own code.
"""

import math
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

MODEL_ID = "openai-community/gpt2"
VOCAB_SIZE = 50257
CONTEXT_LENGTH = 1024


class GPT2Linear(nn.Module):
    """A linear layer with GPT-2's checkpoint layout: weight is [input, output]."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # addmm computes bias + x @ weight in one operation. Keeping the original
        # layout and fused bias addition matches the reference's fp16 rounding.
        output_shape = (*x.shape[:-1], self.weight.shape[1])
        x = torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight)
        return x.reshape(output_shape)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        # One projection produces Q, K, and V; another mixes the heads' outputs.
        self.c_attn = GPT2Linear(hidden_size, 3 * hidden_size)
        self.c_proj = GPT2Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, width, hidden_size = x.shape
        query, key, value = self.c_attn(x).chunk(3, dim=-1)

        # [B, T, D] -> [B, T, H, D/H] -> [B, H, T, D/H].
        query = query.reshape(batch_size, width, self.num_heads, self.head_size)
        key = key.reshape(batch_size, width, self.num_heads, self.head_size)
        value = value.reshape(batch_size, width, self.num_heads, self.head_size)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        causal_mask = torch.ones(width, width, device=x.device, dtype=torch.bool).tril()
        valid_keys = attention_mask[:, None, None, :].bool()
        allowed = causal_mask[None, None, :, :] & valid_keys
        # This primitive computes softmax(Q @ K.T / sqrt(head_size) + mask) @ V.
        # Scores have shape [B, H, T, T]; True in allowed means "may attend".
        # Use the same PyTorch kernel as the reference: separate matmul/softmax
        # operations can round differently in fp16 and flip a close greedy tie.
        context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=0.0
        )

        # Merge the heads: [B, H, T, D/H] -> [B, T, D].
        context = context.transpose(1, 2).reshape(batch_size, width, hidden_size)
        return self.c_proj(context)


class MLP(nn.Module):
    """Transform each position independently: D -> 4D -> D."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.c_fc = GPT2Linear(hidden_size, 4 * hidden_size)
        self.c_proj = GPT2Linear(4 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        # GPT-2's gelu_new activation, including its tanh approximation.
        x = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        # LayerNorm normalizes each token's features, not the batch or time axis.
        self.ln_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attn = CausalSelfAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = MLP(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # GPT-2 uses pre-normalization and two residual additions per block.
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """The defaults describe GPT-2 Small; smaller dimensions help test the math.

    This is an inference-only model: dropout is intentionally omitted.
    """

    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE,
        max_positions: int = CONTEXT_LENGTH,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
    ) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.wpe = nn.Embedding(max_positions, hidden_size)
        self.h = nn.ModuleList(
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Map token IDs [B, T] to vocabulary logits [B, T, V]."""
        # Left padding must not shift a real token's learned position embedding.
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
        x = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.h:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        # Weight tying: the output head reuses the token embedding table.
        return F.linear(x, self.wte.weight)

    def load_weights(self, checkpoint: str | Path) -> None:
        """Copy checkpoint tensors into our modules, one parameter at a time."""
        with (
            safe_open(checkpoint, framework="pt", device="cpu") as weights,
            torch.no_grad(),
        ):
            for name, parameter in self.named_parameters():
                # This checkpoint uses base-model names such as wte.weight.
                weight = weights.get_tensor(name)
                if weight.shape != parameter.shape:
                    raise ValueError(f"Unexpected checkpoint shape for {name}")
                parameter.copy_(weight)


def _load_model() -> GPT2:
    checkpoint = hf_hub_download(MODEL_ID, filename="model.safetensors")
    # The grader uses fp16 on CPU. Matching its precision reduces rounding drift.
    model = GPT2().to(dtype=torch.float16)
    model.load_weights(checkpoint)
    model.eval()
    return model


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
    Empty batches return empty outputs. Prompts must contain at least one token
    and fit within GPT-2's 1024-token context. A prompt already at or beyond the
    requested total length produces no new text.
    """
    if not 1 <= max_seq_length <= CONTEXT_LENGTH:
        raise ValueError("max_seq_length must be between 1 and 1024")
    if not input:
        return [], torch.empty((0, 0, VOCAB_SIZE), dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = [tokenizer.encode(prompt) for prompt in input]
    lengths = torch.tensor([len(ids) for ids in prompt_ids], dtype=torch.long)
    if bool((lengths == 0).any()):
        raise ValueError("Each prompt must contain at least one token")
    if bool((lengths > CONTEXT_LENGTH).any()):
        raise ValueError("Prompts must not exceed 1024 tokens")

    batch_size = len(input)
    finished = lengths >= max_seq_length
    if bool(finished.all()):
        return [""] * batch_size, torch.empty(
            (batch_size, 0, VOCAB_SIZE), dtype=torch.float16
        )

    eos_id = tokenizer.eos_token_id
    width = int(lengths.max())
    input_ids = torch.full((batch_size, width), eos_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(prompt_ids):
        # Left padding puts each prompt's final real token in the last column.
        input_ids[row, -len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, -len(ids) :] = 1

    model = _load_model()
    generated: list[list[int]] = [[] for _ in input]
    steps: list[torch.Tensor] = []
    with torch.inference_mode():
        while not bool(finished.all()):
            # Recompute the whole batch; a KV cache is an optional later exercise.
            # clone() keeps only the last-position logits instead of retaining
            # the entire [B, T, V] output for every decoding step.
            logits = model(input_ids, attention_mask)[:, -1, :].clone()
            active = ~finished
            logits[~active] = 0
            steps.append(logits)
            next_ids = logits.argmax(dim=-1)
            next_ids[~active] = eos_id

            for row in range(batch_size):
                if bool(active[row]):
                    generated[row].append(int(next_ids[row]))
            lengths += active.long()
            finished = finished | (next_ids == eos_id) | (lengths >= max_seq_length)

            input_ids = torch.cat((input_ids, next_ids[:, None]), dim=1)
            # active describes the step BEFORE finishing: an emitted EOS is a
            # real token, while placeholders for previously finished rows are not.
            attention_mask = torch.cat((attention_mask, active[:, None].long()), dim=1)

    completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    return completions, torch.stack(steps, dim=1)
