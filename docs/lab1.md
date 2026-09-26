# Reading the Lab 1 implementation

The implementation is in [src/labs/lab1.py](../src/labs/lab1.py). It performs
inference with pretrained GPT-2 Small weights. There is no training loop.
Hugging Face supplies the tokenizer and checkpoint; our code implements the
transformer and greedy generation.

## Read the code in this order

1. `GPT2.forward`: follow token IDs through the complete model.
2. `TransformerBlock.forward`: see how attention and the MLP update each token.
3. `CausalSelfAttention.forward`: follow the head dimensions and masks.
4. `GPT2Linear` and `GPT2.load_weights`: see how pretrained numbers enter layers.
5. `gpt2_complete`: follow a batch through repeated next-token predictions.

Before reading attention, get comfortable with matrix multiplication, tensor
shapes, reshaping, and transposing. A token is an integer representing a piece
of text; an embedding is a learned vector selected using that integer.

## Follow the shapes

Let `B` be the number of prompts and `T` the current padded sequence length.
GPT-2 Small uses 12 blocks, 12 attention heads, 768 hidden features, and a
vocabulary of 50,257 tokens.

| Value | Shape | Meaning |
| --- | --- | --- |
| `input_ids` | `[B, T]` | Integer tokens |
| `attention_mask` | `[B, T]` | Which positions contain actual tokens |
| `x` | `[B, T, 768]` | One vector per position |
| Query, key, value after splitting heads | `[B, 12, T, 64]` | Each head gets 768 / 12 features |
| Attention scores | `[B, 12, T, T]` | Each query position's scores for all key positions |
| Model output | `[B, T, 50257]` | Raw vocabulary scores at every position |
| One generation step | `[B, 50257]` | Last-position scores predicting the next token |
| Returned logits | `[B, steps, 50257]` | Scores saved across generation steps |

`GPT2.forward` adds token and learned position embeddings. Each block then
computes `x + attention(layer_norm(x))`, followed by
`x + mlp(layer_norm(x))`. Layer normalization normalizes each token's feature
vector and applies learned scale and bias. Residual addition lets each
sublayer add an update to the existing representation.

Attention computes `softmax(Q @ K.T / sqrt(64) + mask) @ V` for each head.
Queries and keys determine relevance; values supply the information gathered.
The MLP transforms each position independently, expanding 768 features to
3072, applying GPT-2's GELU activation, and returning to 768.

The final vocabulary projection reuses the token embedding matrix. This is
weight tying: the same learned table represents tokens on input and scores
them on output. Logits are raw scores; greedy selection only needs `argmax`.
There is no need to convert the output logits to probabilities.

## Why the details matter

- **Causal masking:** position `i` may read positions up to `i`, never later ones.
- **Padding masking:** added padding must not influence real tokens. Left
  padding aligns the prompts' final tokens so `[:, -1, :]` selects their
  next-token scores. Position IDs count real tokens, not padding columns.
- **Checkpoint orientation:** `GPT2Linear` keeps the checkpoint's matrix shape
  `[input_features, output_features]` and computes `bias + x @ weight` with
  `torch.addmm`. PyTorch `nn.Linear` stores the transpose. Keeping GPT-2's layout
  and fused bias addition also reproduces the reference's fp16 rounding.
- **Attention primitive:** we construct Q/K/V, heads, and masks ourselves, then
  use `F.scaled_dot_product_attention` for the equation above. Its kernel handles
  masking, scaling, softmax, and the weighted sum, including fully masked rows.
- **Precision:** weights and layer outputs use fp16 to match the judge. Using
  the same attention kernel matters: an explicit sequence of mathematically
  equivalent tensor operations can round differently and change a close argmax.
- **Inference:** dropout is omitted and generation uses `torch.inference_mode()`
  so PyTorch does not retain a backward computation graph.

For example, assume two prompts have 2 and 3 tokens and the total limit is 4.
With no early EOS, they generate 2 and 1 tokens respectively. The result has
shape `[2, 2, 50257]`, and `logits[1, 1]` is zero because that row has finished.
When a row emits EOS, its logits for that step are retained, but the decoded
text excludes the special token. Only subsequent steps are zeroed.

## Run the checks

From the repository root:

```bash
uv sync --frozen
uv run python -m pytest -q tests/test_lab1_implementation.py
uv run python -m judge.run_task --task lab1 --submission . --output /tmp/lab1-result.json
cat /tmp/lab1-result.json
```

The implementation tests use tiny random reference models and scripted
generation outputs, so they need no model download. They check numerical
agreement, causality, padding, EOS, and per-row length limits. The actual judge
downloads GPT-2 and compares all 20 Shakespeare samples. Inspect `passed` in
the JSON: the runner's exit code alone does not indicate a passing grade.
This command does not enforce the deployed worker's 600-second limit. Check
runtime separately before submitting, especially on CPUs without fast fp16
operations; the reference model's computation also counts toward that limit.

This first version recomputes the entire sequence each step. It makes the data
flow explicit but gets expensive for long completions. A KV cache can later
reuse previous keys and values. Empty batches are supported; empty prompts
are rejected explicitly. Both prompt length and the requested total length
must fit within the 1024-token context window.

## Check your understanding

1. Why does `Q @ K.T` produce a `T x T` matrix for each head?
2. Why must the same sentence keep its position IDs when left padding changes?
3. Why are logits retained on the step that generates EOS, then zeroed later?
