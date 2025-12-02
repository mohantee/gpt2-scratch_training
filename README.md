# GPT-2 Training (minGPT-style)

This repository contains a compact GPT-2 style implementation (minGPT-like) and a Jupyter notebook (`train_nb.ipynb`) that demonstrates model architecture, training loop, and a minimal generation example. The code is intended for experimentation, education, and small-scale training on text data stored in `input.txt`.

## Contents

- `train_nb.ipynb` — main notebook with model definition, training loop, and generation example.
- `train.py` — same implementation as a python Module
- `input.txt` — training data (plain text) used by the lightweight DataLoader in the notebook.
- `pyproject.toml` — basic project metadata.

## Project overview

The notebook implements a compact GPT model similar to karpathy's minGPT:

- Transformer blocks with multi-head causal self-attention and MLP (GELU).
- Token and positional embeddings, final LayerNorm and a linear language model head.
- Custom weight initialization which scales some linear layers (noted by `NANOGPT_SCALE_INIT`).
- A tiny `DataLoaderLite` that encodes `input.txt` using `tiktoken` (GPT-2 BPE) and yields contiguous batches.

The training loop in the notebook includes:

- AdamW optimizer with weight decay.
- Learning rate schedule: linear warmup followed by cosine decay.
- Gradient accumulation and gradient clipping.
- Early stopping condition based on a target average loss.

## Requirements

These are the packages required to run the notebook. The `pyproject.toml` currently contains no listed dependencies, so install the packages below manually.

Recommended Python: 3.8+ (the `pyproject.toml` lists `requires-python = ">=3.13"` — if you need to adhere to that, use an appropriate Python runtime; otherwise Python 3.8+ is sufficient for PyTorch and tooling).

Core Python packages:

 - torch (PyTorch) — install the appropriate build for your OS and CUDA version
 - tiktoken — tokenizer used to encode `input.txt` to GPT-2 tokens
 - transformers (optional) — used by `GPT.from_pretrained()` to import HuggingFace weights
 - torchsummary (optional) — used for model summaries inside the notebook

Install with pip (CPU-only example):

```powershell
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tiktoken transformers torchsummary
```

For CUDA-enabled PyTorch, follow the official install instructions at https://pytorch.org to get the matching wheel for your CUDA version.

## How to run

Open the notebook `train_nb.ipynb` in Jupyter or VS Code and run the cells in order.

High-level steps performed in the notebook:

1. Imports and device setup (CPU / CUDA / MPS detection).
2. Model definitions: `CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`, `GPT`.
3. `DataLoaderLite` encodes `input.txt` with `tiktoken` and provides next_batch().
4. Instantiate `model = GPT(GPTConfig())` and move to the device.
5. Create optimizer: AdamW with lr=6e-4 and weight decay=0.1.
6. Training loop with warmup, cosine decay, gradient accumulation and logging.
7. After training the notebook demonstrates autoregressive generation using top-k sampling.

If you prefer running a script, inspect `train.py` for a runnable entry point (it may require small adapations depending on how you want to provide arguments and device settings).

## Key hyperparameters (as used in the notebook)

- Batch size (B): 32
- Sequence length (T): 256
- Learning rate: 6e-4 (with warmup and cosine decay)
- Weight decay: 0.1
- Warmup steps: 500
- Gradient accumulation steps: 4
- Max training steps: 10000 (adjust as needed)
- Target early-stop avg loss: 0.099 (noted in the notebook)

Model default config (GPTConfig):

- block_size: 1024
- vocab_size: 50257
- n_layer: 12
- n_head: 12
- n_embd: 768

These values correspond to a small GPT-2 style model (roughly the GPT-2 124M configuration). You can reduce layers/heads/embedding size to fit resource constraints when experimenting locally.

## Checkpoints & outputs

The notebook does not automatically save checkpoints to disk. If you want to persist model weights, add code in the training loop to save `model.state_dict()` periodically, for example:

```python
torch.save(model.state_dict(), 'checkpoint_step_{:06d}.pt'.format(step))
```

For generation, the notebook demonstrates autoregressive sampling using top-k=50 and `torch.multinomial` over the top-k probabilities. It prints `num_return_sequences` generated sequences of `max_length` tokens.

## Notes, assumptions and caveats

- The notebook uses `tiktoken.get_encoding('gpt2')` to tokenize `input.txt`. Make sure `input.txt` is plain text and large enough for meaningful training.
- The training loop in the notebook is educational and minimal — for production-scale training you should add proper checkpointing, evaluation, mixed precision (AMP), distributed training, and dataset streaming.
- The `pyproject.toml` currently lists no dependencies; update it if you want to manage dependencies via Poetry or PEP 621 tooling.
- Assumption: Python 3.8+ works; adjust to your environment if your runtime requires a different version.

## Suggested next steps / improvements

- Add periodic checkpointing and evaluation on a validation split.
- Add a command-line training script or a small training harness in `train.py` that accepts arguments (batch size, learning rate, output dir).
- Add mixed precision training (torch.cuda.amp) to speed up training on GPUs.
- Add logging with TensorBoard or Weights & Biases for better run tracking.

## License

This repository contains study/example code. Add a license file if you want to specify reuse terms (e.g., MIT, Apache-2.0).

## Contact

If you have questions about the code, open an issue in this repository or contact the maintainer.


