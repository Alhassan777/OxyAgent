# PaliGemma Model Load Issue

## Root Cause

Your local `paligemma2-3b-pt-224` folder contains **weight files from a different model variant**:

| Component | Your weights (ckpt) | Expected (224pt config) |
|-----------|----------------------|--------------------------|
| Vision MLP | 4304 dim             | 4096 dim                 |
| Projector  | 2304 dim             | 2048 dim                 |

This matches **paligemma2-3b-pt-384** or **paligemma2-3b-mix** architecture, not 224pt.

The `config.json` in your folder + transformers defaults define 4096/2048, but the `.safetensors` files have 4304/2304.

## Fix Options

1. **Use HuggingFace Hub** (recommended) – Downloads the correct model once (~6GB) and caches it
2. **Replace local weights** – Download correct 224pt weights into `paligemma2-3b-pt-224/`
3. **Use matching base for your LoRA** – Your LoRA was trained on the 4304/2304 variant; use that base model
