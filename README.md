# AlignedGen (Reproduction): FLUX with User Reference Image Style Injection

<!-- [中文](./README.md) | [English](./README_EN.md) -->

This repository is a personal reproduction and extension of **[AlignedGen](https://arxiv.org/abs/2509.17088)**.  
In addition to the original "style sharing across multiple prompts (AAS)", it adds a **user reference image** branch that precomputes reference Q/K/V and injects them step-by-step during generation.

> Note: this repo documents a reproduction and engineering modifications, not the official release mirror.  
> For strict paper-level setup and results, please refer to the [official project page](https://jiexuanz.github.io/AlignedGen/) and the [paper](https://arxiv.org/abs/2509.17088).

---

## Table of Contents

- [What Is Added](#what-is-added)
- [Quick Start](#quick-start)
- [Inference Modes](#inference-modes)
- [Key Arguments](#key-arguments)
- [Code Structure](#code-structure)
- [Acknowledgements and Citation](#acknowledgements-and-citation)
- [License](#license)

---

## What Is Added

| Module | Changes |
|---|---|
| Reference precompute | Precomputes Q/K/V from a user reference image, with caches managed in `aligngen/reference_style.py`. |
| Attention processor | `ShareAttnFluxAttnProcessor2_0` supports `ref_*_override` to override/concatenate reference key-values at each denoising step. |
| Pipeline extension | `FluxPipeline.__call__` in `aligngen/aligned_pipeline.py` adds `style_reference_image`, `reference_prompt`, `reference_cache_generator`, etc. |
| Entry scripts | `inference_reference.py` for "with reference image"; `inference.py` for "without external reference image". |

---

## Quick Start

### 1) Environment setup (example)

```bash
conda create -n aligned python=3.10
conda activate aligned
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
pip install diffusers transformers sentencepiece protobuf==3.19.0
```

### 2) Model access

Make sure `black-forest-labs/FLUX.1-dev` is accessible (local path or Hugging Face Hub).

---

## Inference Modes

### A. No external reference image (close to original AAS)

```bash
python inference.py \
  --model_path black-forest-labs/FLUX.1-dev \
  --style_lambda 1.1
```

### B. With user reference image (recommended)

```bash
python inference_reference.py \
  --model_path black-forest-labs/FLUX.1-dev \
  --style_reference_image path/to/your_style.jpg \
  --reference_prompt "A short prompt aligned with the reference content/style" \
  --style_lambda 1.1 \
  --num_inference_steps 30 \
  --output_dir output_ref
```

---

## Key Arguments

| Argument | Purpose | Recommendation |
|---|---|---|
| `--style_reference_image` | Path to style reference image | Use a clean RGB image with clear style cues. |
| `--reference_prompt` | Text encoding for reference branch only | Avoid leaving empty; include style/subject hints. |
| `--style_lambda` | Style injection strength | Typical starting range: `1.0 ~ 1.2`. |
| `--cache_seed` | Noise seed for reference branch | Keep fixed for reproducible experiments. |
| `--reference_kv_precompute_mode` | Reference KV precompute strategy | `appendix_a` or `denoise`. |
| `--num_inference_steps` | Number of denoising steps | Increase for quality, reduce for speed. |

---

## Code Structure

```text
aligngen/
  reference_style.py       # Reference KV precompute and per-step injection
  attention_processor.py   # ShareAttnFluxAttnProcessor2_0 (AdaIN / override / concat)
  aligned_pipeline.py      # Main pipeline flow (reference branch + denoising loop)
  aligned_transformer.py   # Transformer and attention processor integration
inference_reference.py     # Entry point with reference image
inference.py               # Entry point without external reference image
```

---

## Acknowledgements and Citation

This implementation builds on open-source projects such as [StyleAligned](https://github.com/google/style-aligned) and [Diffusers](https://github.com/huggingface/diffusers).  
If you cite the **AlignedGen method**, please cite the original paper:

```bibtex
@article{zhang2025alignedgen,
  title={AlignedGen: Aligning Style Across Generated Images},
  author={Zhang, Jiexuan and Du, Yiheng and Wang, Qian and Li, Weiqi and Gu, Yu and Zhang, Jian},
  journal={arXiv preprint arXiv:2509.17088},
  year={2025}
}
```

If you use this repository in papers/reports, consider describing it as "a reproduction/modification based on AlignedGen."

---

## License

Please comply with:

- File-level licenses in this repository (for example Apache 2.0 where specified)
- Usage terms of the models and dependencies (for example FLUX / diffusers)
