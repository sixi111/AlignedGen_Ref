# AlignedGen（复现版）: FLUX + 用户参考图风格注入

<!-- [中文](./README.md) | [English](./README_EN.md) -->

基于 **[AlignedGen](https://arxiv.org/abs/2509.17088)** 的个人复现与扩展：  
在原始「多 prompt 共享风格注意力（AAS）」能力上，新增了**用户参考图**分支，用于预计算参考 Q/K/V 并在主生成阶段按步注入。

> 说明：本仓库描述的是复现与工程改动，不是官方发行版镜像。  
> 若需严格对齐论文结果与官方设定，请优先查看 [官方项目页](https://jiexuanz.github.io/AlignedGen/) 与 [论文](https://arxiv.org/abs/2509.17088)。

---

## 目录

- [核心改动](#核心改动)
- [快速开始](#快速开始)
- [推理方式](#推理方式)
- [关键参数说明](#关键参数说明)
- [代码结构](#代码结构)
- [致谢与引用](#致谢与引用)
- [许可证](#许可证)

---

## 核心改动

| 模块 | 改动内容 |
|---|---|
| 参考图预计算 | 对用户参考图执行 Q/K/V 预计算，缓存由 `aligngen/reference_style.py` 管理。 |
| 注意力处理器 | `ShareAttnFluxAttnProcessor2_0` 支持 `ref_*_override`，在主去噪中按步覆盖/拼接参考键值。 |
| Pipeline 扩展 | `aligngen/aligned_pipeline.py` 的 `FluxPipeline.__call__` 增加 `style_reference_image`、`reference_prompt`、`reference_cache_generator` 等参数。 |
| 入口脚本 | `inference_reference.py` 用于“带参考图”模式；`inference.py` 保留“无外部参考图”模式。 |

---

## 快速开始

### 1) 环境安装（示例）

```bash
conda create -n aligned python=3.10
conda activate aligned
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
pip install diffusers transformers sentencepiece protobuf==3.19.0
```

### 2) 模型准备

确保可以访问 `black-forest-labs/FLUX.1-dev`（本地路径或 Hugging Face Hub）。

---

## 推理方式

### A. 无外部参考图（接近原版 AAS）

```bash
python inference.py \
  --model_path black-forest-labs/FLUX.1-dev \
  --style_lambda 1.1
```

### B. 带用户参考图（推荐）

```bash
python inference_reference.py \
  --model_path black-forest-labs/FLUX.1-dev \
  --style_reference_image path/to/your_style.jpg \
  --reference_prompt "与参考图内容/风格一致的短描述" \
  --style_lambda 1.1 \
  --num_inference_steps 30 \
  --output_dir output_ref
```

---

## 关键参数说明

| 参数 | 作用 | 建议 |
|---|---|---|
| `--style_reference_image` | 参考风格图路径 | 使用风格清晰、压缩较少的 RGB 图片。 |
| `--reference_prompt` | 仅用于参考分支编码 | 不建议留空，写明风格/主体更稳定。 |
| `--style_lambda` | 风格注入强度 | 常见起点 `1.0 ~ 1.2`。 |
| `--cache_seed` | 参考分支噪声种子 | 固定后便于复现实验。 |
| `--reference_kv_precompute_mode` | 参考 KV 预计算策略 | `appendix_a`（论文附录 A）或 `denoise`。 |
| `--num_inference_steps` | 采样步数 | 质量优先可提高，速度优先可降低。 |

---

## 代码结构

```text
aligngen/
  reference_style.py       # 参考 KV 预计算与按步注入
  attention_processor.py   # ShareAttnFluxAttnProcessor2_0（AdaIN / 覆盖 / 拼接）
  aligned_pipeline.py      # FluxPipeline 主流程（参考图分支 + 去噪循环）
  aligned_transformer.py   # Transformer 结构与 attn processor 接口
inference_reference.py     # 带参考图入口
inference.py               # 无外部参考图入口
```

---

## 致谢与引用

实现依赖 [StyleAligned](https://github.com/google/style-aligned) 与 [Diffusers](https://github.com/huggingface/diffusers) 等开源生态。  
若引用 **AlignedGen 方法本身**，请使用原论文：

```bibtex
@article{zhang2025alignedgen,
  title={AlignedGen: Aligning Style Across Generated Images},
  author={Zhang, Jiexuan and Du, Yiheng and Wang, Qian and Li, Weiqi and Gu, Yu and Zhang, Jian},
  journal={arXiv preprint arXiv:2509.17088},
  year={2025}
}
```

若在论文/报告中使用本仓库实现，建议标注为“基于 AlignedGen 的复现/修改版本”。

---

## 许可证

请同时遵守：

- 本仓库各文件头声明的许可证（如 Apache 2.0）
- 所使用模型与依赖（如 FLUX / diffusers）的使用条款
