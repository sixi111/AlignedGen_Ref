# AlignedGen（复现）— 带用户参考图的 FLUX 版本

本仓库是在 **[AlignedGen](https://arxiv.org/abs/2509.17088)**（NeurIPS 2025，PKU 团队）工作基础上的**个人复现与扩展**：在原有「多 prompt 共享风格注意力（AAS）」之上，增加了**用户给定参考图**时的风格注入流程，便于做实验与对比。

> 若你关心论文原始设定与更多展示，请优先阅读 [官方项目页](https://jiexuanz.github.io/AlignedGen/) 与 [arXiv:2509.17088](https://arxiv.org/abs/2509.17088)。本 README 仅描述**本仓库**的行为与用法。

---

## 与官方代码的主要差异（本仓库做了什么）

| 能力 | 说明 |
|------|------|
| **外部参考图** | 对用户提供的风格参考图做 **Q/K/V 预计算**（见 `aligngen/reference_style.py`），在主生成每一步将缓存写入 `ShareAttnFluxAttnProcessor2_0`（`ref_*_override`），与 AAS 中的 AdaIN + 拼接键一致。 |
| **Pipeline** | `aligngen/aligned_pipeline.py` 中 `FluxPipeline.__call__` 增加 `style_reference_image`、`reference_prompt`、`reference_cache_generator`、`reference_kv_precompute_mode` 等参数。 |
| **预计算模式** | `reference_kv_precompute_mode="appendix_a"`：论文附录 A 的噪声–参考 latent **插值**轨迹；`"denoise"`：**纯噪声 + scheduler** 逐步去噪并每步记录 KV（用于验证轨迹是否与主推理一致）。 |
| **推理入口** | `inference_reference.py`：带参考图的批量生成；`inference.py`：仍接近原版「仅 batch 内风格对齐」用法（若保留）。 |

本仓库**不是**官方 AlignedGen 发行版的镜像；若要与论文表格严格对齐，请以原作者发布版本为准。

---

## 环境依赖（示例）

```bash
conda create -n aligned python=3.10
conda activate aligned
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
pip install diffusers transformers sentencepiece protobuf==3.19.0
```

需能访问 **FLUX.1-dev**（本地路径或 Hub）。

---

## 推理

### 1. 原版风格：多 prompt、无外部参考图

```bash
python inference.py --model_path black-forest-labs/FLUX.1-dev --style_lambda 1.1
```

（具体参数以 `inference.py` 内 argparse 为准。）

### 2. 本仓库重点：用户参考图 + 预计算 KV 注入

```bash
python inference_reference.py \
  --model_path black-forest-labs/FLUX.1-dev \
  --style_reference_image path/to/your_style.jpg \
  --reference_prompt "与参考图内容/风格一致的短描述" \
  --style_lambda 1.1 \
  --num_inference_steps 30 \
  --output_dir output_ref
```

常用参数含义简述：

- **`--reference_prompt`**：仅用于**预计算参考分支**时的文本编码；过空时参考侧语义弱，建议写清风格/内容。
- **`--cache_seed`**：附录 A 插值里的噪声种子；在 **`denoise`** 预计算模式下也用于**初始 latent** 的高斯噪声。
- **`--reference_kv_precompute_mode`**：`appendix_a`（默认）或 `denoise`（验证用全 scheduler 轨迹）。

显存不足时可减少 prompt 条数、降低分辨率，或在 pipeline 内按需开启 offload（若你已接入）。

---

## 代码结构（与参考图相关）

```
aligngen/
  reference_style.py    # 参考 KV 预计算、apply_kv_cache_for_step
  attention_processor.py  # ShareAttnFluxAttnProcessor2_0：AdaIN、ref 覆盖、拼接
  aligned_pipeline.py   # FluxPipeline：参考图分支与去噪循环
  aligned_transformer.py
inference_reference.py  # 带参考图的命令行入口
inference.py            # 无外部参考时的入口（若保留）
```

---

## 致谢与引用

实现依赖 [StyleAligned](https://github.com/google/style-aligned) 与 [Diffusers](https://github.com/huggingface/diffusers) 等开源生态。**AlignedGen 方法本身**请引用原论文：

```bibtex
@article{zhang2025alignedgen,
  title={AlignedGen: Aligning Style Across Generated Images},
  author={Zhang, Jiexuan and Du, Yiheng and Wang, Qian and Li, Weiqi and Gu, Yu and Zhang, Jian},
  journal={arXiv preprint arXiv:2509.17088},
  year={2025}
}
```

若你在论文或报告中使用本仓库的**参考图扩展实现**，建议在正文中说明为「基于 AlignedGen 的复现/修改版本」，并区分于官方发布代码。

---

## 许可证

请同时遵守本仓库内各文件头声明的许可证（如 Apache 2.0 等）以及你所使用的 **FLUX / diffusers** 模型的使用条款。
