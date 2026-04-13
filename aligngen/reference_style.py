"""
论文附录 A Algorithm 1：对用户给定参考图做前向加噪插值，并在每层 DiT 中缓存
用于 AAS 拼接的图像 token 上的 K/V（与当前推理步对齐）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from diffusers.utils.torch_utils import randn_tensor

if TYPE_CHECKING:
    from PIL import Image

    from aligngen.aligned_pipeline import FluxPipeline

# 每层每步：(Q_img^ref, K_img^ref 于 AdaIN 前, k_/v_ 于 additional RoPE 后，用于拼接)
RefKvCache = Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]


def _aligned_canvas_hw(pipe: "FluxPipeline", height: int, width: int) -> Tuple[int, int]:
    """与 `prepare_latents` 一致：返回 **latent 空间** 对齐后的高宽（非像素）。"""
    h = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    w = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    return h, w


def _encode_packed_latents(
    pipe: "FluxPipeline",
    image: "Image.Image",
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int]:
    """与 prepare_latents 相同 latent 画布；返回 packed latent 与 latent 空间 (lh, lw)。

    注意：`_prepare_latent_image_ids` 的网格尺寸必须是 **latent 高宽** 的 `//2`，
    不能用像素对齐后的 ah//2（否则会与 packed 序列长度不一致，导致 RoPE 维度错误）。

    `image_processor.preprocess` 需要的是 **像素** 高宽；`_aligned_canvas_hw` 得到的是 **latent** 高宽，
    必须乘以 `vae_scale_factor`（通常 8），否则会误把 latent 尺寸当成像素，得到 16×16 latent 等小图。
    """
    lh, lw = _aligned_canvas_hw(pipe, height, width)
    pixel_h = lh * pipe.vae_scale_factor
    pixel_w = lw * pipe.vae_scale_factor
    image_tensor = pipe.image_processor.preprocess(image, height=pixel_h, width=pixel_w)
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    enc = pipe.vae.encode(image_tensor)
    latents = enc.latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    b, c, lh, lw = latents.shape
    latents = pipe._pack_latents(latents, b, c, lh, lw)
    return latents, lh, lw


def clear_ref_overrides(pipe: "FluxPipeline") -> None:
    for proc in pipe.transformer.attn_processors.values():
        if hasattr(proc, "clear_ref_override"):
            proc.clear_ref_override()


def precompute_style_reference_kv_cache(
    pipe: "FluxPipeline",
    ref_image: "Image.Image",
    ref_prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    timesteps: torch.Tensor,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float,
    max_sequence_length: int,
) -> RefKvCache:
    """
    按论文式对参考图 latent 与随机噪声做线性插值，在每个推理步上跑一次 batch=1 的 transformer，
    并从 ShareAttnFluxAttnProcessor2_0 中捕获
    (Q_img^ref, K_img^ref 于 AdaIN 前, k_/v_ 于 additional RoPE 后)，与论文 Concat 及 AdaIN(·,·^ref) 一致。
    """
    names = list(pipe.transformer.attn_processors.keys())
    cache: RefKvCache = {n: [] for n in names}

    ref_latent, lh, lw = _encode_packed_latents(pipe, ref_image, height, width, dtype, device)  # 例：1024px→latent 128²→pack (4096,64)
    noise = randn_tensor(ref_latent.shape, generator=generator, device=device, dtype=dtype)

    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipe.encode_prompt(
        prompt=ref_prompt,
        prompt_2=ref_prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )

    # 与 aligned_pipeline.prepare_latents 一致：grid 为 (latent_h//2, latent_w//2)
    latent_image_ids = pipe._prepare_latent_image_ids(1, lh // 2, lw // 2, device, dtype)
    reference_image_ids = latent_image_ids.clone()
    # 与 aligned_pipeline.__call__ 中一致：使用用户传入的 width（未做 latent 对齐前）
    reference_image_ids[:, 2] += width // 16

    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance = None

    for i, t in enumerate(timesteps):
        # 使用 scheduler 实际的 sigma（= t/1000）做插值，与主去噪循环的噪声水平对齐
        sigma = t.item() / 1000.0
        noisy_latent = sigma * noise + (1.0 - sigma) * ref_latent

        timestep = t.expand(1).to(noisy_latent.dtype)
        for name in names:
            proc = pipe.transformer.attn_processors[name]
            proc.set_timesteps(i, t)
            proc._last_ref_q_pre = None
            proc._last_ref_k_pre = None
            proc._last_ref_k = None
            proc._last_ref_v = None
            proc.capture_ref_kv = True

        pipe.transformer(
            hidden_states=noisy_latent,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            concat_img_ids=reference_image_ids,
            return_dict=False,
        )[0]

        for name in names: 
            proc = pipe.transformer.attn_processors[name]
            proc.capture_ref_kv = False
            if proc._last_ref_k is None:
                z = torch.empty(0)
                cache[name].append((z, z, z, z))
            else:
                q_pre = proc._last_ref_q_pre.clone() if proc._last_ref_q_pre is not None else torch.empty(0)
                k_pre = proc._last_ref_k_pre.clone() if proc._last_ref_k_pre is not None else torch.empty(0)
                cache[name].append((q_pre, k_pre, proc._last_ref_k.clone(), proc._last_ref_v.clone()))

    torch.cuda.empty_cache()
    return cache


def apply_kv_cache_for_step(pipe: "FluxPipeline", cache: RefKvCache, step_index: int, device: torch.device) -> None:
    """把 `precompute_style_reference_kv_cache` 在第 `step_index` 步存的张量挂到各层 attention 上。

    Args:
        pipe: 含 `transformer.attn_processors` 的 Flux 管线。
        cache: 每层名 -> 按推理步索引的列表；每步为四元组或旧版二元组（见函数体内）。
        step_index: 当前主生成循环的去噪步下标，与预计算时 `enumerate(timesteps)` 对齐。
        device: 主生成计算设备；缓存多在 CPU，此处 `.to(device)` 再写入 override。
    """
    for name, proc in pipe.transformer.attn_processors.items():
        # name: 该 DiT 层 attention 模块在 transformer 中的注册名；proc: 对应的 ShareAttnFluxAttnProcessor2_0
        if name not in cache or step_index >= len(cache[name]):
            continue
        entry = cache[name][step_index]  # 该层在当前去噪步的一条缓存
        if len(entry) == 2:
            # 兼容旧缓存：(k_rope, v_rope)
            rk, rv = entry
            rq, rk_st = torch.empty(0), torch.empty(0)
        else:
            # rq / rk_st: 参考前向在 norm 之后、AdaIN 之前的图像 token Q / K（供主生成 AdaIN(tar, ref)）
            # rk / rv: 仅图像段在 additional RoPE 之后的 k_ / v_（主生成里覆盖当前 k_/v_ 再与当前路拼接）
            rq, rk_st, rk, rv = entry
        if rk.numel() == 0:
            proc.ref_q_override = None
            proc.ref_k_style_override = None
            proc.ref_k_override = None
            proc.ref_v_override = None
            continue
        if rq.numel() > 0:
            proc.ref_q_override = rq.to(device=device)
        else:
            proc.ref_q_override = None
        if rk_st.numel() > 0:
            proc.ref_k_style_override = rk_st.to(device=device)
        else:
            proc.ref_k_style_override = None
        proc.ref_k_override = rk.to(device=device)
        proc.ref_v_override = rv.to(device=device)
