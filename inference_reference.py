"""
使用论文附录 A 的流程：对用户提供的参考风格图预计算 QKV，再在 AAS 中注入（external_style_reference）。

示例：
  python inference_reference.py \\
    --model_path black-forest-labs/FLUX.1-dev \\
    --style_reference_image path/to/style.jpg \\
    --style_lambda 1.1
"""

import argparse
import os

import torch
from PIL import Image

from aligngen.aligned_pipeline import FluxPipeline
from aligngen.aligned_transformer import FluxTransformer2DModel
from aligngen.attention_processor import ShareAttnFluxAttnProcessor2_0, StyleAlignedArgs


def init_attention_processors(pipeline: FluxPipeline, style_aligned_args: StyleAlignedArgs | None = None):
    attn_procs = {}
    transformer = pipeline.transformer
    for i, name in enumerate(transformer.attn_processors.keys()):
        attn_procs[name] = ShareAttnFluxAttnProcessor2_0(
            cnt=i,
            style_aligned_args=style_aligned_args,
        )
    transformer.set_attn_processor(attn_procs)


def concat_img(images, k):
    img_width, img_height = images[0].size
    n = len(images)
    rows = (n + k - 1) // k

    total_width = img_width * k
    total_height = img_height * rows
    new_img = Image.new("RGB", (total_width, total_height), "white")

    for i, img in enumerate(images):
        row = i // k
        col = i % k
        new_img.paste(img, (col * img_width, row * img_height))

    return new_img


def main(args):
    # timesteps 第二项为步索引上界（exclusive）：须 ≥ num_inference_steps，否则 i≥30 时走 ori_call，
    # ref_k_override 等全部被忽略，后半程易糊、风格与语义脱节。
    style_args = StyleAlignedArgs(
        share_attention=True,
        block=(19, 57),
        timesteps=(0, max(1, args.num_inference_steps)),
        style_lambda_mode="fix",
        style_lambda=args.style_lambda,
        constrain_first=True,
        external_style_reference=False,
    )

    print(f"Loading model from: {args.model_path}")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    transformer.set_style_aligned_args(style_args)

    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    init_attention_processors(pipe, style_args)
    pipe = pipe.to("cuda")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ref_image = Image.open(args.style_reference_image).convert("RGB")

    print(
        f"Generating (ref KV precompute: {args.reference_kv_precompute_mode})..."
    )
    preview_dir = output_dir if args.reference_kv_precompute_mode == "denoise" else None
    images = pipe(
        args.prompts,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        style_reference_image=ref_image,
        reference_prompt=args.reference_prompt,
        reference_cache_generator=torch.Generator("cpu").manual_seed(args.cache_seed),
        reference_kv_precompute_mode=args.reference_kv_precompute_mode,
        reference_denoise_preview_dir=preview_dir,
    ).images
    torch.cuda.empty_cache()

    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f"{i}.jpg"))

    preview_path = os.path.join(output_dir, "denoise_ref_preview.jpg")
    concat_list = list(images)
    if os.path.isfile(preview_path):
        concat_list.insert(0, Image.open(preview_path).convert("RGB"))
    concat_result = concat_img(concat_list, len(concat_list))
    concat_result.save(os.path.join(output_dir, "concat.jpg"))
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlignedGen with user style reference image (paper Appendix A)")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--style_lambda", type=float, default=1.1)
    parser.add_argument(
        "--style_reference_image",
        type=str,
        required=True,
        help="Path to a style reference image (RGB).",
    )
    parser.add_argument(
        "--reference_prompt",
        type=str,
        default=" ",
        help="Caption used only when encoding the reference branch during QKV cache (default: minimal).",
    )
    parser.add_argument("--output_dir", type=str, default="output_ref")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache_seed",
        type=int,
        default=42,
        help="附录 A：插值噪声种子；denoise 模式：初始 packed latent 的高斯噪声种子。",
    )
    parser.add_argument(
        "--reference_kv_precompute_mode",
        type=str,
        choices=["appendix_a", "denoise"],
        default="appendix_a",
        help="参考 KV 预计算：appendix_a=论文插值 latent；denoise=纯噪声+scheduler 逐步去噪并每步记录 KV。",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Pass local_files_only=True to from_pretrained (no Hub download).",
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        help="Target prompts (optional; default: four demo prompts).",
    )
    parsed = parser.parse_args()
    if not parsed.prompts:
        parsed.prompts = [
            # "Dog in 3D realism style.",
            "Clock in 3D realism style.",
            "Globe in 3D realism style.",
            "Bicycle in 3D realism style.",
        ]
    main(parsed)
