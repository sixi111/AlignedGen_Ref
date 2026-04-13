import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
from dataclasses import dataclass
from diffusers.models.embeddings import apply_rotary_emb


@dataclass(frozen=True)
class StyleAlignedArgs:
    share_attention: bool = True
    block: tuple[int, int] = (19, 57)
    timesteps: tuple[int, int] = (0, 30)
    style_lambda_mode: str = "decrease"  # ["decrease", "fix"]
    style_lambda: float = 1.
    constrain_first: bool = True
    # 用户参考风格图时，batch 中无“生成出来的参考图”，不应屏蔽第 0 行对拼接键的注意
    external_style_reference: bool = False


T = torch.Tensor


def expand_first(feat: T, scale=1., ) -> T:
    bs = feat.shape[0]
    feat_style = feat[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    return feat_style


def concat_first(feat: T, dim=2, scale=1.) -> T:
    bs = feat.shape[0]
    feat_style = feat[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    return torch.cat((feat, feat_style), dim=dim)


def concat_first_block(feat_all: T, feat_block: T, dim=2, scale=1.) -> T:
    bs = feat_all.shape[0]  # [4, 24, 4608, 128]
    if scale == 1.:
        feat_style = feat_block[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    else:
        feat_style = (scale * feat_block[0]).unsqueeze(0).repeat(bs - 1, 1, 1, 1)   # [1, 24, 64, 128]-->[3, 24, 64, 128]
        feat_style = torch.cat((feat_block[0].unsqueeze(0), feat_style), dim=0)
    return torch.cat((feat_all, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


def adain_tar_ref(feat_tar: T, feat_ref: T) -> T:
    """AdaIN(feat_tar, feat_ref)：用 tar 的实例统计做标准化，用 ref 的均值/方差（序列维）做反标准化。"""
    mu_tar, std_tar = calc_mean_std(feat_tar)
    mu_ref, std_ref = calc_mean_std(feat_ref)
    if feat_tar.shape[0] != feat_ref.shape[0]:
        mu_ref = mu_ref.expand(feat_tar.shape[0], -1, -1, -1)
        std_ref = std_ref.expand(feat_tar.shape[0], -1, -1, -1)
    return (feat_tar - mu_tar) / std_tar * std_ref + mu_ref


class ShareAttnFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, cnt: int, style_aligned_args: StyleAlignedArgs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.cnt = cnt
        self.t = 1
        self.args = style_aligned_args
        self.attn_weights = []
        # 论文附录 A：外部参考图预计算的 K/V（已含 ShiftPE 分支上的 RoPE）
        self.ref_k_override: Optional[torch.Tensor] = None
        self.ref_v_override: Optional[torch.Tensor] = None
        # 论文：AdaIN 用 Q_img^ref / K_img^ref（与 tar 同阶段：norm 后、RoPE 与 concat 前）
        self.ref_q_override: Optional[torch.Tensor] = None
        self.ref_k_style_override: Optional[torch.Tensor] = None
        self.capture_ref_kv: bool = False
        self._last_ref_q_pre: Optional[torch.Tensor] = None
        self._last_ref_k_pre: Optional[torch.Tensor] = None
        self._last_ref_k: Optional[torch.Tensor] = None
        self._last_ref_v: Optional[torch.Tensor] = None

    def set_timesteps(self, t, timesteps):
        self.t = t
        if self.args.style_lambda_mode == "decrease":
            self.scale = (timesteps / 1000) * (timesteps / 1000)
        elif self.args.style_lambda_mode == "fix":
            self.scale = self.args.style_lambda

    def set_args(self, style_aligned_args):
        self.args = style_aligned_args

    def clear_ref_override(self) -> None:
        self.ref_k_override = None
        self.ref_v_override = None
        self.ref_q_override = None
        self.ref_k_style_override = None

    def ori_call(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            image_rotary_emb_additional: Optional[torch.Tensor] = None,
            txt_length: int = None,
    ) -> torch.FloatTensor:
        if not self.args.share_attention:
            return self.ori_call(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )
        if not ((self.cnt >= self.args.block[0] and self.cnt < self.args.block[1])
                and (self.t >= self.args.timesteps[0] and self.t < self.args.timesteps[1])):
            return self.ori_call(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is None:
            encoder_hidden_states_q = query[:, :, :txt_length, :]
            encoder_hidden_states_k = key[:, :, :txt_length, :]
            encoder_hidden_states_v = value[:, :, :txt_length, :]
            query = query[:, :, txt_length:, :]
            key = key[:, :, txt_length:, :]
            value = value[:, :, txt_length:, :]

        # ------------------------------------------------------------------
        # 分支 A：参考图预计算（仅写 cache，不做 AdaIN、不做 AAS 第二路拼接）
        # 分支 B：主生成（AdaIN(tar,ref) + 可选 cache 覆盖 k_/v_ + AAS 拼接）
        # ------------------------------------------------------------------
        if self.capture_ref_kv:
            self._last_ref_q_pre = query.detach().cpu()
            self._last_ref_k_pre = key.detach().cpu()
        else:
            if self.ref_q_override is not None:
                rq = self.ref_q_override.to(device=query.device, dtype=query.dtype)
                query = adain_tar_ref(query, rq)
            else:
                query = adain(query)
            if self.ref_k_style_override is not None:
                rk_st = self.ref_k_style_override.to(device=key.device, dtype=key.dtype)
                key = adain_tar_ref(key, rk_st)
            else:
                key = adain(key)

        if encoder_hidden_states is None:
            # Q^F = Concat(Q_{txt}^{tar}, \hat{Q}_{img}^{tar})（单流块：先拆出文本与图像 Q，图像段已上式对齐，再拼回）
            query = torch.cat([encoder_hidden_states_q, query], dim=-2)
            key = torch.cat([encoder_hidden_states_k, key], dim=-2)
            value = torch.cat([encoder_hidden_states_v, value], dim=-2)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Q^F = Concat(Q_{txt}^{tar}, \hat{Q}_{img}^{tar})（双流块：query 仅为图像段经 AdaIN 后的 \hat{Q}_{img}^{tar}）
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if txt_length is None:
            assert encoder_hidden_states is not None
            txt_length = encoder_hidden_states.shape[1]

        # K_img^ref / V_img^ref：与 k_ 同相位（图像段 + additional RoPE）；须每步先算再 capture / override / concat
        k_ = key[:, :, txt_length:, :]      # 参考分支：[1, 24, 4096, 128]
        v_ = value[:, :, txt_length:, :]
        k_ = apply_rotary_emb(k_, image_rotary_emb_additional)

        if self.capture_ref_kv:
            self._last_ref_k = k_.detach().cpu()
            self._last_ref_v = v_.detach().cpu()
        elif self.ref_k_override is not None:
            k_ = self.ref_k_override.to(device=key.device, dtype=key.dtype)   # [1, 24, 64, 128]
            v_ = self.ref_v_override.to(device=value.device, dtype=value.dtype)

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        if not self.capture_ref_kv:
            # K^F = Concat(K_txt, \hat{K}_img^tar, K_img^ref)；V^F = Concat(V_txt, V_img^tar, V_img^ref)（Q^F 不拼 ref）
            key = concat_first_block(key, k_, -2, scale=self.scale) # (batch, heads, seq_len, head_dim)
            value = concat_first_block(value, v_, -2)

        if self.args.constrain_first and not self.args.external_style_reference:
            rows, cols = query.shape[-2], key.shape[-2]
            attn_mask = torch.zeros((query.shape[0], 24, rows, cols), dtype=query.dtype, device=query.device)
            attn_mask[0, :, :, query.shape[-2]:] = -float("inf")
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False,
                                                           attn_mask=attn_mask)
            del attn_mask
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
