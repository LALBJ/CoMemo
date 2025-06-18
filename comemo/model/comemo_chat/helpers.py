"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import math
from typing import Optional, Tuple, Union
from comemo.model.internlm2.modeling_internlm2 import InternLM2RMSNorm, InternLM2RotaryEmbedding
from .configuration_mixin import MixinConfig
import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn

from transformers.activations import ACT2FN

from flash_attn.flash_attn_interface import flash_attn_varlen_func

# Copied from transformers.model.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim).float()
    sin = sin[position_ids].unsqueeze(unsqueeze_dim).float()
    q_dtype = q.dtype
    q = q.float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed.to(dtype=q_dtype)

class CrossAttention(nn.Module):
    def __init__(
        self,
        config: MixinConfig
    ):
        super().__init__()
        dim = config.language_dim
        dim_visual = config.vision_dim
        dim_head = config.head_dim
        heads = config.num_heads

        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.head_dim = dim_head
        self.max_position_embeddings = 32768

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self._init_rope()

        self.text_position_ids = None

        self.cu_seqlens_q = None
        self.cu_seqlens_k = None

    def _init_rope(self):
        self.rotary_emb = InternLM2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=1000000,
        )
        return self.rotary_emb

    def forward(self, x, media, use_cached_media=False, media_position_ids=None, text_position_ids=None, text_time=None):
        h = self.heads

        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        if use_cached_media and self.text_position_ids is not None:
            text_position_ids = self.text_position_ids[:, -1].unsqueeze(0)
            t_cos, t_sin = self.rotary_emb(v, seq_len=(text_position_ids.max().item()+1))
            q = apply_rotary_pos_emb_single(q, t_cos, t_sin, text_position_ids)
        else:
            t_cos, t_sin = self.rotary_emb(v, seq_len=(text_position_ids.max().item()+1))
            q = apply_rotary_pos_emb_single(q, t_cos, t_sin, text_position_ids)
        
        ## To support the update of position_ids in RoPE-DHR.
        if use_cached_media:
            if self.text_position_ids is None:
                self.text_position_ids = text_position_ids
            next_position_ids = torch.tensor([[self.text_position_ids.shape[1]]], device=self.text_position_ids.device, dtype=self.text_position_ids.dtype)
            self.text_position_ids = torch.cat((self.text_position_ids, next_position_ids), dim=1)
        
        m_cos, m_sin = self.rotary_emb(v, seq_len=(media_position_ids.max().item()+1))
        k = apply_rotary_pos_emb_single(k, m_cos, m_sin, media_position_ids)

        if self.cu_seqlens_k is not None and self.cu_seqlens_q is not None:
            # Use flash-attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = self._flash_attention_forward(q, k, v, self.cu_seqlens_q, self.cu_seqlens_k.to(torch.int32))
            attn_output = attn_output.unsqueeze(0).transpose(1, 2)
        else:
            # Use torch.sdpa
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, self.media_attn_mask)
        
        if text_time is not None:
            text_without_media_mask = text_time == 1
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn_output = attn_output.masked_fill(text_without_media_mask, 0.0)

        out = rearrange(attn_output, "b h n d -> b n (h d)")
        return self.to_out(out)

    def _flash_attention_forward(
            self, query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                rename from cu_seqlens to keep compatability - (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                    of the sequences in the batch.
            cu_seqlens_q (`torch.Tensor`):
                The length of each sequence in the query.
                To support data packing based cross-attention computation.
            cu_seqlens_k (`torch.Tensor`):
                The length of each sequence in the keys.
                To support data packing based cross-attention computation.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        cu_seqlens_q = cu_seqlens_q.squeeze(0)
        cu_seqlens_k = cu_seqlens_k.squeeze(0)

        with torch.no_grad():
            max_seqlen_q = max([
                cu_seqlens_q[idx+1] - cu_seqlens_q[idx]
                for idx in range(cu_seqlens_q.size(0) - 1)
            ]).item()

            max_seqlen_k = max([
                cu_seqlens_k[idx+1] - cu_seqlens_k[idx]
                for idx in range(cu_seqlens_k.size(0) - 1)
            ]).item()
        
        # Contains at least one padding token in the sequence
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=False,
        )

        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)
        return attn_output

class InternLM2MLP(nn.Module):
    def __init__(self, config, hidden_act='silu'):
        super().__init__()
        self.hidden_size = config.language_dim
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        return down_proj

class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        config: MixinConfig
    ):
        super().__init__()
        dim = config.language_dim
        intermediate_size = config.intermediate_size

        self.cross_attention_norm = InternLM2RMSNorm(dim, eps=1e-5)
        self.ffn_norm_2 = InternLM2RMSNorm(dim, eps=1e-5)

        self.cross_attn = CrossAttention(
            config=config
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ffn_2 = InternLM2MLP(config)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

        self.media = None

    def forward(
        self,
        x,
        media,
        use_cached_media=False,
    ):  
        residual = x
        x = self.cross_attention_norm(x)
        media = self.cross_attention_norm(media) 
        x = (
            self.cross_attn(
                x,
                media,
                use_cached_media=use_cached_media,
                media_position_ids=self.cross_attn_media_position_ids,
                text_position_ids=self.cross_attn_text_position_ids
            )
            * self.attn_gate.tanh()
            + residual
        )

        residual = x
        x = self.ffn_norm_2(x)
        x = self.ffn_2(x) * self.ff_gate.tanh() + residual

        return x
    