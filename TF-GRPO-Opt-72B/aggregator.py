"""
aggregator.py — Experience Aggregator（唯一可训练模块）

Q-Former 风格：
  输入  top_k_vecs  [B, k, d_enc=768]  — MemoryBank 检索结果，已 detach
  输出  soft_prefix [B, L, d_llm=3584] — 直接拼接在 LLM 输入 embedding 之前

参数量（默认超参）：
  Learnable Query Tokens (L=10, d=768)   ~7,680
  Cross-Attention (W_Q/K/V/O, d=768)    ~2,359,296
  Linear(768 → 3584)                    ~2,752,512
  LayerNorm × 2                         ~3,072
  合计                                  ~5,122,560  ≈ 511 万
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ExperienceAggregator(nn.Module):

    def __init__(
        self,
        L: int = 10,
        k: int = 5,
        d_enc: int = 768,
        d_llm: int = 3584,
        H: int = 8,
    ) -> None:
        super().__init__()
        assert d_enc % H == 0, f"d_enc({d_enc}) 必须能被 H({H}) 整除"

        self.L = L
        self.k = k
        self.d_enc = d_enc
        self.d_llm = d_llm
        self.H = H
        self.d_head = d_enc // H

        # ── 可训练 Query Tokens ──────────────────────────────────────────────
        # 初始化为小量正态，避免与冻结 Encoder 向量尺度差距过大
        self.query_tokens = nn.Parameter(torch.randn(L, d_enc) * 0.02)

        # ── Cross-Attention 投影 ─────────────────────────────────────────────
        self.W_Q = nn.Linear(d_enc, d_enc, bias=False)
        self.W_K = nn.Linear(d_enc, d_enc, bias=False)
        self.W_V = nn.Linear(d_enc, d_enc, bias=False)
        self.W_O = nn.Linear(d_enc, d_enc, bias=False)

        # ── Layer Norm ───────────────────────────────────────────────────────
        self.ln_q = nn.LayerNorm(d_enc)
        self.ln_out = nn.LayerNorm(d_enc)

        # ── 投影到 LLM 维度 ──────────────────────────────────────────────────
        self.proj = nn.Linear(d_enc, d_llm)

        self._init_weights()

    def _init_weights(self) -> None:
        for lin in [self.W_Q, self.W_K, self.W_V, self.W_O, self.proj]:
            nn.init.xavier_uniform_(lin.weight)

    # ── 前向传播 ──────────────────────────────────────────────────────────

    def forward(self, top_k_vecs: torch.Tensor) -> torch.Tensor:
        """
        top_k_vecs: [B, k, d_enc]  已 detach，不含梯度
        返回:       [B, L, d_llm]  soft prefix，梯度在此处生成
        """
        B = top_k_vecs.shape[0]

        # Batch 化 Learnable Queries
        Q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, L, d_enc]
        Q = self.ln_q(Q)

        # 投影
        Q_proj = self.W_Q(Q)              # [B, L, d_enc]
        K_proj = self.W_K(top_k_vecs)    # [B, k, d_enc]
        V_proj = self.W_V(top_k_vecs)    # [B, k, d_enc]

        # 多头拆分  →  [B, H, seq, d_head]
        Q_h = self._split_heads(Q_proj)   # [B, H, L, d_head]
        K_h = self._split_heads(K_proj)   # [B, H, k, d_head]
        V_h = self._split_heads(V_proj)   # [B, H, k, d_head]

        # Scaled Dot-Product Attention
        scale = math.sqrt(self.d_head)
        attn_w = (Q_h @ K_h.transpose(-2, -1)) / scale  # [B, H, L, k]
        attn_w = torch.softmax(attn_w, dim=-1)

        out = attn_w @ V_h                               # [B, H, L, d_head]
        out = self._merge_heads(out)                     # [B, L, d_enc]
        out = self.W_O(out)

        # 残差 + LayerNorm
        Q_res = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        out = self.ln_out(out + Q_res)                   # [B, L, d_enc]

        # 投影到 LLM 维度
        prefix = self.proj(out)                          # [B, L, d_llm]
        return prefix

    # ── 工具函数 ──────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, d_enc] → [B, H, S, d_head]"""
        B, S, _ = x.shape
        return x.view(B, S, self.H, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, S, d_head] → [B, S, d_enc]"""
        B, H, S, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * dh)
