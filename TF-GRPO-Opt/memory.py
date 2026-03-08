"""
memory.py — 经验向量库 M

N 个槽，每槽 768 维；不参与梯度计算，由优势分数驱动显式 CRUD。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class MemoryBank:
    """
    向量库 M：N × d_enc 的显式存储结构。

    接口：
        query(q, k)    → top-k 经验向量（detached）
        add(vec, score)→ 写入空槽或替换最低分槽
        delete(idx)    → 清空指定槽
        prune(thresh)  → 批量删除低分槽
        merge(thresh)  → 合并高相似槽
    """

    def __init__(self, N: int, d: int, device: torch.device) -> None:
        self.N = N
        self.d = d
        self.device = device

        # 槽向量 [N, d]
        self.slots: torch.Tensor = torch.zeros(N, d, device=device)
        # 是否被占用
        self.occupied: torch.Tensor = torch.zeros(N, dtype=torch.bool, device=device)
        # 累计优势分
        self.scores: torch.Tensor = torch.zeros(N, device=device)
        # 文字元数据（调试用）
        self.texts: List[Optional[str]] = [None] * N

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def num_occupied(self) -> int:
        return int(self.occupied.sum().item())

    # ── 检索 ──────────────────────────────────────────────────────────────

    def query(self, q: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        余弦相似度 top-k 检索。

        参数:
            q: [d]  查询向量（已归一化亦可）
        返回:
            top_k_vecs [k, d]  — 已 detach，不参与反向传播
            top_k_idx  [k]     — 全局槽编号
        """
        if self.num_occupied == 0:
            # 冷启动：返回零向量
            zero_vecs = torch.zeros(k, self.d, device=self.device)
            zero_idx = torch.zeros(k, dtype=torch.long, device=self.device)
            return zero_vecs, zero_idx

        occ_idx = torch.where(self.occupied)[0]          # [M,]
        occ_slots = self.slots[occ_idx]                  # [M, d]

        q_norm = F.normalize(q.unsqueeze(0), dim=-1)     # [1, d]
        s_norm = F.normalize(occ_slots, dim=-1)          # [M, d]
        sim = (q_norm @ s_norm.T).squeeze(0)             # [M,]

        actual_k = min(k, len(occ_idx))
        _, topk_local = sim.topk(actual_k)
        topk_global = occ_idx[topk_local]                # [actual_k,]

        top_k_vecs = self.slots[topk_global]             # [actual_k, d]

        # 不足 k 条时用零填充
        if actual_k < k:
            pad_v = torch.zeros(k - actual_k, self.d, device=self.device)
            pad_i = torch.zeros(k - actual_k, dtype=torch.long, device=self.device)
            top_k_vecs = torch.cat([top_k_vecs, pad_v], dim=0)
            topk_global = torch.cat([topk_global, pad_i], dim=0)

        return top_k_vecs.detach(), topk_global

    # ── 写入 ──────────────────────────────────────────────────────────────

    def add(
        self,
        vec: torch.Tensor,
        score: float,
        text: Optional[str] = None,
    ) -> int:
        """写入第一个空槽；若全满则替换最低分槽。返回槽编号。"""
        vec = vec.detach()
        empty = torch.where(~self.occupied)[0]
        slot_idx = int(empty[0].item()) if len(empty) > 0 else int(self.scores.argmin().item())

        self.slots[slot_idx] = vec
        self.occupied[slot_idx] = True
        self.scores[slot_idx] = score
        self.texts[slot_idx] = text
        return slot_idx

    # ── 删除 ──────────────────────────────────────────────────────────────

    def delete(self, slot_idx: int) -> None:
        """清空指定槽。"""
        self.slots[slot_idx] = 0.0
        self.occupied[slot_idx] = False
        self.scores[slot_idx] = 0.0
        self.texts[slot_idx] = None

    def update_score(self, slot_idx: int, delta: float, decay: float = 0.99) -> None:
        """指数衰减更新槽的累计分。"""
        self.scores[slot_idx] = self.scores[slot_idx] * decay + delta

    # ── 批量维护 ──────────────────────────────────────────────────────────

    def prune(self, delete_threshold: float) -> int:
        """删除累计分低于阈值的槽，返回删除数量。"""
        if self.num_occupied == 0:
            return 0
        occ_idx = torch.where(self.occupied)[0]
        to_del = occ_idx[self.scores[occ_idx] < delete_threshold]
        for idx in to_del:
            self.delete(int(idx.item()))
        return len(to_del)

    def merge(self, merge_threshold: float) -> int:
        """
        合并余弦相似度超过阈值的槽（加权平均），返回合并次数。
        合并后被合并方槽清空。
        """
        if self.num_occupied < 2:
            return 0

        merges = 0
        occ_idx = torch.where(self.occupied)[0].tolist()
        occ_slots = self.slots[occ_idx]                         # [M, d]
        norm_slots = F.normalize(occ_slots, dim=-1)             # [M, d]
        sim_matrix = norm_slots @ norm_slots.T                  # [M, M]

        merged: set = set()
        for i in range(len(occ_idx)):
            if occ_idx[i] in merged:
                continue
            for j in range(i + 1, len(occ_idx)):
                if occ_idx[j] in merged:
                    continue
                if sim_matrix[i, j].item() > merge_threshold:
                    si = max(float(self.scores[occ_idx[i]].item()), 1e-3)
                    sj = max(float(self.scores[occ_idx[j]].item()), 1e-3)
                    w = si / (si + sj)
                    merged_vec = w * self.slots[occ_idx[i]] + (1 - w) * self.slots[occ_idx[j]]
                    self.slots[occ_idx[i]] = merged_vec
                    self.scores[occ_idx[i]] = (si + sj) / 2.0
                    self.delete(occ_idx[j])
                    merged.add(occ_idx[j])
                    merges += 1
        return merges

    # ── 持久化 ────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "slots": self.slots.cpu(),
            "occupied": self.occupied.cpu(),
            "scores": self.scores.cpu(),
            "texts": self.texts,
        }

    def load_state_dict(self, state: dict) -> None:
        self.slots = state["slots"].to(self.device)
        self.occupied = state["occupied"].to(self.device)
        self.scores = state["scores"].to(self.device)
        self.texts = state["texts"]
