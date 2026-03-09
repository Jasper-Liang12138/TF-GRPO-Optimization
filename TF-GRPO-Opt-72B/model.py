"""
model.py — ParameterizedExperienceModel（72B 多卡版）

整合四个组件：
  Encoder（冻结）→ MemoryBank（显式 CRUD）→ ExperienceAggregator（唯一可训练）
  → 冻结 Qwen2.5-Math-72B-Instruct

与 7B 版的主要区别：
  LLM 使用 device_map="auto" 分片到 8 张 NPU；
  Encoder / MemoryBank / Aggregator 保留在 npu:0；
  所有需要与 LLM 交互的 tensor 送往 self._llm_device（LLM 首层所在卡）。

梯度路径：
  loss → logprob → inputs_embeds（prefix 部分）→ prefix P → Aggregator 权重
  LLM 参数本身不更新，但其 forward 运算充当梯度传导通道。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from config import Config
from memory import MemoryBank
from aggregator import ExperienceAggregator


class ParameterizedExperienceModel(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float16 if config.torch_dtype == "float16" else torch.bfloat16

        # ── 1. Encoder（冻结） ─────────────────────────────────────────────
        print(f"[Model] 加载 Encoder: {config.encoder_name}")
        self.encoder = SentenceTransformer(
            config.encoder_name,
            device=str(self.device),
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ── 2. 经验向量库 M（设备在 LLM 加载后确定，见下方）────────────────
        self.memory = None  # 占位，LLM 加载后重新初始化到 _llm_device

        # ── 3. Experience Aggregator（可训练，设备在 LLM 加载后确定） ────────
        # 先暂放 CPU，待 LLM device_map 确定首层设备后再迁移
        self.aggregator = ExperienceAggregator(
            L=config.L,
            k=config.k,
            d_enc=config.d_enc,
            d_llm=config.d_llm,
            H=config.H,
        ).to(self.dtype)

        # ── 4. LLM（冻结，多卡分片） ──────────────────────────────────────
        print(f"[Model] 加载 LLM: {config.llm_name}  (dtype={self.dtype}, device_map=auto)")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name,
            trust_remote_code=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto",   # 自动分片到所有可见 NPU / GPU
        )

        # 冻结 LLM
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # LLM 首层所在设备（embed_tokens 和 inputs 需送到此卡）
        self._llm_device = next(self.llm.parameters()).device
        print(f"[Model] LLM 首层设备: {self._llm_device}")

        # Aggregator / MemoryBank 迁移到与 LLM 首层同一设备
        self.aggregator = self.aggregator.to(self._llm_device)
        self.memory = MemoryBank(N=config.N, d=config.d_enc, device=self._llm_device)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] 可训练参数: {n_trainable:,}  (~{n_trainable/1e6:.2f}M)")

    # ── Encoder 工具 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """文字 → 768 维归一化向量（送往 LLM 首层设备）。"""
        vec = self.encoder.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return vec.to(self._llm_device)

    # ── Soft Prefix 构造 ──────────────────────────────────────────────────

    def build_prefix(
        self,
        question_text: str,
        detach: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构造 soft prefix P。

        返回:
            P         [1, L, d_llm]   — 若 detach=True 则无梯度
            top_k_idx [k]             — 本次检索使用的槽编号
        """
        with torch.no_grad():
            q_vec = self.encode_text(question_text)                       # [768]
            top_k_vecs, top_k_idx = self.memory.query(q_vec, self.config.k)
            # 转为 LLM dtype，再升维为 [1, k, 768]
            top_k_vecs = top_k_vecs.to(self.dtype).unsqueeze(0)

        P = self.aggregator(top_k_vecs)   # [1, L, d_llm]，梯度在此产生

        if detach:
            P = P.detach()

        return P, top_k_idx

    # ── Rollout 生成（无梯度） ────────────────────────────────────────────

    @torch.no_grad()
    def generate_rollouts(
        self,
        question_text: str,
        prefix: torch.Tensor,   # [1, L, d_llm]，已 detach
        G: int,
    ) -> List[str]:
        """
        使用 soft prefix 对问题采样 G 条输出。
        inputs_embeds = [P ; question_embeds]
        使用 Qwen2.5-Math chat 格式。
        """
        enc = self.tokenizer(
            question_text,
            return_tensors="pt",
            padding=True,
        ).to(self._llm_device)

        q_ids = enc.input_ids           # [1, T_q]
        q_mask = enc.attention_mask     # [1, T_q]

        q_embeds = self.llm.model.embed_tokens(q_ids).to(self.dtype)  # [1, T_q, d_llm]

        # 拼接 prefix
        inputs_embeds = torch.cat([prefix, q_embeds], dim=1)   # [1, L+T_q, d_llm]

        # 构造 attention mask
        B = inputs_embeds.shape[0]
        prefix_mask = torch.ones(B, self.config.L, device=self._llm_device, dtype=q_mask.dtype)
        attention_mask = torch.cat([prefix_mask, q_mask], dim=1)

        # 多条采样：repeat
        if G > 1:
            inputs_embeds = inputs_embeds.repeat(G, 1, 1)
            attention_mask = attention_mask.repeat(G, 1)

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]
        return texts

    # ── 对数概率打分（保留梯度） ──────────────────────────────────────────

    def compute_logprob(
        self,
        question_text: str,
        output_text: str,
        prefix: torch.Tensor,   # [1, L, d_llm]，含梯度
    ) -> torch.Tensor:
        """
        计算 log P(output | [P ; question]) 的总和（非平均）。

        前向图：
          logprob → inputs_embeds（prefix 部分）→ P → Aggregator 权重
          LLM 参数本身不更新，仅作为运算节点传导梯度至 P。
        """
        q_enc = self.tokenizer(question_text, return_tensors="pt").to(self._llm_device)
        o_enc = self.tokenizer(output_text, return_tensors="pt").to(self._llm_device)

        q_ids = q_enc.input_ids    # [1, T_q]
        o_ids = o_enc.input_ids    # [1, T_o]
        T_q = q_ids.shape[1]
        T_o = o_ids.shape[1]
        L = self.config.L

        # 嵌入（无需梯度）
        with torch.no_grad():
            q_embeds = self.llm.model.embed_tokens(q_ids).to(self.dtype)   # [1, T_q, d_llm]
            o_embeds = self.llm.model.embed_tokens(o_ids).to(self.dtype)   # [1, T_o, d_llm]

        # 完整输入序列：[P, question, output]
        # 因果 LM 在位置 t 预测 token t+1
        # labels 在 prefix+question 位置设为 -100（忽略），在 output 位置设为 o_ids
        inputs_embeds = torch.cat([prefix, q_embeds, o_embeds], dim=1)   # [1, L+T_q+T_o, d_llm]

        labels = torch.full(
            (1, L + T_q + T_o),
            fill_value=-100,
            dtype=torch.long,
            device=self._llm_device,
        )
        labels[:, L + T_q:] = o_ids

        attention_mask = torch.ones(
            1, L + T_q + T_o,
            dtype=torch.long,
            device=self._llm_device,
        )

        # LLM forward（LLM 参数无 grad，但运算传导梯度至 prefix）
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        # out.loss = -1/T_o * Σ log P(o_t | ...)
        # 我们要总和对数概率（非平均）
        total_logprob = -out.loss * T_o
        return total_logprob

    # ── 检查点 ────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "aggregator": self.aggregator.state_dict(),
                "memory": self.memory.state_dict(),
                "config": self.config,
            },
            path,
        )
        print(f"[Checkpoint] 保存 → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.aggregator.load_state_dict(ckpt["aggregator"])
        self.memory.load_state_dict(ckpt["memory"])
        print(f"[Checkpoint] 加载 ← {path}")
