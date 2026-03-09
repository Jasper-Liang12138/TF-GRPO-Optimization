"""
trainer.py — GRPO 训练器

严格按照 Parameterized-Experience-TF-GRPO.md §6 训练流程实现：

  Step 1  检索（无梯度）
  Step 2  构造 soft prefix（开始保留计算图）
  Step 3  Rollout 采样（无梯度）
  Step 4  计算优势
  Step 5  重新打分（保留计算图）
  Step 6  backward + 只更新 ExperienceAggregator

同时负责 MemoryBank 的 CRUD（§8）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
import torch.optim as optim
from tqdm import tqdm

from config import Config
from model import ParameterizedExperienceModel
from rewards import compute_math_reward, compute_advantages


class GRPOTrainer:

    def __init__(
        self,
        model: ParameterizedExperienceModel,
        config: Config,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # 只优化 Aggregator
        self.optimizer = optim.AdamW(
            model.aggregator.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Warmup scheduler（线性 warmup → 常数）
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )

        self.global_step: int = 0
        self.log_history: List[Dict] = []

    # ── 单步训练 ──────────────────────────────────────────────────────────

    def train_step(self, item: Dict) -> Dict:
        """
        对单道题执行一次完整 GRPO 迭代。
        返回本步指标 dict。
        """
        cfg = self.config
        question_raw = item["question"]
        gold_answer = item["answer"]

        # 用 chat 模板包装
        question_prompt = self._format_prompt(question_raw)

        # ── Step 1+2：检索 + 构造 soft prefix（保留计算图） ────────────────
        # 注意：build_prefix 内部对 encode / query 使用 no_grad，
        # 只有 aggregator(top_k_vecs) 产生梯度。
        P_for_rollout, _ = self.model.build_prefix(question_raw, detach=True)

        # ── Step 3：Rollout 采样（无梯度） ───────────────────────────────
        generated_outputs = self.model.generate_rollouts(
            question_prompt,
            prefix=P_for_rollout,
            G=cfg.G,
        )

        # ── Step 4：计算奖励与优势 ────────────────────────────────────────
        rewards = [
            compute_math_reward(out, gold_answer, cfg.dataset_name)
            for out in generated_outputs
        ]
        advantages = compute_advantages(rewards)

        # ── Step 5+6：重新打分并反向传播 ─────────────────────────────────
        self.optimizer.zero_grad()

        # 重新构造 prefix（新的计算图，含梯度）
        P_grad, _ = self.model.build_prefix(question_raw, detach=False)

        loss_items: List[torch.Tensor] = []
        for out_text, adv in zip(generated_outputs, advantages):
            if not out_text.strip():
                continue
            adv_t = torch.tensor(adv, dtype=self.model.dtype, device=self.device)
            logp = self.model.compute_logprob(
                question_text=question_prompt,
                output_text=out_text,
                prefix=P_grad,
            )
            loss_items.append(-adv_t * logp)

        if loss_items:
            loss = torch.stack(loss_items).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.aggregator.parameters(),
                cfg.grad_clip,
            )
            self.optimizer.step()
            self.scheduler.step()
            loss_val = loss.item()
        else:
            loss_val = 0.0

        # ── Memory CRUD ───────────────────────────────────────────────────
        self._update_memory(question_raw, generated_outputs, advantages)

        self.global_step += 1

        return {
            "step": self.global_step,
            "loss": loss_val,
            "mean_reward": sum(rewards) / max(len(rewards), 1),
            "max_reward": max(rewards) if rewards else 0.0,
            "acc": float(sum(r > 0.5 for r in rewards)) / max(len(rewards), 1),
            "memory_size": self.model.memory.num_occupied,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # ── 记忆库更新（§8 CRUD） ─────────────────────────────────────────────

    def _update_memory(
        self,
        question_raw: str,
        outputs: List[str],
        advantages: List[float],
    ) -> None:
        cfg = self.config

        # Add：优势高的推理过程写入 M
        for out, adv in zip(outputs, advantages):
            if adv > cfg.mem_add_adv_threshold and out.strip():
                exp_text = f"Q: {question_raw}\nA: {out}"
                vec = self.model.encode_text(exp_text)
                # 更新槽分（若相似槽已存在则 add 内部会覆盖最低分）
                self.model.memory.add(vec, score=adv, text=exp_text[:300])

        # Prune + Merge（每 mem_crud_every 步执行一次）
        if self.global_step % cfg.mem_crud_every == 0:
            n_del = self.model.memory.prune(cfg.mem_delete_score_threshold)
            n_mrg = self.model.memory.merge(cfg.mem_merge_cos_threshold)
            if n_del > 0 or n_mrg > 0:
                occ = self.model.memory.num_occupied
                print(
                    f"\n[Memory CRUD] step={self.global_step}  "
                    f"删除={n_del}  合并={n_mrg}  已占用={occ}/{cfg.N}"
                )

    # ── 完整训练循环 ──────────────────────────────────────────────────────

    def train(self, dataset) -> None:
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        for epoch in range(cfg.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} / {cfg.num_epochs}")
            print(f"{'='*60}")

            pbar = tqdm(dataset, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
            for item in pbar:
                metrics = self.train_step(item)
                self.log_history.append(metrics)

                if self.global_step % cfg.log_every == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{metrics['loss']:.4f}",
                            "acc": f"{metrics['acc']:.2%}",
                            "r̄": f"{metrics['mean_reward']:.3f}",
                            "mem": metrics["memory_size"],
                        }
                    )

                if self.global_step % cfg.save_every == 0:
                    ckpt = os.path.join(
                        cfg.output_dir, f"ckpt_step{self.global_step}.pt"
                    )
                    self.model.save_checkpoint(ckpt)
                    self._dump_logs()

        # 最终保存
        self.model.save_checkpoint(os.path.join(cfg.output_dir, "final.pt"))
        self._dump_logs()
        print("\n[Trainer] 训练完成！")

    def _dump_logs(self) -> None:
        path = os.path.join(self.config.output_dir, "train_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.log_history, f, indent=2, ensure_ascii=False)

    # ── 工具 ──────────────────────────────────────────────────────────────

    def _format_prompt(self, question: str) -> str:
        """Qwen2.5-Math-Instruct chat 格式。"""
        if self.config.dataset_name == "gsm8k":
            system = (
                "Please reason step by step, and put your final answer "
                "within #### at the end."
            )
        else:
            system = (
                "Please reason step by step, and put your final answer "
                r"within \boxed{}."
            )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
