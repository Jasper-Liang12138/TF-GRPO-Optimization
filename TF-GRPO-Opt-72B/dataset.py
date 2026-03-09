"""
dataset.py — 数学数据集加载与 prompt 构造

支持:
  gsm8k  — openai/gsm8k
  math   — hendrycks/competition_math
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from torch.utils.data import Dataset


class MathDataset(Dataset):
    """
    每条样本格式：
      {
          "question":     str,   # 原始问题文字
          "answer":       str,   # 用于奖励计算的最终答案
          "full_answer":  str,   # 完整参考解答（可用于经验初始化）
      }
    """

    def __init__(
        self,
        dataset_name: str = "gsm8k",
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        from datasets import load_dataset

        self.dataset_name = dataset_name
        self.data: List[Dict] = []

        if dataset_name == "gsm8k":
            raw = load_dataset("openai/gsm8k", "main", split=split)
            for item in raw:
                self.data.append(
                    {
                        "question": item["question"],
                        "answer": self._gsm8k_final(item["answer"]),
                        "full_answer": item["answer"],
                    }
                )
        elif dataset_name == "math":
            raw = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)
            for item in raw:
                self.data.append(
                    {
                        "question": item["problem"],
                        "answer": item["solution"],   # 含 \boxed{}
                        "full_answer": item["solution"],
                    }
                )
        else:
            raise ValueError(f"未知数据集: {dataset_name}，支持 gsm8k / math")

        if max_samples is not None:
            self.data = self.data[:max_samples]

    # ── 工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _gsm8k_final(answer_text: str) -> str:
        m = re.search(r"####\s*([\-\d,\.]+)", answer_text)
        if m:
            return m.group(1).replace(",", "").strip()
        return answer_text.strip()

    # ── torch Dataset 接口 ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

    # ── Prompt 构造 ───────────────────────────────────────────────────────

    def build_prompt(self, question: str) -> str:
        """
        使用 Qwen2.5-Math-Instruct 的 chat 模板格式。
        仅包含 system + user 部分，assistant 由模型补全。
        """
        if self.dataset_name == "gsm8k":
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
