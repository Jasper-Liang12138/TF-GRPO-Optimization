"""
rewards.py — 数学任务奖励函数

支持 GSM8K（#### 最终答案）和 MATH（\\boxed{} 答案）两种格式。
奖励为二值：正确 → 1.0，错误 → 0.0。
"""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np


# ── 答案提取 ──────────────────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """从 GSM8K 格式输出中提取 #### 后的数字。"""
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # 回退：取最后一个数字
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None


def extract_boxed_answer(text: str) -> Optional[str]:
    """从 MATH 格式输出中提取 \\boxed{...} 内容。"""
    # 支持嵌套花括号的简单版本
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m.group(1).strip() if m else None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def is_correct(pred: str, gold: str) -> bool:
    """先尝试数值比对，再尝试字符串精确匹配。"""
    # 数值比对
    pf, gf = _safe_float(pred), _safe_float(gold)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-6

    # 尝试 sympy 符号化比对（需要安装 sympy）
    try:
        import sympy
        expr_p = sympy.sympify(pred, evaluate=True)
        expr_g = sympy.sympify(gold, evaluate=True)
        return sympy.simplify(expr_p - expr_g) == 0
    except Exception:
        pass

    # 字符串归一化比对
    return pred.strip().lower() == gold.strip().lower()


# ── 主奖励函数 ────────────────────────────────────────────────────────────

def compute_math_reward(
    output_text: str,
    gold_answer: str,
    dataset: str = "gsm8k",
) -> float:
    """
    计算单条输出的奖励。
    返回 1.0（正确）或 0.0（错误）。
    """
    if dataset == "gsm8k":
        pred = extract_gsm8k_answer(output_text)
    else:
        pred = extract_boxed_answer(output_text) or extract_gsm8k_answer(output_text)

    if pred is None:
        return 0.0

    return 1.0 if is_correct(pred, gold_answer) else 0.0


# ── GRPO 优势计算 ─────────────────────────────────────────────────────────

def compute_advantages(rewards: List[float]) -> List[float]:
    """
    GRPO 优势：z-score 归一化。
      A_i = (r_i - mean(r)) / (std(r) + ε)
    当所有奖励相同时返回全零（无信号可学习）。
    """
    r = np.array(rewards, dtype=np.float32)
    mean_r = r.mean()
    std_r = r.std()
    if std_r < 1e-8:
        return [0.0] * len(rewards)
    return ((r - mean_r) / (std_r + 1e-8)).tolist()
