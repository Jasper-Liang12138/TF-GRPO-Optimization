"""
build_experience.py — 构建经验库入口

对应原版 build_experience_deepseek.py，将 API 参数替换为本地模型路径。

用法示例（昇腾单卡）：

    # 使用 parquet 数据文件（与原版格式兼容）
    python build_experience.py \
        --data_path /path/to/dapo-math-17k.parquet \
        --sample_size 100 \
        --group_size 4 \
        --epochs 3 \
        --output_dir ./output_logs

    # 使用 HuggingFace GSM8K（不传 data_path 时自动下载）
    python build_experience.py \
        --sample_size 100 \
        --group_size 4 \
        --epochs 3
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

# ── 昇腾 NPU ──────────────────────────────────────────────────────────────
try:
    import torch_npu                               # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    print("[NPU] torch_npu 加载成功，使用昇腾 NPU")
except ImportError:
    print("[NPU] 未找到 torch_npu，回退至 CUDA / CPU")

from tf_grpo import TF_GRPO


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.npu.manual_seed_all(seed)
    except AttributeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-GRPO Baseline — 构建经验库（本地 Qwen2.5-Math-7B-Instruct）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="本地模型路径或 HuggingFace Hub 名称",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Parquet 数据文件路径；不传则自动加载 HuggingFace GSM8K",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="经验精炼轮数",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="每题 rollout 采样数 G",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="从数据集随机采样的题目数量",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_logs",
        help="经验库 JSON 保存目录",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="模型生成最大 token 数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="npu:0",
        help="推理设备，例如 npu:0 / cuda:0 / cpu",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "=" * 50)
    print("  TF-GRPO Baseline — 构建经验库")
    print("=" * 50)
    print(f"  模型:       {args.model}")
    print(f"  设备:       {args.device}")
    print(f"  数据:       {args.data_path or 'HuggingFace GSM8K'}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Group Size: {args.group_size}")
    print(f"  Sample:     {args.sample_size}")
    print("=" * 50 + "\n")

    agent = TF_GRPO(
        model_name=args.model,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    print(">>> Starting TF-GRPO <<<")
    agent.train_loop(
        parquet_path=args.data_path,
        epochs=args.epochs,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
