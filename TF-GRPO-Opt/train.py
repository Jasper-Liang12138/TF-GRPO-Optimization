"""
train.py — 主入口

用法示例（昇腾单卡）：
    python train.py \
        --device npu:0 \
        --dataset_name gsm8k \
        --G 8 \
        --L 10 \
        --k 5 \
        --N 64 \
        --lr 1e-4 \
        --num_epochs 3 \
        --output_dir ./checkpoints

恢复训练：
    python train.py --resume_checkpoint ./checkpoints/ckpt_step100.pt
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

# ── 昇腾 NPU 初始化 ───────────────────────────────────────────────────────
try:
    import torch_npu                            # noqa: F401 — 注册 npu 后端
    from torch_npu.contrib import transfer_to_npu  # noqa: F401 — 算子迁移补丁
    print("[NPU] torch_npu 加载成功，使用昇腾 NPU")
except ImportError:
    print("[NPU] 未找到 torch_npu，回退至 CUDA / CPU")

from config import Config
from dataset import MathDataset
from model import ParameterizedExperienceModel
from trainer import GRPOTrainer


# ── 随机种子 ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.npu.manual_seed_all(seed)
    except AttributeError:
        pass
    try:
        torch.cuda.manual_seed_all(seed)
    except AttributeError:
        pass


# ── CLI 参数解析 ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameterized Experience TF-GRPO — Ascend NPU 版本"
    )

    # 模型
    parser.add_argument("--llm_name", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16"])

    # 架构
    parser.add_argument("--N", type=int, default=64,  help="记忆槽数量")
    parser.add_argument("--L", type=int, default=10,  help="soft prefix 长度")
    parser.add_argument("--k", type=int, default=5,   help="top-k 检索")
    parser.add_argument("--H", type=int, default=8,   help="Cross-Attention 头数")

    # GRPO
    parser.add_argument("--G", type=int, default=8,   help="每题 rollout 采样数")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)

    # 训练
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--output_dir", default="./checkpoints")

    # 数据
    parser.add_argument("--dataset_name", default="gsm8k", choices=["gsm8k", "math"])

    # 设备
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=42)

    # 恢复
    parser.add_argument("--resume_checkpoint", default=None)

    return parser.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = Config(
        llm_name=args.llm_name,
        encoder_name=args.encoder_name,
        torch_dtype=args.torch_dtype,
        N=args.N,
        L=args.L,
        k=args.k,
        H=args.H,
        G=args.G,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        device=args.device,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("  Parameterized Experience TF-GRPO")
    print("  设备:   ", config.device)
    print("  模型:   ", config.llm_name)
    print("  数据集: ", config.dataset_name)
    print(f"  N={config.N}  L={config.L}  k={config.k}  G={config.G}")
    print("=" * 60 + "\n")

    # 构建模型
    model = ParameterizedExperienceModel(config)

    if args.resume_checkpoint:
        model.load_checkpoint(args.resume_checkpoint)

    # 加载数据集
    print(f"[Data] 加载 {config.dataset_name} ...")
    dataset = MathDataset(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        max_samples=config.max_train_samples,
    )
    print(f"[Data] 样本数: {len(dataset)}")

    # 训练
    trainer = GRPOTrainer(model, config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
