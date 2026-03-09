"""
config.py — 所有超参数与路径配置
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── 模型路径 ────────────────────────────────────────────────────────────
    llm_name: str = "Qwen/Qwen2.5-Math-72B-Instruct"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # torch_dtype: float16 在昇腾 NPU 上更稳定；bfloat16 若 CANN 版本支持可切换
    torch_dtype: str = "float16"

    # ── Memory M 参数 ───────────────────────────────────────────────────────
    N: int = 64        # 记忆槽数量
    d_enc: int = 768   # Encoder 输出维度

    # ── Experience Aggregator 参数 ──────────────────────────────────────────
    L: int = 10        # soft prefix token 数（论文推荐值）
    k: int = 5         # top-k 检索条数
    H: int = 8         # Cross-Attention 头数
    d_llm: int = 7168  # Qwen2.5-Math-72B-Instruct 的 d_model

    # ── GRPO 参数 ───────────────────────────────────────────────────────────
    G: int = 8                 # 每题 rollout 采样数
    max_new_tokens: int = 512  # 生成最大长度
    temperature: float = 0.7
    top_p: float = 0.9

    # ── 优化器 ──────────────────────────────────────────────────────────────
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 50

    # ── Memory CRUD 阈值 ────────────────────────────────────────────────────
    mem_add_adv_threshold: float = 0.3      # advantage > 阈值 → 写入 M
    mem_delete_score_threshold: float = -2.0 # 累计分 < 阈值 → 删除槽
    mem_merge_cos_threshold: float = 0.95    # 余弦相似度 > 阈值 → 合并
    mem_score_decay: float = 0.99            # 槽分数衰减因子
    mem_crud_every: int = 20                 # 每隔多少 step 执行一次 prune+merge

    # ── 训练 ────────────────────────────────────────────────────────────────
    num_epochs: int = 3
    max_train_samples: Optional[int] = None
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "./checkpoints"

    # ── 数据集 ──────────────────────────────────────────────────────────────
    dataset_name: str = "gsm8k"   # "gsm8k" | "math"
    dataset_split: str = "train"

    # ── 设备（昇腾 NPU）───────────────────────────────────────────────────────
    device: str = "npu:0"

    # ── 随机种子 ────────────────────────────────────────────────────────────
    seed: int = 42
