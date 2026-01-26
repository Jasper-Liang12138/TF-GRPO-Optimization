import sys
import os
import argparse
import random
from tf_grpo_deepseek import TF_GRPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser("Build Experience Bank with DEEPSEEK API")

    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("DEEPSEEK_API_KEY"),
        help="DEEPSEEK_API_KEY (optional if env var set)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-chat",
        help="Model name to use",
    )
    parser.add_argument(
        "--dapo_parquet",
        type=str,
        default="dataset/dapo-math-17k.parquet",
        help="DAPO-Math-17K parquet path",
    )
    parser.add_argument(
        "--exp_size",
        type=int,
        default=50,
        help="Number of samples to build experience bank",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Group rollout size for TF-GRPO",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Error: OpenAI API Key is required via --api_key or OPENAI_API_KEY env var.")
        return

    print("="*50)
    print(f"[Init] API Mode")
    print(f"       Model: {args.model_name}")
    print("="*50)

    # 初始化 TF_GRPO (API版)
    print("[Init] Initializing TF_GRPO (API)...")
    tf_grpo = TF_GRPO(
        api_key=args.api_key,
        model_name=args.model_name,
        group_size=args.group_size,
        max_experiences=args.exp_size,
        max_new_tokens=args.max_new_tokens
    )

    # Build Experience Bank
    print(f"[Run] Building Experience Bank (Size: {args.exp_size})...")
    if not os.path.exists(args.dapo_parquet):
        print(f"Error: Parquet file not found: {args.dapo_parquet}")
        return

    tf_grpo.build_experience_from_dapo_epochs(
        parquet_path=args.dapo_parquet,
        sample_size=args.exp_size,
        epochs=args.epochs
    )

    print("\n✅ Build finished Successfully")

if __name__ == "__main__":
    main()