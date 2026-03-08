import sys
import os
import argparse
import random
from tf_grpo_deepseek import TF_GRPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Run Training-Free GRPO on Math Datasets")
    
    # 必需参数
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPSEEK_API_KEY"), help="Your DeepSeek Key")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input .parquet file")
    
    # 可选参数
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com", help="LLM Base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name (e.g., deepseek-chat, gpt-4o)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of refinement epochs")
    parser.add_argument("--group_size", type=int, default=4, help="Group size (G) for GRPO sampling")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of problems to sample from dataset")
    parser.add_argument("--output_dir", type=str, default="./output_logs", help="Directory to save results")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens for generation")

    args = parser.parse_args()

    # 初始化并运行
    agent = TF_GRPO(
        api_key=args.api_key,
        model_name=args.model,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens
    )

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    print(">>> Starting TF-GRPO <<<")
    print(f"Model: {args.model}")
    print(f"Group Size: {args.group_size}")
    print(f"Sample Size: {args.sample_size}")
    
    agent.train_loop(
        parquet_path=args.data_path, 
        epochs=args.epochs, 
        sample_size=args.sample_size
    )

if __name__ == "__main__":
    main()