import argparse
import os
import sys


try:
    from tf_grpo_LLaMA13b import TF_GRPO
except ImportError:
    print("Error: Could not import 'TF_GRPO_Local' from 'tf_grpo.py'.")
    print("Please make sure the class file is named 'tf_grpo.py' and is in the same directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Training-Free GRPO on Math Datasets (Local vLLM)")
    
    # === 必需参数 ===
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the input dataset file (.parquet, .json, or .jsonl)")
    
    # === 模型与硬件参数 ===
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf", 
                        help="Hugging Face Model ID or local path (default: Llama-2-13b)")
    parser.add_argument("--tp", "--tensor_parallel_size", type=int, default=2, dest="tp",
                        help="Number of GPUs to use for Tensor Parallelism (default: 2)")
    
    # === 算法超参数 ===
    parser.add_argument("--group_size", type=int, default=4, 
                        help="Group size (G) for GRPO sampling (default: 4)")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of refinement epochs (default: 3)")
    parser.add_argument("--sample_size", type=int, default=100, 
                        help="Number of problems to sample from dataset (default: 100)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, 
                        help="Max new tokens for generation (default: 2048)")
    
    # === 输出配置 ===
    parser.add_argument("--output_dir", type=str, default="./output_logs", 
                        help="Directory to save experience banks and logs")

    args = parser.parse_args()

    # 1. 检查数据路径
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    print(f"\n{'='*10} Starting TF-GRPO (Local) {'='*10}")
    print(f"Model ID     : {args.model}")
    print(f"GPUs (TP)    : {args.tp}")
    print(f"Data Path    : {args.data_path}")
    print(f"Sample Size  : {args.sample_size}")
    print(f"Group Size   : {args.group_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Output Dir   : {args.output_dir}")
    print(f"{'='*32}\n")

    # 2. 初始化 Agent (vLLM加载在这一步发生)
    # 注意：这里调用的是 TF_GRPO 类
    try:
        agent = TF_GRPO(
            model_name=args.model,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tp,
            # 如果你的类没有 output_dir 参数，请删除下面这行；
            # 但建议在类里加上 output_dir 以便管理输出路径
            output_dir=args.output_dir 
        )
    except TypeError:
        # 兼容旧版本如果不接受 output_dir 的情况
        agent = TF_GRPO(
            model_name=args.model,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tp
        )

    # 3. 开始训练循环
    agent.train_loop(
        parquet_path=args.data_path, 
        epochs=args.epochs, 
        sample_size=args.sample_size
    )
    
    print("\n>>> TF-GRPO Process Finished. <<<")

if __name__ == "__main__":
    main()