import json
import sys
import os
import argparse
import random
import copy
from typing import List
from tqdm import tqdm
from tf_grpo_deepseek import TF_GRPO

# ====================== 答案抽取辅助函数 ======================
def extract_answer_letter(sentence: str) -> str:
    import re
    sentence_ = sentence.strip()
    
    # 1. 优先匹配标准的 "Answer: X" 格式
    #    正则解释：
    #    Answer    : 匹配 "Answer" (忽略大小写)
    #    \s*:\s*   : 匹配冒号，允许冒号前后有空格
    #    ([A-E])   : 捕获 A 到 E 之间的任意一个字母
    matches = re.findall(r"Answer\s*:\s*([A-E])", sentence_, re.IGNORECASE)
    
    if matches:
        return matches[-1].upper()  # 返回列表中的最后一个（即最后一次出现的 Answer: X）

    # 2. 保底策略：如果没找到 "Answer: X"，则提取文本中最后出现的独立字母 A-E
    #    \b 表示单词边界，防止匹配到单词内部（如 'Apple' 中的 'A'）
    fallback_matches = re.findall(r"\b([A-E])\b", sentence_)
    if fallback_matches:
        return fallback_matches[-1].upper()
    
    return ""

def extract_answer_number(sentence: str) -> float:
    import re
    sentence = sentence.replace(",", "")
    preds = re.findall(r"-?\d+\.?\d*", sentence)
    if not preds: return float("inf")
    try: return float(preds[-1])
    except ValueError: return float("inf")
# ====================== 构建benchmark相关函数 ======================
def build_prompt_inference_without_grpo(question: str) -> List[dict[str, str]]:
    # 修改为 Chat 格式
    messages = [
        {
            "role": "system",
            "content": "You are a math problem solver. Please solve the problems."
        },
        {
            "role": "user",
            "content": f"Problem: {question}\n"
        }
    ]
    return messages

def build_aqua_prompt_without_grpo(question_with_choices: str) -> List[dict[str, str]]:
        # 修改为 Chat 格式
        messages = [
            {
                "role": "system",
                "content": "You are a math problem solver. Please Solve the following multiple-choice math problem.Pick exactly one option from {A, B, C, D, E}.Answer: <A/B/C/D/E>"
            },
            {
                "role": "user",
                "content": f"Problem: {question_with_choices}\n"
            }
        ]
        return messages

# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser("TF-GRPO Inference with deepseek-chat API")
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    
    parser.add_argument("--dataset", type=str, default="AQuA", choices=["AQuA", "gsm8k", "SVAMP"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dapo_parquet", type=str, default="dataset/dapo-math-17k.parquet")
    
    parser.add_argument("--exp_size", type=int, default=100)
    parser.add_argument("--group_size", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="results_api.json")
    parser.add_argument("--experience_bank_path", type=str, default=None)
    parser.add_argument("--IF_TF_GRPO_MODE", action="store_true", help="Add this flag to enable TF-GRPO mode", default=False)

    return parser.parse_args()

def main():
    args = parse_args()
    if not args.api_key:
        print("Error: API Key missing.")
        return

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 1. 初始化 TF_GRPO
    print(f"[Init] TF-GRPO using {args.model_name}...")
    tf_grpo = TF_GRPO(
        api_key=args.api_key,
        model_name=args.model_name,
        group_size=args.group_size,
        max_experiences=args.exp_size
    )

    # 2. 加载经验库
    if args.experience_bank_path and os.path.exists(args.experience_bank_path):
        print(f"[Step 1] Load Experience Bank from {args.experience_bank_path}")
        with open(args.experience_bank_path, "r") as f:
            data = json.load(f)
        tf_grpo.load_experience_bank(data)
    elif args.IF_TF_GRPO_MODE==True:
        print("[Step 1] Build Experience Bank (Fast mode)")
        # 如果没有现成的，快速构建一点点用于测试
        tf_grpo.build_experience_from_dapo_epochs(
            parquet_path=args.dapo_parquet,
            sample_size=20, 
            epochs=1
        )
    elif args.IF_TF_GRPO_MODE==False:
        print("[Step 1] Skip Experience Bank Loading")

    # 3. 加载数据集
    if args.data_path:
        dataset_path = args.data_path
    else:
        dataset_path = f"dataset/{args.dataset}/test.json"

    print(f"[Step 2] Load dataset: {args.dataset}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # 采样
    #random.seed(42)
    #samples = random.sample(dataset, min(50, len(dataset))) # API 较贵，建议先跑50条测试
    
    correct = 0
    results = []
    if_tf_grpo_mode = args.IF_TF_GRPO_MODE
    print(f"IF_TF_GRPO_MODE: {if_tf_grpo_mode}")
    if if_tf_grpo_mode:
        print(f"[Step 3] Inference Loop...")
        for idx, data in enumerate(tqdm(dataset)):
            question = data.get("instruction", "") or data.get("question", "")
            gold_answer = data.get("answer", "")

            # 检索经验
            top_experiences = tf_grpo.extract_similar_experiences(question, args.group_size)
            
            # 准备生成
            outputs_text = []
            
            # 根据数据集类型构建 Prompt
            if args.dataset == "AQuA":
                base_prompt = tf_grpo.build_aqua_prompt(question)
            else:
                base_prompt = tf_grpo.build_prompt_inference(question)

            # 循环调用 API (TF-GRPO 逻辑：结合不同经验生成)
            if not top_experiences:
                # 无经验，直接跑
                out = tf_grpo.batch_group_generate(base_prompt)[0]
                outputs_text.append(out)
            else:
                for exp_idx, exp in enumerate(top_experiences, 1):
                    p = copy.deepcopy(base_prompt)
                    p[0]["content"] += (
                        f"\nReference Experience {exp_idx} (do not copy verbatim):\n{exp}\n"
                    )
                    out = tf_grpo.batch_group_generate(p)[0]
                    outputs_text.append(out)

            # 选出最佳答案
            best_output = tf_grpo.select_best(outputs_text, gold_answer)
            
            # 评估
            if args.dataset == "AQuA":
                pred = extract_answer_letter(best_output)
                flag = (pred == gold_answer)
            else:
                # 数值比较逻辑
                try:
                    label_num = float(str(gold_answer).replace(",", ""))
                except:
                    label_num = float("inf")
                pred_num = extract_answer_number(best_output)
                flag = abs(label_num - pred_num) < 1e-3

            if flag: correct += 1
            

            # 实时计算并显示准确率
            current_acc = correct / (idx + 1)

            # 记录
            res = {
                "question": question, 
                "gold": gold_answer, 
                "pred": pred if args.dataset=="AQuA" else pred_num, 
                "output": best_output,
                "flag": flag
            }
            results.append(res)
            print(res)
            print(f"Acc:{current_acc}")
            
            # 实时写文件
            with open(args.save_path, "w") as f:
                json.dump(results, f, indent=4)

        print(f"Final Accuracy: {correct / len(dataset):.4f}")
    else:
        print(f"[Step 3] Build Benchmark...")
        for idx, data in enumerate(tqdm(dataset)):
            question = data.get("instruction", "") or data.get("question", "")
            gold_answer = data.get("answer", "")
          
            # 根据数据集类型构建 Prompt
            if args.dataset == "AQuA":
                base_prompt = build_aqua_prompt_without_grpo(question)
            else:
                base_prompt = build_prompt_inference_without_grpo(question)

            output = tf_grpo.generate_without_grpo(base_prompt)[0]
            # 评估
            if args.dataset == "AQuA":
                pred = extract_answer_letter(output)
                flag = (pred == gold_answer)
            else:
                # 数值比较逻辑
                try:
                    label_num = float(str(gold_answer).replace(",", ""))
                except:
                    label_num = float("inf")
                pred_num = extract_answer_number(output)
                flag = abs(label_num - pred_num) < 1e-3

            if flag: correct += 1
            
            # 实时计算并显示准确率
            current_acc = correct / (idx + 1)

            # 记录
            res = {
                "question": question, 
                "gold": gold_answer, 
                "pred": pred if args.dataset=="AQuA" else pred_num, 
                "output": output,
                "flag": flag
            }
            results.append(res)
            print(res)
            print(f"Acc:{current_acc}")
            
            # 实时写文件
            with open(args.save_path, "w") as f:
                json.dump(results, f, indent=4)

        print(f"Final Accuracy: {correct / len(dataset):.4f}")



if __name__ == "__main__":
    main()