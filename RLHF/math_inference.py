import json
import sys
import os
import argparse
import re
from tqdm import tqdm
from tf_grpo_deepseek import TF_GRPO

# ====================== 答案抽取辅助函数 ======================
def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    # 1. 优先匹配标准的 "Answer: X" 格式
    matches = re.findall(r"Answer\s*:\s*([A-E])", sentence_, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # 2. 保底策略：提取文本中最后出现的独立字母 A-E
    fallback_matches = re.findall(r"\b([A-E])\b", sentence_)
    if fallback_matches:
        return fallback_matches[-1].upper()
    return ""

def extract_answer_number(sentence: str) -> float:
    # 移除千分位逗号等
    sentence = sentence.replace(",", "")
    # 匹配数字
    preds = re.findall(r"-?\d+\.?\d*", sentence)
    if not preds: return float("inf")
    try: return float(preds[-1])
    except ValueError: return float("inf")

# ====================== 构建 Prompt 相关函数 ======================

def build_inference_prompt(question: str, experiences_text: str = "") -> list:
    """
    构建推理用的 Chat 消息。
    如果有经验库文本，则将其拼接到 System Prompt 中。
    """
    system_content = "You are a math problem solver."
    
    if experiences_text:
        system_content += (
            "\n\nHere are some learned experiences and insights from similar problems. "
            "Please refer to them to solve the current problem better:\n"
            f"{experiences_text}\n\n"
            "Now, solve the following problem step by step."
        )
    else:
        system_content += "Please solve the problems."

    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": f"Problem: {question}\n"
        }
    ]
    return messages

def build_aqua_prompt(question_with_choices: str, experiences_text: str = "") -> list:
    """
    针对 AQuA 数据集的多选题格式构建 Prompt
    """
    system_content = (
        "You are a math problem solver. Please Solve the following multiple-choice math problem. "
        "Pick exactly one option from {A, B, C, D, E}. Answer: <A/B/C/D/E>"
    )

    if experiences_text:
        system_content += (
            "\n\nHere are some learned experiences and insights.\n"
            "Please refer to them to solve the problem better:\n"
            f"{experiences_text}\n"
        )

    messages = [
        {
            "role": "system",
            "content": system_content
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
    
    parser.add_argument("--save_path", type=str, default="results_api.json")
    parser.add_argument("--experience_bank_path", type=str, default=None)
    
    # 模式开关：True表示带经验推理，False表示Zero-shot直接推理
    parser.add_argument("--IF_TF_GRPO_MODE", action="store_true", help="Enable Experience-Augmented Inference", default=False)

    return parser.parse_args()

def main():
    args = parse_args()
    if not args.api_key:
        print("Error: API Key missing. Please provide --api_key or set DEEPSEEK_API_KEY env var.")
        return

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 1. 初始化 TF_GRPO (主要用于调用其封装好的 LLM 接口)
    print(f"[Init] Initializing Agent with model {args.model_name}...")
    agent = TF_GRPO(
        api_key=args.api_key,
        model_name=args.model_name,
        group_size=1  # 推理阶段不需要分组采样
    )

    # 2. 准备经验库文本 (核心修改部分)
    global_experiences_text = ""
    
    if args.IF_TF_GRPO_MODE:
        if args.experience_bank_path and os.path.exists(args.experience_bank_path):
            print(f"[Step 1] Loading Experience Bank from {args.experience_bank_path}")
            with open(args.experience_bank_path, "r") as f:
                raw_data = json.load(f)
            
            # 处理经验库格式
            # 假设 raw_data 是 train_loop 输出的格式: [{"index": 0, "experiences": ["exp1", "exp2"]}, ...]
            # 或者是简单的字符串列表
            exp_list = []
            if isinstance(raw_data, list):
                for item in raw_data:
                    if isinstance(item, dict) and "experiences" in item:
                        # 展平 experiences
                        for exp in item["experiences"]:
                            if exp and isinstance(exp, str):
                                exp_list.append(exp.strip())
                    elif isinstance(item, str):
                        exp_list.append(item.strip())
            
            # 去重（可选）并拼接
            unique_exps = list(set(exp_list))
            print(f"         Loaded {len(unique_exps)} unique experiences.")
            
            # 拼接成一个大字符串，每个经验用序号标记
            global_experiences_text = "\n".join([f"[{i+1}] {e}" for i, e in enumerate(unique_exps)])
            
        else:
            print("[Warning] TF-GRPO mode enabled but no experience bank file found/provided. Running without experiences.")
    else:
        print("[Step 1] Standard Zero-Shot Mode (No Experience Bank).")

    # 3. 加载数据集
    if args.data_path:
        dataset_path = args.data_path
    else:
        dataset_path = f"dataset/{args.dataset}/test.json"

    print(f"[Step 2] Load dataset: {args.dataset} from {dataset_path}")
    try:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    # 4. 推理循环
    correct = 0
    results = []
    
    print(f"[Step 3] Start Inference Loop (Total: {len(dataset)})...")
    
    for idx, data in enumerate(tqdm(dataset)):
        question = data.get("instruction", "") or data.get("question", "")
        gold_answer = data.get("answer", "")

        # 根据数据集类型构建 Prompt
        # 如果 global_experiences_text 为空，则相当于普通 Zero-shot
        if args.dataset == "AQuA":
            messages = build_aqua_prompt(question, global_experiences_text)
        else:
            messages = build_inference_prompt(question, global_experiences_text)

        # 调用 LLM
        # 使用 TF_GRPO 类中的 call_llm 方法
        output = agent.call_llm(messages, temperature=0.3) # 推理阶段通常使用低温 greedy decoding

        # 评估
        if args.dataset == "AQuA":
            pred = extract_answer_letter(output)
            # AQuA 的 gold answer 通常是一个字母
            flag = (pred == str(gold_answer).strip())
        else:
            # 数值比较逻辑
            try:
                label_num = float(str(gold_answer).replace(",", ""))
            except:
                label_num = float("inf")
            
            pred_num = extract_answer_number(output)
            
            # 简单的数值容错比较
            if label_num == float("inf"):
                flag = False
            else:
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
        
        # 仅每10条或最后一条打印一次 log，避免刷屏
        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
             print(f"Idx: {idx+1} | Acc: {current_acc:.4f} | Flag: {flag}")

        # 实时写文件
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"{'='*30}")
    print(f"Final Accuracy: {correct / len(dataset):.4f}")
    print(f"Results saved to {args.save_path}")

if __name__ == "__main__":
    main()