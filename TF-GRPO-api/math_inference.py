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

def build_inference_prompt(question: str, formatted_experiences: str = "") -> list:
    system_content = (
        "You are an advanced math problem solver.\n\n"
    )
    
    if formatted_experiences:
        system_content += (
            "### LEARNING FROM HISTORY\n"
            "Below are historical problems and the insights/experiences derived from them.\n"
            "Pay attention to the 'Confidence' scores. High confidence means the insight was verified as highly effective.\n\n"
            f"{formatted_experiences}\n\n"
            "### YOUR TASK\n"
            "Refer to the logic and insights from the cases above to solve the new problem.\n"
        )
    else:
        system_content += "Please solve the problem step by step."

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Current Problem: {question}\n"}
    ]
    return messages

def build_aqua_prompt(question_with_choices: str, formatted_experiences: str = "") -> list:
    """
    针对 AQuA 数据集的多选题格式构建 Prompt
    """
    system_content = (
        "You are a math problem solver. Please Solve the following multiple-choice math problem. "
        "Pick exactly one option from {A, B, C, D, E}. Answer: <A/B/C/D/E>"
    )

    if formatted_experiences:
        system_content += (
            "### LEARNING FROM HISTORY\n"
            "Below are historical problems and the insights/experiences derived from them.\n"
            "Pay attention to the 'Confidence' scores. High confidence means the insight was verified as highly effective.\n\n"
            f"{formatted_experiences}\n\n"
            "### YOUR TASK\n"
            "Refer to the logic and insights from the cases above to solve the new problem.\n"
        )
    else:
        system_content += "Please solve the problem step by step."

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Current Problem: {question_with_choices}\n"}
    ]
    return messages

def format_experience_bank(experience_data: list[dict]) -> str:
    """
    将结构化的经验库转换为 Prompt 友好的文本格式。
    保留问题上下文、经验类型（attempt/summary）和评分。
    """
    formatted_blocks = []
    
    for idx, entry in enumerate(experience_data):
        # 1. 获取问题文本 (去除首尾空格)
        problem_text = entry.get("problem", "").strip()
        experiences = entry.get("experiences", [])
        
        if not experiences:
            continue

        # 2. 构建单个案例块
        # 使用分隔符让 LLM 清楚区分不同的案例
        block_lines = []
        block_lines.append(f"=== [Case Study {idx + 1}] ===")
        block_lines.append(f"Problem: {problem_text}") 
        block_lines.append("Reference Insights:")

        # 3. 遍历该问题下的所有经验条目
        for exp_item in experiences:
            # exp_item 结构示例: {"attempt1": "...", "score": 1.5}
            # 我们需要提取出非 "score" 的那个键作为标签
            score = exp_item.get("score", 0.0)
            
            content = ""
            label = "Insight"
            
            for k, v in exp_item.items():
                if k != "score":
                    label = k  # 例如 "attempt1" 或 "overall summary"
                    content = v
                    break
            
            # 只有当内容非空时才添加
            if content:
                # 格式化为: * [Label] (Conf: 1.5): Content
                # 将分数显式展示给 LLM，提示它关注高分经验
                block_lines.append(f"  * [{label}] (Confidence: {score:.2f}): {content}")

        # 将这个 Case 的所有行合并
        formatted_blocks.append("\n".join(block_lines))

    # 将所有 Case 用双换行符连接
    return "\n\n".join(formatted_blocks)

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

    # 2. 准备经验库文本
    global_experiences_text = ""
    
    if args.IF_TF_GRPO_MODE and args.experience_bank_path:
        print(f"[Step 1] Loading Experience Bank from {args.experience_bank_path}")
        try:
            with open(args.experience_bank_path, "r", encoding='utf-8') as f:
                raw_data = json.load(f) # 这是一个 List[Dict]
            
            print(f"Loaded {len(raw_data)} cases.")
            
            # === 使用新的格式化函数 ===
            # 这里会生成带有 Problem Context 和 Score 的结构化文本
            global_experiences_text = format_experience_bank(raw_data)
            
            # 打印前500字符检查一下格式对不对
            print("\n--- Experience Text Preview ---")
            print(global_experiences_text[:500]) 
            print("-------------------------------\n")
            
        except Exception as e:
            print(f"[Error] Failed to load experience bank: {e}")
            return

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

        # === 调试：保存第一道题的 Prompt ===
        if idx == 0:
            debug_file = "debug_prompt_problem_1.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=4, ensure_ascii=False)
            print(f"\n[DEBUG] Full prompt for the first problem saved to: {debug_file}")
            print(f"[DEBUG] Check this file to verify the experience bank format.\n")


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
        print(res)
        print(f"Accuracy: {current_acc:.4f} | Flag: {flag}")

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