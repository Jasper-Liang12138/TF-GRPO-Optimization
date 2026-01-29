import json
import sys
import os
import re
import argparse
import torch
from tqdm import tqdm
from typing import List, Dict, Any
from tf_grpo_LLaMA13b import TF_GRPO

# === 引入 vLLM 相关库 ===
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ====================== 1. 答案抽取辅助函数 (保持原逻辑) ======================
def extract_answer_letter(sentence: str) -> str:
    """针对 AQuA 等选择题提取 A-E"""
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
    """针对 GSM8K 等数值题提取数字"""
    # 优先寻找 \boxed{}
    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", sentence)
    if boxed_match:
        text_to_parse = boxed_match.group(1)
    else:
        # 移除逗号，从最后一行找
        text_to_parse = sentence.replace(",", "")
    
    preds = re.findall(r"-?\d+\.?\d*", text_to_parse)
    if not preds: return float("inf")
    try: return float(preds[-1])
    except ValueError: return float("inf")

# ====================== 2. 经验库格式化 (保持原逻辑) ======================
def format_experience_bank(experience_data: List[dict]) -> str:
    """
    将结构化的经验库转换为 Prompt 友好的文本格式。
    保留问题上下文、经验类型（attempt/summary）和评分。
    """
    formatted_blocks = []
    
    for idx, entry in enumerate(experience_data):
        # 1. 获取问题文本
        problem_text = entry.get("problem", "").strip()
        experiences = entry.get("experiences", [])
        
        if not experiences:
            continue

        # 2. 构建单个案例块
        block_lines = []
        block_lines.append(f"=== [Case Study {idx + 1}] ===")
        block_lines.append(f"Problem: {problem_text}") 
        block_lines.append("Reference Insights:")

        # 3. 遍历该问题下的所有经验条目
        has_content = False
        for exp_item in experiences:
            # 兼容不同格式的 key
            score = exp_item.get("score", 0.0)
            content = ""
            label = "Insight"
            
            # 寻找非 score 的键作为内容
            for k, v in exp_item.items():
                if k != "score" and isinstance(v, str) and v.strip():
                    label = k  # e.g., "attempt1", "overall summary"
                    content = v
                    break
            
            # 策略：全量拼接时，通常只保留正向经验 (Score > 0)
            if content and score > 0:
                block_lines.append(f"  * [{label}] (Confidence: {score:.2f}): {content}")
                has_content = True

        if has_content:
            formatted_blocks.append("\n".join(block_lines))

    return "\n\n".join(formatted_blocks)
    def __init__(self, model_name: str, tensor_parallel_size: int = 2):
        print(f"[Init] Loading vLLM Model: {model_name} (TP={tensor_parallel_size})")
        
        # 初始化 vLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            dtype="bfloat16", # A800 推荐 bf16
            enforce_eager=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.global_experiences_text = ""

    def load_experience_bank(self, path: str):
        """加载并全量格式化经验库"""
        if not path or not os.path.exists(path):
            print("[Warning] Experience bank path invalid. Running Zero-shot.")
            return

        print(f"[Step 1] Loading Experience Bank from {path}")
        try:
            with open(path, "r", encoding='utf-8') as f:
                raw_data = json.load(f)
            
            print(f"Loaded {len(raw_data)} cases.")
            self.global_experiences_text = format_experience_bank(raw_data)
            
            # Token 长度检查与预警
            token_len = len(self.tokenizer.encode(self.global_experiences_text))
            print(f"[Info] Total Experience Context Length: ~{token_len} tokens")
            if token_len > 3000:
                print("[Warning] Experience context is very long! It may truncate the problem.")
                
        except Exception as e:
            print(f"[Error] Failed to format experience bank: {e}")

    def build_messages(self, question: str, is_aqua: bool = False) -> List[Dict]:
        """构建 Chat 格式的消息列表"""
        
        # 1. System Prompt (包含全量经验)
        if is_aqua:
            base_instruction = (
                "You are a math problem solver. Please Solve the following multiple-choice math problem. "
                "Pick exactly one option from {A, B, C, D, E}. Answer: <A/B/C/D/E>"
            )
        else:
            base_instruction = "You are an advanced math problem solver. Solve the problem step by step."

        system_content = f"{base_instruction}\n\n"

        if self.global_experiences_text:
            system_content += (
                "### LEARNING FROM HISTORY\n"
                "Below are historical problems and the insights/experiences derived from them.\n"
                "Pay attention to the 'Confidence' scores. High confidence means the insight was verified as highly effective.\n\n"
                f"{self.global_experiences_text}\n\n"
                "### YOUR TASK\n"
                "Refer to the logic and insights from the cases above to solve the new problem.\n"
                "Use the insights to help you, but do NOT copy them verbatim.\n"
            )

        # 2. User Prompt
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Current Problem: {question}\n"}
        ]
        return messages

    def generate_one(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        """单次贪婪生成"""
        # 应用 Chat Template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            # 兜底：简单拼接
            prompt = ""
            for m in messages:
                prompt += f"{m['role'].upper()}: {m['content']}\n"
            prompt += "ASSISTANT:"

        sampling_params = SamplingParams(
            temperature=0.0, # Greedy Decoding for OOD stability
            max_tokens=max_tokens,
            top_p=1.0
        )
        
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

# ====================== 4. 参数解析与主流程 ======================

def parse_args():
    parser = argparse.ArgumentParser("TF-GRPO Local Inference (LLaMA-13B)")
    
    # 硬件与模型
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="HuggingFace Model ID")
    parser.add_argument("--tp", type=int, default=2, help="Tensor Parallel Size (Cards)")
    
    # 数据
    parser.add_argument("--dataset", type=str, default="AQuA", choices=["AQuA", "gsm8k", "SVAMP", "math"])
    parser.add_argument("--data_path", type=str, required=True, help="Path to test.json")
    parser.add_argument("--experience_bank_path", type=str, default=None, help="Path to experience bank json")
    
    # 输出
    parser.add_argument("--save_path", type=str, default="results_local.json")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--sample_size", type=int, default=-1, help="-1 for all")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 初始化 Agent
    agent = TF_GRPO(
        model_name=args.model,
        tensor_parallel_size=args.tp
    )

    # 2. 加载经验库 (如果路径存在)
    if args.experience_bank_path:
        agent.load_experience_bank(args.experience_bank_path)

    # 3. 加载数据集
    print(f"[Step 2] Load dataset: {args.dataset} from {args.data_path}")
    data_list = []
    try:
        with open(args.data_path, "r") as f:
            if args.data_path.endswith('.jsonl'):
                for line in f:
                    if line.strip(): data_list.append(json.loads(line))
            else:
                data_list = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 采样 (可选)
    if args.sample_size > 0 and args.sample_size < len(data_list):
        import random
        random.seed(42)
        data_list = random.sample(data_list, args.sample_size)

    # 4. 推理循环
    print(f"[Step 3] Start Inference Loop (Total: {len(data_list)})...")
    
    correct = 0
    results = []
    
    # 确定是否为 AQuA 模式
    is_aqua = (args.dataset == "AQuA")

    for idx, data in enumerate(tqdm(data_list)):
        question = data.get("instruction", "") or data.get("question", "") or data.get("problem", "")
        gold_answer = data.get("answer", "") or data.get("ground_truth", "")

        # 构建 Messages (全量拼接发生在内部)
        messages = agent.build_messages(question, is_aqua=is_aqua)

        # 调试：打印第一个问题的 Prompt 长度
        if idx == 0:
            full_text = messages[0]['content'] + messages[1]['content']
            print(f"\n[DEBUG] First Problem Prompt Length: {len(agent.tokenizer.encode(full_text))} tokens")

        # 生成
        output = agent.generate_one(messages, max_tokens=args.max_new_tokens)

        # 评估
        flag = False
        pred_val = None
        
        if is_aqua:
            pred_val = extract_answer_letter(output)
            # 简单的字母比较
            if pred_val and str(gold_answer).strip():
                flag = (pred_val == str(gold_answer).strip().upper())
        else:
            # 数值比较
            pred_val = extract_answer_number(output)
            try:
                label_num = float(str(gold_answer).replace(",", ""))
                if label_num != float("inf") and pred_val != float("inf"):
                    flag = abs(label_num - pred_val) < 1e-3
            except:
                flag = False

        if flag: correct += 1
        
        # 记录结果
        current_acc = correct / (idx + 1)
        res = {
            "question": question,
            "gold": gold_answer,
            "pred": pred_val,
            "output": output,
            "flag": flag
        }
        results.append(res)

        # 进度条显示
        if (idx + 1) % 10 == 0:
            tqdm.write(f"Idx: {idx+1} | Acc: {current_acc:.4f}")
            # 实时保存
            with open(args.save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # 最终保存
    print(f"\n{'='*30}")
    print(f"Final Accuracy: {correct / len(data_list):.4f}")
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {args.save_path}")

if __name__ == "__main__":
    main()