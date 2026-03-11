"""
math_inference.py — 推理脚本

对应原版 math_inference.py，将 API 调用替换为本地模型推理。
逻辑完全一致：
  - 加载经验库 JSON（与原版格式相同）
  - 格式化为带 Confidence 分数的文本块注入 system prompt
  - 循环推理，输出正确率

用法示例：

    # 带经验库推理（TF-GRPO 模式）
    python math_inference.py \
        --dataset gsm8k \
        --data_path ./dataset/gsm8k/test.json \
        --save_path ./results/gsm8k_tfgrpo.json \
        --experience_bank_path ./output_logs/experience_bank_epoch_3.json \
        --IF_TF_GRPO_MODE

    # Zero-shot 基线推理
    python math_inference.py \
        --dataset gsm8k \
        --data_path ./dataset/gsm8k/test.json \
        --save_path ./results/gsm8k_zeroshot.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

from tqdm import tqdm

# ── 昇腾 NPU ──────────────────────────────────────────────────────────────
try:
    import torch_npu                               # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    print("[NPU] torch_npu 加载成功")
except ImportError:
    print("[NPU] 未找到 torch_npu，回退至 CUDA / CPU")

from tf_grpo import TF_GRPO


# ====================== 答案抽取（与原版一致） ======================

def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    matches = re.findall(r"Answer\s*:\s*([A-E])", sentence_, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    fallback = re.findall(r"\b([A-E])\b", sentence_)
    if fallback:
        return fallback[-1].upper()
    return ""


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    preds = re.findall(r"-?\d+\.?\d*", sentence)
    if not preds:
        return float("inf")
    try:
        return float(preds[-1])
    except ValueError:
        return float("inf")


# ====================== Prompt 构造（与原版一致） ======================

def build_inference_prompt(question: str, formatted_experiences: str = "") -> list:
    system_content = "You are an advanced math problem solver.\n\n"

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

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Current Problem: {question}\n"},
    ]


def build_aqua_prompt(question_with_choices: str, formatted_experiences: str = "") -> list:
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

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Current Problem: {question_with_choices}\n"},
    ]


def format_experience_bank(experience_data: list) -> str:
    """
    将结构化经验库转换为 Prompt 文本（与原版 format_experience_bank 完全一致）。
    """
    formatted_blocks = []
    for idx, entry in enumerate(experience_data):
        problem_text = entry.get("problem", "").strip()
        experiences = entry.get("experiences", [])
        if not experiences:
            continue

        block_lines = [
            f"=== [Case Study {idx + 1}] ===",
            f"Problem: {problem_text}",
            "Reference Insights:",
        ]
        for exp_item in experiences:
            score = exp_item.get("score", 0.0)
            content = ""
            label = "Insight"
            for k, v in exp_item.items():
                if k != "score":
                    label = k
                    content = v
                    break
            if content:
                block_lines.append(f"  * [{label}] (Confidence: {score:.2f}): {content}")

        formatted_blocks.append("\n".join(block_lines))

    return "\n\n".join(formatted_blocks)


# ====================== 参数解析 ======================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-GRPO Baseline — 推理评估（本地 Qwen2.5-Math-7B-Instruct）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["AQuA", "gsm8k", "SVAMP"],
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="results.json")
    parser.add_argument(
        "--experience_bank_path",
        type=str,
        default=None,
        help="经验库 JSON 路径（build_experience.py 生成）",
    )
    parser.add_argument(
        "--IF_TF_GRPO_MODE",
        action="store_true",
        default=False,
        help="启用经验增强推理模式（对应原版 --IF_TF_GRPO_MODE）",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )
    return parser.parse_args()


# ====================== 主函数 ======================

def main() -> None:
    args = parse_args()

    # 确保输出目录存在
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 1. 初始化模型（仅用 call_llm，group_size 无关紧要）
    print(f"[Init] 加载模型: {args.model}")
    agent = TF_GRPO(
        model_name=args.model,
        group_size=1,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    # 2. 加载经验库（与原版一致）
    global_experiences_text = ""
    if args.IF_TF_GRPO_MODE and args.experience_bank_path:
        print(f"[Step 1] 加载经验库: {args.experience_bank_path}")
        try:
            with open(args.experience_bank_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            print(f"共加载 {len(raw_data)} 条案例。")
            global_experiences_text = format_experience_bank(raw_data)
            print("\n--- Experience Text Preview ---")
            print(global_experiences_text[:500])
            print("-------------------------------\n")
        except Exception as e:
            print(f"[Error] 经验库加载失败: {e}")
            return

    # 3. 加载数据集
    if args.data_path:
        dataset_path = args.data_path
    else:
        dataset_path = f"dataset/{args.dataset}/test.json"

    print(f"[Step 2] 加载数据集: {args.dataset} from {dataset_path}")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: 数据集文件未找到: {dataset_path}")
        return

    # 4. 推理循环（与原版逻辑一致）
    correct = 0
    results = []

    print(f"[Step 3] 开始推理（共 {len(dataset)} 条）...")
    for idx, data in enumerate(tqdm(dataset)):
        question = data.get("instruction", "") or data.get("question", "") or data.get("sQuestion", "")
        gold_answer = data.get("answer", data.get("lSolutions", ""))

        if args.dataset == "AQuA":
            messages = build_aqua_prompt(question, global_experiences_text)
        else:
            messages = build_inference_prompt(question, global_experiences_text)

        # 保存第一条 prompt 用于调试（与原版一致）
        if idx == 0:
            debug_file = os.path.join(save_dir or ".", "debug_prompt_problem_1.json")
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=4, ensure_ascii=False)
            print(f"\n[DEBUG] 第 1 题完整 prompt 已保存至: {debug_file}\n")

        # 推理阶段用低温（与原版一致，temperature=0.3）
        output = agent.call_llm(messages, temperature=0.3)

        # 评估（与原版一致）
        if args.dataset == "AQuA":
            pred = extract_answer_letter(output)
            flag = pred == str(gold_answer).strip()
        else:
            try:
                label_num = float(str(gold_answer).replace(",", ""))
            except ValueError:
                label_num = float("inf")
            pred_num = extract_answer_number(output)
            flag = label_num != float("inf") and abs(label_num - pred_num) < 1e-3

        if flag:
            correct += 1
        current_acc = correct / (idx + 1)

        res = {
            "question": question,
            "gold": gold_answer,
            "pred": pred if args.dataset == "AQuA" else pred_num,
            "output": output,
            "flag": flag,
        }
        results.append(res)
        print(res)
        print(f"Accuracy: {current_acc:.4f} | Flag: {flag}")

        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
            print(f"Idx: {idx+1} | Acc: {current_acc:.4f}")

        # 实时写结果（与原版一致）
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"{'='*30}")
    print(f"Final Accuracy: {correct / len(dataset):.4f}")
    print(f"Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
