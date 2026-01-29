import json
import sys
import os
import re
import copy
import argparse
import random
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from tf_grpo_gptoss120b import TF_GRPO

# 假设 inference_math.py 在 LLM_Adapters/RLHF/ 下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# ====================== 答案抽取 ======================
def extract_answer_letter(sentence: str) -> str:
    """
    多选题（目前仅 AQuA）选项字母抽取逻辑，参考 evaluate.py：
    - 直接从文本中抽取 A-E，第一个匹配即为预测。
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        return pred_answers[0]
    return ""


def extract_answer_number(sentence: str) -> float:
    """
    数值型答案抽取逻辑，参考 evaluate.py：
    - 去掉逗号
    - 正则抓取所有数字/小数，取最后一个
    - 解析失败或不存在时返回 inf，方便后续判错
    """
    sentence = sentence.replace(",", "")
    preds = re.findall(r"-?\d+\.?\d*", sentence)
    if not preds:
        return float("inf")
    try:
        pred_answer = float(preds[-1])
    except ValueError:
        return float("inf")
    return pred_answer


def is_choice_dataset(dataset: str) -> bool:
    """
    当前只有 AQuA 是多选题，需要输出选项字母；其他数据集默认输出数值。
    """
    return dataset.lower() == "aqua"


def build_aqua_prompt(question_with_choices: str) -> str:
    """
    AQuA 是多选题，题干里通常包含 Answer Choices。
    强制模型只输出选项字母，避免输出空的 Answer: 或输出数值。
    """
    return (
        "Solve the following multiple-choice math problem.\n"
        "Pick exactly one option from {A, B, C, D, E}.\n"
        "Do NOT output the option text.\n"
        "Your output must end with exactly one line:\n"
        "Answer: <A/B/C/D/E>\n\n"
        f"Problem:\n{question_with_choices}\n"
    )


# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser("TF-GRPO Inference on Math Datasets")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base LLM, e.g. yahma/llama-7b-hf",
    )
    # 兼容旧参数：仅 AQuA 使用
    parser.add_argument(
        "--aqua_path",
        type=str,
        default="dataset/AQuA/test.json",
        help="(only for AQuA) Path to AQuA test.json",
    )
    # 通用数据路径，如果提供则覆盖默认 dataset 路径
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset test.json; if not set, use dataset/{dataset}/test.json "
             "for non-AQuA datasets and --aqua_path for AQuA.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AQuA",
        choices=["AQuA", "AddSub", "MultiArith", "SingleEq", "gsm8k", "SVAMP", "mawps", "mathqa"],
        help="Math dataset name. AQuA 输出选项字母，其他数据集输出数值答案。",
    )
    parser.add_argument(
        "--dapo_parquet",
        type=str,
        default="dataset/dapo-math-17k.parquet",
        help="DAPO-Math-17K parquet (used only to build experience bank)",
    )
    parser.add_argument(
        "--exp_size",
        type=int,
        default=100,
        help="Number of DAPO samples to build experience bank",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Group rollout size for TF-GRPO",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiment/tf_grpo_math_results.json",
        help="Path to save inference results",
    )
    parser.add_argument(
        "--load_8bit",
        default=False,
        action="store_true",
        help="Load model in 8-bit precision to save GPU memory",
    )
    parser.add_argument(
        "--experience_bank_path",
        type=str,
        default=None,
        help="Path to prebuilt experience_bank.json",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per step",
    )
    return parser.parse_args()

def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')

    load_8bit = args.load_8bit
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,   # A800 推荐 bf16
            attn_implementation="sdpa"
        ) 
       
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
       
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model

# ====================== main ======================
def main():
    torch.cuda.empty_cache()
    args = parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ========== Load model ==========
    print("[Load] Base model")
    tokenizer, model = load_model(args)

    # ========== Init TF-GRPO ==========
    tf_grpo = TF_GRPO(
        model=model,
        tokenizer=tokenizer,
        group_size=args.group_size,
        max_experiences=args.exp_size,
        max_new_tokens=args.max_new_tokens
    )

    # ========== Load or Build Experience Bank ==========
    if args.experience_bank_path and os.path.exists(args.experience_bank_path):
        print(f"[Step 1] Load Experience Bank from {args.experience_bank_path}")
        with open(args.experience_bank_path, "r") as f:
            experience_bank_data = json.load(f)
        tf_grpo.load_experience_bank(experience_bank_data)  # 假设 TF_GRPO 有 load_experience_bank 方法
    else:
        print("[Step 1] Build Experience Bank from DAPO")
        tf_grpo.build_experience_from_dapo_epochs(
            parquet_path=args.dapo_parquet,
            sample_size=args.exp_size,
        )

    # ========== Load dataset ==========
    if args.data_path:
        dataset_path = args.data_path
    else:
        if is_choice_dataset(args.dataset):
            dataset_path = args.aqua_path
        else:
            dataset_path = os.path.join("dataset", args.dataset, "test.json")

    print(f"[Step 2] Load dataset: {args.dataset} from {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # 固定随机种子，保证可复现
    random.seed(42)

    # 随机抽 100 条（如果不足 100 条则全用）
    sample_size = min(100, len(dataset))
    dataset_samples = random.sample(dataset, sample_size)

    total = len(dataset_samples)
    correct = 0
    results = []

    print(f"[Step 3] TF-GRPO inference on {args.dataset}")

    # 与 evaluate.py 风格一致的进度条与评价逻辑
    miss = 1e-3
    pbar = tqdm(total=total)
    for idx, data in enumerate(dataset_samples):
        question = data.get("instruction", "")

        # 获取标签
        gold_answer = data.get("answer", "")
        
        if is_choice_dataset(args.dataset):
            # AQuA：多选题 prompt
            prompt = tf_grpo.build_aqua_prompt(question)
            top_experiences = tf_grpo.extract_similar_experiences(question, args.group_size)
            outputs = []  # 用来收集每个经验生成的 output
            for exp_idx, exp in enumerate(top_experiences, 1):
                # 每次从 prompt 开始，添加当前经验
                prompt_with_exp = prompt
                prompt_with_exp[1]["content"] += "Use the following reasoning experiences internally to help your solution, " \
                                    "but do NOT copy them verbatim into your answer.\n"
                prompt_with_exp[1]["content"] += f"[Experience {exp_idx}]\n{exp}\n\n"
                        
                out = tf_grpo.batch_group_generate(prompt_with_exp)[0]
                reasoning = tf_grpo.extract_reasoning_from_output(prompt_with_exp, out)
                if reasoning:    
                    outputs.append(reasoning)
            best = tf_grpo.select_best(outputs, gold_answer) if outputs else prompt
        else:
            # 其他数学数据集：数值题 prompt
            prompt = tf_grpo.build_prompt_inference(question)
            top_experiences = tf_grpo.extract_similar_experiences(question, args.group_size)
            outputs = []  # 用来收集每个经验生成的 output
            for exp_idx, exp in enumerate(top_experiences, 1):
                # 每次从 prompt 开始，添加当前经验
                prompt_with_exp = prompt
                prompt_with_exp[1]["content"] += "Use the following reasoning experiences internally to help your solution, " \
                                          "but do NOT copy them verbatim into your answer."
                prompt_with_exp[1]["content"] += f"[Experience {exp_idx}]\n{exp}\n\n"
                        
                out = tf_grpo.batch_group_generate(prompt_with_exp)[0]
                reasoning = tf_grpo.extract_reasoning_from_output(prompt_with_exp, out)
                if reasoning:    
                    outputs.append(reasoning)
            best = tf_grpo.select_best(outputs, gold_answer) if outputs else prompt

        if is_choice_dataset(args.dataset):
            pred = extract_answer_letter(best)
            flag = (pred == gold_answer)
        else:
            # 数值题：将标签转为 float，再与预测数值进行近似比较
            label_num = gold_answer
            if isinstance(label_num, str):
                try:
                    label_num = float(label_num.replace(",", ""))
                except ValueError:
                    label_num = float("inf")
            try:
                label_num = float(label_num)
            except (TypeError, ValueError):
                label_num = float("inf")

            pred_num = extract_answer_number(best)
            pred = pred_num
            flag = abs(label_num - pred_num) <= miss

        if flag:
            correct += 1

        record = copy.deepcopy(data)
        record["output_pred"] = best
        record["pred"] = pred
        record["flag"] = flag
        results.append(record)

        print("\n---------------")
        print(best)
        print("prediction:", pred)
        print("answer:", gold_answer)
        print("---------------")
        print(
            f"test:{idx + 1}/{total} | "
            f"accuracy {correct} {correct / (idx + 1):.4f}"
        )

        # 与 evaluate.py 一样，每条样本都写一次结果文件
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=4)

        pbar.update(1)

    pbar.close()

    print("\nTest finished")
    print(f"Final {args.dataset} Accuracy: {correct / total:.4f}")
    print(f"Results saved to: {args.save_path}")


if __name__ == "__main__":
    main()





