import re
import os
import random
import json
import statistics
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from fractions import Fraction
from transformers import AutoTokenizer

# 引入 vLLM
from vllm import LLM, SamplingParams

class TF_GRPO:
    def __init__(
        self,
        model_name: str,  # 这里传入 HuggingFace ID，例如 "meta-llama/Llama-2-13b-chat-hf"
        group_size=4,
        max_new_tokens=4096,
        tensor_parallel_size=2,  # A800 8卡设置为8
    ):
        print(f"[Init] Loading model '{model_name}' from HuggingFace (or cache)...")
        print(f"[Init] Tensor Parallel Size = {tensor_parallel_size}")
        
        # === 1. 初始化 vLLM ===
        # vLLM 会自动处理 HuggingFace 的下载和缓存
        # download_dir 参数可选，默认是 ~/.cache/huggingface
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.90, # 占用90%显存，留一点给系统
            dtype="bfloat16" # A800 建议使用 bf16
        )
        
        # 加载 tokenizer 用于处理对话模板
        # 同样会自动下载或从缓存读取
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        
        self.experience_bank: Dict[int, List[Dict[str, Any]]] = {}
        self.dataset: List[Dict] = []

    # ===================== 1. 基础工具 & 优势计算 (保持不变) =====================
    
    def extract_answer(self, text: str) -> str:
        patterns = [
            r"\\boxed\{([^{}]+)\}",
            r"Answer\s*[:\-]?\s*([^\n$]+)"
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().replace("$", "").replace("\\", "").strip()
        for l in reversed(text.splitlines()):
            l_clean = l.replace("$","").replace("\\","").strip()
            if re.match(r"^-?[\d\.\-/]+$", l_clean): return l_clean
        return ""

    def check_format(self, text: str) -> float:
        if r"\boxed{" in text:
            return 1.0
        return 0.0

    def check_process_quality(self, text: str) -> float:
        if len(text) < 50:
            return 0.0
        score = 0.0
        if "$" in text or "\\" in text:
            score += 0.5
        steps_keywords = ["step", "first", "therefore", "since", "because"]
        if any(w in text.lower() for w in steps_keywords):
            score += 0.5
        return min(1.0, score)

    def check_correctness(self, output: str, gold: str) -> float:
        pred = self.extract_answer(output)
        if not pred: return 0.0
        try:
            def clean_num(s):
                s = str(s).replace(",", "")
                if "/" in s: return float(Fraction(s))
                return float(s)
            return 1.0 if abs(clean_num(pred) - clean_num(str(gold))) < 1e-6 else 0.0
        except:
            return 1.0 if pred.strip() == str(gold).strip() else 0.0

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        if not rewards: return []
        if len(rewards) == 1: return [0.0]
        mean_r = statistics.mean(rewards)
        std_r = statistics.stdev(rewards)
        if std_r == 0:
            return [0.0] * len(rewards)
        return [(r - mean_r) / (std_r + 1e-8) for r in rewards]
    
    def compute_composite_reward(self, output: str, gold: str) -> float:
        r_acc = self.check_correctness(output, gold)
        r_fmt = self.check_format(output)
        r_proc = self.check_process_quality(output)
        w_acc, w_fmt, w_proc = 1.0, 0.1, 0.1
        total_reward = (r_acc * w_acc) + (r_fmt * w_fmt) + (r_proc * w_proc)
        return total_reward

    # ===================== 2. 本地 LLM 调用封装 =====================

    def _prepare_prompts(self, messages_list: List[List[Dict]]) -> List[str]:
        """将消息列表转换为模型输入的 Prompt 字符串"""
        prompts = []
        for msgs in messages_list:
            # 使用 chat template 自动适配 Llama-2/3/DeepSeek 格式
            try:
                formatted = self.tokenizer.apply_chat_template(
                    msgs, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                # 兜底：如果 tokenizer 没有 chat_template，手动简单拼接
                # print(f"[Warning] Chat template failed, using simple concat: {e}")
                formatted = ""
                for m in msgs:
                    formatted += f"Role: {m['role']}\nContent: {m['content']}\n"
                formatted += "Role: assistant\n"
                
            prompts.append(formatted)
        return prompts

    def batch_generate(self, messages_list: List[List[Dict]], temperature=0.7, json_mode=False) -> List[str]:
        """批量生成函数：充分利用 vLLM 的吞吐量"""
        prompts = self._prepare_prompts(messages_list)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=self.max_new_tokens if not json_mode else 2048,
            top_p=0.95,
        )
        
        # vLLM 执行推理
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)
        return results

    def extract_json_content(self, text: str) -> str:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return text[start:end+1]
            return text
        except:
            return text

    # ===================== 3. 核心算子实现 =====================

    def batch_summarize(self, question: str, outputs: List[str]) -> Dict[str, Any]:
        traces_text = ""
        for i, out in enumerate(outputs):
            traces_text += f"--- [Attempt {i+1}] ---\n{out[:2000]}...\n\n"

        prompt = (
            f"Problem: {question}\n\n"
            f"Here are {len(outputs)} different attempts to solve this problem:\n"
            f"{traces_text}\n"
            "--- TASK ---\n"
            "1. Analyze each attempt INDIVIDUALLY: Summarize method and errors.\n"
            "2. Analyze ALL attempts COLLECTIVELY: Provide overall insight.\n\n"
            "--- OUTPUT FORMAT ---\n"
            "Strictly Output JSON ONLY:\n"
            "{\n"
            '  "individual_summaries": ["summary 1", "summary 2", ...],\n'
            '  "overall_summary": "overall insight"\n'
            "}"
        )
        
        messages = [[{"role": "user", "content": prompt}]]
        res_text = self.batch_generate(messages, temperature=0.7, json_mode=True)[0]
        json_str = self.extract_json_content(res_text)
        
        try:
            data = json.loads(json_str)
            if not isinstance(data.get("individual_summaries"), list):
                data["individual_summaries"] = ["Error parsing"] * len(outputs)
            return data
        except:
            return {"individual_summaries": ["Summary failed"]*len(outputs), "overall_summary": "Summary failed"}

    def exp_controller(self, question: str, old_exps: List[Dict[str, Any]], observations: List[Dict]) -> List[Dict[str, Any]]:
        target_count = len(observations)

        formatted_old_exps = []
        if old_exps:
            for i, exp in enumerate(old_exps):
                key = "overall summary" if i == len(old_exps) - 1 else f"attempt{i+1}"
                formatted_old_exps.append({
                    key: exp.get("text", ""),
                    "score": exp.get("score", 0.0)
                })
            e_text = json.dumps(formatted_old_exps, indent=1)
        else:
            e_text = "[] (No prior experience)"
        
        obs_text = ""
        for i, obs in enumerate(observations):
            prefix = f"[Attempt {i+1}]" if i != len(observations)-1 else "[Overall Summary]"
            obs_text += (
                f"{prefix}\n"
                f"Advantage Score: {obs['advantage']:.4f}\n"
                f"Summary: {obs['summary']}\n\n"
            )

        system_prompt = "You are a meta-learning optimizer for math reasoning."
        user_prompt = (
            f"Current Problem: {question}\n\n"
            f"=== OLD EXPERIENCE BANK (With Advantage Scores) ===\n{e_text}\n\n"
            f"=== NEW OBSERVATIONS (G={self.group_size} rollouts + 1 global) ===\n{obs_text}\n\n"
            "=== INSTRUCTIONS ===\n"
            "Refine the Experience Bank based on the new Advantage Scores.\n"
            f"IMPORTANT: You MUST output exactly {target_count} experience items.\n"
            "Output strictly valid JSON:\n"
            "{\n"
            "  \"experiences\": [\n"
            "    {\"text\": \"Use method X...\", \"score\": 0.85},\n"
            "    {\"text\": \"Avoid error Y...\", \"score\": -0.5}\n"
            "  ]\n"
            "}"
        )

        messages = [[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]]
        
        res_text = self.batch_generate(messages, temperature=0.7, json_mode=True)[0]
        json_str = self.extract_json_content(res_text)
        default_item = {"text": "Think clearly and step by step.", "score": 0.0}
        
        try:
            data = json.loads(json_str)
            new_exps = data.get("experiences", [])
            
            cleaned_exps = []
            for item in new_exps:
                if isinstance(item, dict):
                    text_val = item.get("text", "")
                    if not text_val:
                        for k, v in item.items():
                            if k != "score" and isinstance(v, str):
                                text_val = v
                                break
                    cleaned_exps.append({"text": text_val, "score": float(item.get("score", 0.0))})
                elif isinstance(item, str):
                    cleaned_exps.append({"text": item, "score": 0.0})
            
            current_len = len(cleaned_exps)
            if current_len == target_count:
                return cleaned_exps
            elif current_len > target_count:
                return cleaned_exps[:target_count]
            else:
                missing = target_count - current_len
                if len(old_exps) >= target_count:
                    cleaned_exps.extend(old_exps[current_len:target_count])
                else:
                    filler = cleaned_exps[-1] if cleaned_exps else default_item
                    cleaned_exps.extend([filler.copy() for _ in range(missing)])
                return cleaned_exps

        except:
            if len(old_exps) == target_count: return old_exps
            if len(old_exps) > target_count: return old_exps[:target_count]
            return old_exps + [default_item] * (target_count - len(old_exps))

    # ===================== 4. 主训练循环 =====================

    def train_loop(self, parquet_path: str, epochs=3, sample_size=100):
        import pandas as pd
        
        if self.dataset:
            print(f"[TF-GRPO] Using existing dataset with {len(self.dataset)} samples.")
        else:
            print(f"[TF-GRPO] Loading dataset from {parquet_path}...")
            try:
                df = pd.read_parquet(parquet_path)
            except:
                print("Error reading parquet, creating dummy data for test...")
                df = pd.DataFrame([
                    {"prompt": [{"content": "What is 2+2?"}], "reward_model": {"ground_truth": "4"}}
                ] * 10)

            records = df.to_dict(orient="records")
            actual_size = min(sample_size, len(records))
            self.dataset = random.sample(records, actual_size)
            print(f"[TF-GRPO] Randomly sampled {actual_size} problems.")

        for ep in range(epochs):
            print(f"\n{'='*20} Epoch {ep+1}/{epochs} {'='*20}")
            pbar = tqdm(self.dataset)
            
            for idx, item in enumerate(pbar):
                try:
                    q = item["prompt"][0]["content"]
                    gold = item["reward_model"]["ground_truth"]
                except:
                    q = item.get("question", "") or item.get("prompt", "")
                    gold = item.get("answer", "") or item.get("ground_truth", "")

                current_experiences = self.experience_bank.get(idx, [])
                if current_experiences:
                    exp_lines = []
                    for i, e in enumerate(current_experiences):
                        text = e.get("text", "")
                        score = e.get("score", 0.0)
                        exp_lines.append(f"{i+1}. {text} (Confidence: {score:.2f})")
                    exp_str = "\n".join(exp_lines)
                    system_content = (
                        "You are a math problem solver. "
                        "Please refer to the following historical experiences to solve this problem:\n"
                        f"{exp_str}\n\n"
                        "Think step by step and put the final answer in \\boxed{}."
                    )
                else:
                    system_content = "You are a math problem solver. Think step by step and put the final answer in \\boxed{}."

                messages_template = [
                    {"role": "system", "content": system_content}, 
                    {"role": "user", "content": f"Problem: {q}\n"}
                ]
                
                batch_inputs = [messages_template] * self.group_size
                outputs = self.batch_generate(batch_inputs, temperature=0.7)

                rewards = [self.compute_composite_reward(o, gold) for o in outputs]
                advantages = self.compute_advantages(rewards)

                summary_result = self.batch_summarize(q, outputs)
                indiv_sums = summary_result.get("individual_summaries", [])
                overall_sum = summary_result.get("overall_summary", "")

                if len(indiv_sums) < self.group_size:
                    indiv_sums.extend(["Summary failed"] * (self.group_size - len(indiv_sums)))

                observations = []
                for i in range(self.group_size):
                    observations.append({"summary": indiv_sums[i], "advantage": advantages[i]})
                
                avg_adv = statistics.mean(advantages) if advantages else 0.0
                observations.append({"summary": f"[Global] {overall_sum}", "advantage": avg_adv})

                updated_experiences = self.exp_controller(q, current_experiences, observations)
                self.experience_bank[idx] = updated_experiences
                
                pbar.set_description(f"Ep {ep+1} | AvgAdv: {avg_adv:.2f}")

            # Save
            save_data = []
            for idx, exps in self.experience_bank.items():
                if idx < len(self.dataset):
                    try:
                        prob_text = self.dataset[idx]["prompt"][0]["content"]
                    except:
                        prob_text = str(self.dataset[idx])
                else:
                    prob_text = "Unknown Problem"
                
                formatted_exps = []
                for i, e in enumerate(exps):
                    key_name = "overall summary" if i == len(exps) - 1 else f"attempt{i+1}"
                    formatted_exps.append({
                        key_name: e.get("text", ""),
                        "score": e.get("score", 0.0)
                    })
                save_data.append({"problem": prob_text, "experiences": formatted_exps})

            with open(f"experience_bank_epoch_{ep+1}.json", "w", encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 配置区
    # 指定 Hugging Face 模型 ID
    # 常用: "meta-llama/Llama-2-13b-chat-hf", "deepseek-ai/deepseek-math-7b-instruct"
    # 注意：如果是 Llama 等受限模型，请先在终端运行 `huggingface-cli login`
    HF_MODEL_ID = "meta-llama/Llama-2-13b-chat-hf" 
    DATA_PATH = "math_dataset.parquet"
    
    # 初始化
    optimizer = TF_GRPO_Local(
        model_name=HF_MODEL_ID,
        group_size=8,        # 并行采样8条路径
        max_new_tokens=2048,
        tensor_parallel_size=8 # A800 8卡
    )
    
    # 开始训练
    optimizer.train_loop(DATA_PATH, epochs=3, sample_size=200)