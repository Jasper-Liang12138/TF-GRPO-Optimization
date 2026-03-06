import re
import os
import random
import json
import statistics
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from fractions import Fraction

class TF_GRPO:
    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-chat",
        group_size=5,
        max_new_tokens=4096,
    ):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_name = model_name
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        
        # 经验库：Key为题目索引，Value为字典列表 [{"text":Str, "score":Float}, ...]
        self.experience_bank: Dict[int, List[Dict[str, Any]]] = {}
        
        self.dataset: List[Dict] = []

    # ===================== 1. 基础工具 & 优势计算 =====================
    
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

    # === 格式检查 ===
    def check_format(self, text: str) -> float:
        """检查是否使用了标准格式，例如 \boxed{}"""
        # 给予微小奖励，鼓励模型遵守指令
        if r"\boxed{" in text:
            return 1.0
        return 0.0

    # === 过程质量检查（启发式） ===
    def check_process_quality(self, text: str) -> float:
        """
        基于规则的简单质量检查：
        1. 长度惩罚：太短可能是在瞎猜。
        2. 结构奖励：包含换行符、步骤词、LaTeX公式符号。
        """
        # 1. 长度检查 (过短给0分)
        if len(text) < 50:
            return 0.0
            
        score = 0.0
        # 2. LaTeX 密度 (粗略判断是否有数学推导)
        if "$" in text or "\\" in text:
            score += 0.5
            
        # 3. 步骤词 (简单的关键词匹配)
        steps_keywords = ["step", "first", "therefore", "since", "because"]
        if any(w in text.lower() for w in steps_keywords):
            score += 0.5
            
        return min(1.0, score) # 上限 1.0

    def check_correctness(self, output: str, gold: str) -> float:
        pred = self.extract_answer(output)
        if not pred: return 0.0
        try:
            def clean_num(s):
                s = s.replace(",", "")
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
        # 1. 计算各个维度的原始分 (0.0 ~ 1.0)
        r_acc = self.check_correctness(output, gold)
        r_fmt = self.check_format(output)
        r_proc = self.check_process_quality(output)
        
        # 2. 设定权重 (Weights)
        # 核心原则：正确性(Accuracy)必须占主导，格式和过程作为辅助(Shaping Rewards)
        w_acc = 1.0   # 正确性权重
        w_fmt = 0.1   # 格式奖励权重 (例如用了 \boxed{})
        w_proc = 0.1  # 过程质量权重 (长度、LaTeX、关键词)
        
        # 3. 计算加权总分
        total_reward = (r_acc * w_acc) + (r_fmt * w_fmt) + (r_proc * w_proc)
        
        return total_reward

    # ===================== 2. LLM 调用封装 =====================

    def call_llm(self, messages: List[Dict], temperature=0.7, json_mode=False) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_new_tokens if not json_mode else 2048,
                response_format={"type": "json_object"} if json_mode else None
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[API Error]: {e}")
            return ""

    # ===================== 3. 核心算子实现 =====================

    def batch_summarize(self, question: str, outputs: List[str]) -> Dict[str, Any]:
            traces_text = ""
            for i, out in enumerate(outputs):
                traces_text += f"--- [Attempt {i+1}] ---\n{out}...\n\n" # 简单截断防止过长

            prompt = (
                f"Problem: {question}\n\n"
                f"Here are {len(outputs)} different attempts to solve this problem:\n"
                f"{traces_text}\n"
                "--- TASK ---\n"
                "1. Analyze each attempt INDIVIDUALLY: Summarize method and errors using about 2-4 sentences.\n"
                "2. Analyze ALL attempts COLLECTIVELY: Provide overall insight.\n\n"
                "--- OUTPUT FORMAT ---\n"
                "Return JSON:\n"
                "{\n"
                '  "individual_summaries": ["summary 1", "summary 2", ...],\n'
                '  "overall_summary": "overall insight"\n'
                "}"
            )
            messages = [{"role": "user", "content": prompt}]
            res = self.call_llm(messages, temperature=0.7, json_mode=True)
            try:
                data = json.loads(res)
                if not isinstance(data.get("individual_summaries"), list):
                    data["individual_summaries"] = ["Error parsing"] * len(outputs)
                return data
            except:
                return {"individual_summaries": ["Summary failed"]*len(outputs), "overall_summary": "Summary failed"}

    def exp_controller(self, question: str, old_exps: List[Dict[str, Any]], observations: List[Dict]) -> List[Dict[str, Any]]:
        """
        Input old_exps format: [{"text": "...", "score": 1.2}, ...]
        Returns updated experience list with same length as observations.
        """
        target_count = len(observations)

        # === 1. 将旧经验序列化为可视化更强的格式 (Attempt X / Overall Summary) ===
        formatted_old_exps = []
        if old_exps:
            for i, exp in enumerate(old_exps):
                # 最后一个默认为 overall summary，前面的为 attempt
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
            "1. **KEEP/MODIFY**: Retain or improve experiences from high-advantage attempts.\n"
            "2. **DELETE**: Remove experiences linked to negative advantages.\n"
            "3. **ADD**: Extract new insights from high-advantage attempts.\n"
            "4. **ASSIGN SCORE**: For each experience, assign a 'score' equal to the Advantage of the attempt it was derived from.\n\n"
            "Output strictly valid JSON (standard 'text' key is fine for output, I will format it later):\n"
            "{\n"
            "  \"experiences\": [\n"
            "    {\"text\": \"Use method X...\", \"score\": 0.85},\n"
            "    {\"text\": \"Avoid error Y...\", \"score\": -0.5}\n"
            "  ]\n"
            "}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        res = self.call_llm(messages, temperature=0.7, json_mode=True)
        
        default_item = {"text": "Think clearly and step by step.", "score": 0.0}
        
        try:
            data = json.loads(res)
            new_exps = data.get("experiences", [])
            
            # 格式清洗
            cleaned_exps = []
            for item in new_exps:
                if isinstance(item, dict) and "text" in item:
                    if "score" not in item: item["score"] = 0.0
                    cleaned_exps.append(item)
                elif isinstance(item, str):
                    cleaned_exps.append({"text": item, "score": 0.0})
            
            # 长度强制对齐
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

        except Exception as e:
            print(f"[Exp Controller Error] {e}")
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
            df = pd.read_parquet(parquet_path)
            records = df.to_dict(orient="records")
            actual_size = min(sample_size, len(records))
            self.dataset = random.sample(records, actual_size)
            print(f"[TF-GRPO] Randomly sampled {actual_size} problems.")

        for ep in range(epochs):
            print(f"\n{'='*20} Epoch {ep+1}/{epochs} {'='*20}")
            
            for idx, item in enumerate(tqdm(self.dataset)):
                q, gold = item["prompt"][0]["content"], item["reward_model"]["ground_truth"]

                # 1. 提取经验
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
                        "Please refer to the following historical experiences (with confidence scores) to solve this problem:\n"
                        f"{exp_str}\n\n"
                        "Think step by step."
                    )
                else:
                    system_content = "You are a math problem solver. Think step by step."

                messages = [{"role": "system", "content": system_content}, 
                            {"role": "user", "content": f"Problem: {q}\n"}]
                
                # 2. Rollout
                outputs = []
                for _ in range(self.group_size):
                    outputs.append(self.call_llm(messages, temperature=0.7))

                # 3. Advantage
                rewards = [self.compute_composite_reward(o, gold) for o in outputs]
                advantages = self.compute_advantages(rewards)

                # 4. Summarize & Update
                summary_result = self.batch_summarize(q, outputs)
                indiv_sums = summary_result.get("individual_summaries", [])
                overall_sum = summary_result.get("overall_summary", "")

                if len(indiv_sums) < self.group_size:
                    indiv_sums.extend([""] * (self.group_size - len(indiv_sums)))

                observations = []
                for i in range(self.group_size):
                    observations.append({"summary": indiv_sums[i], "advantage": advantages[i]})
                
                avg_adv = statistics.mean(advantages) if advantages else 0.0
                observations.append({"summary": f"[Global] {overall_sum}", "advantage": avg_adv})

                updated_experiences = self.exp_controller(q, current_experiences, observations)
                self.experience_bank[idx] = updated_experiences
                print(f"problem{idx+1}:{updated_experiences}")

                if (idx + 1) % 10 == 0:
                    print(f" Problem {idx+1}: AvgReward={avg_adv:.2f}")

            # === 保存逻辑更新：格式化为指定结构 ===
            print(f"Saving Experience Bank for Epoch {ep+1}...")
            save_data = []
            for idx, exps in self.experience_bank.items():
                # 获取问题文本 (防御性编程：确保索引存在)
                if idx < len(self.dataset):
                    prob_text = self.dataset[idx]["prompt"][0]["content"]
                else:
                    prob_text = "Unknown Problem"

                # 转换 experiences 列表为带 attempt 键的格式
                formatted_exps = []
                for i, e in enumerate(exps):
                    # 判断是 attempt 还是 overall summary (最后一条)
                    if i == len(exps) - 1:
                        key_name = "overall summary"
                    else:
                        key_name = f"attempt{i+1}"
                    
                    formatted_exps.append({
                        key_name: e.get("text", ""),
                        "score": e.get("score", 0.0)
                    })
                
                save_data.append({
                    "problem": prob_text,
                    "experiences": formatted_exps
                })

            with open(f"experience_bank_epoch_{ep+1}.json", "w", encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)


