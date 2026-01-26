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
        
        # 经验库：Key为题目在 dataset 中的索引，Value为经验列表
        self.experience_bank: Dict[int, List[str]] = {}
        
        # 新增：持久化的数据集，确保多轮 Epoch 使用同一批随机采样的数据
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

    def check_correctness(self, output: str, gold: str) -> float:
        pred = self.extract_answer(output)
        if not pred: return 0.0
        
        try:
            # 尝试数值转换比较
            # 处理分数、逗号等，例如 "1,000" -> 1000
            def clean_num(s):
                s = s.replace(",", "")
                if "/" in s: return float(Fraction(s))
                return float(s)
                
            return 1.0 if abs(clean_num(pred) - clean_num(str(gold))) < 1e-6 else 0.0
        except:
            # 如果无法转数字，回退到字符串比较
            return 1.0 if pred.strip() == str(gold).strip() else 0.0

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        if not rewards: return []
        if len(rewards) == 1: return [0.0]
        mean_r = statistics.mean(rewards)
        std_r = statistics.stdev(rewards)
        if std_r == 0:
            return [0.0] * len(rewards)
        return [(r - mean_r) / (std_r + 1e-8) for r in rewards]

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
            """
            一次性对 G 条轨迹进行分别总结 + 整体总结。
            返回格式:
            {
                "individual_summaries": ["summary_for_trace_1", "summary_for_trace_2", ...],
                "overall_summary": "summary_for_all"
            }
            """
            # 构建输入文本，标记清楚每一条轨迹
            traces_text = ""
            for i, out in enumerate(outputs):
                # 为了防止 prompt 过长，可以适当截断每条轨迹（例如前2000字符）
                traces_text += f"--- [Attempt {i+1}] ---\n{out}\n\n"

            prompt = (
                f"Problem: {question}\n\n"
                f"Here are {len(outputs)} different attempts to solve this problem:\n"
                f"{traces_text}\n"
                "--- TASK ---\n"
                "1. Analyze each attempt INDIVIDUALLY: Summarize its method and identify key errors or success factors (2-3 sentences each).For example, 'In attempt 1, the agent...'.\n"
                "2. Analyze ALL attempts COLLECTIVELY: Provide a brief overall summary and precautions for this problem type.\n\n"
                "--- OUTPUT FORMAT ---\n"
                "Return a strictly valid JSON object with exactly two keys:\n"
                "{\n"
                '  "individual_summaries": ["summary 1", "summary 2", ...],\n'
                '  "overall_summary": "your overall insight string"\n'
                "}"
            )
            
            messages = [{"role": "user", "content": prompt}]
            
            # 强制 JSON 模式
            res = self.call_llm(messages, temperature=0.7, json_mode=True)
            
            try:
                data = json.loads(res)
                # 简单的校验，确保返回列表长度对得上
                if not isinstance(data.get("individual_summaries"), list):
                    data["individual_summaries"] = ["Error parsing summary"] * len(outputs)
                return data
            except json.JSONDecodeError:
                print("[Batch Summarize] JSON parsing failed.")
                return {
                    "individual_summaries": ["Summary failed"] * len(outputs),
                    "overall_summary": "Summary failed"
                }

    def exp_controller(self, question: str, old_exps: List[str], observations: List[Dict]) -> List[str]:
        e_text = json.dumps(old_exps, indent=1) if old_exps else "[] (No prior experience)"
        obs_text = ""
        for i, obs in enumerate(observations):
            if i != len(observations) - 1:
                obs_text += (
                    f"[Attempt {i+1}]\n"
                    f"Advantage Score: {obs['advantage']:.4f}\n"
                    f"Summary: {obs['summary']}\n\n"
                )
            else:  # 最后一个是整体总结
                obs_text += (
                    f"[Overall Summary]\n"
                    f"Advantage Score: {obs['advantage']:.4f}\n"
                    f"Summary: {obs['summary']}\n\n"
                )
        system_prompt = "You are a meta-learning optimizer for math reasoning."
        user_prompt = (
            f"Current Problem: {question}\n\n"
            f"=== OLD EXPERIENCE BANK (E) ===\n{e_text}\n\n"
            f"=== NEW OBSERVATIONS (from current G={self.group_size} rollouts) ===\n{obs_text}\n\n"
            "=== INSTRUCTIONS ===\n"
            "Use the Advantage Scores (A_i) to refine the Experience Bank(Experience Bank's attempts'number should be equal to G):\n"
            "1. **KEEP**: Retain old experiences that align with high-advantage attempts.\n"
            "2. **MODIFY**: Correct an old experience if a high-advantage attempt shows a better way.\n"
            "3. **DELETE**: Remove experiences that led to low-advantage outcomes.\n"
            "4. **ADD**: Extract new insights from high-advantage attempts.\n\n"
            "5. **IMPROVEMENT**: Revise the Overall Summary.\n\n" 
            "Output the updated list of experiences as a JSON object: {\"experiences\": [\"exp1\", \"exp2\"]}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        res = self.call_llm(messages, temperature=0.2, json_mode=True)
        try:
            data = json.loads(res)
            new_exps = data.get("experiences", [])
            return [str(e) for e in new_exps if isinstance(e, str)]
        except:
            return old_exps

    # ===================== 4. 主训练循环 =====================

    def train_loop(self, parquet_path: str, epochs=3, sample_size=100):
        import pandas as pd
        
        # ====== 修改部分：随机采样与 Dataset 持久化逻辑 ======
        if self.dataset:
            print(f"[TF-GRPO] Using existing dataset with {len(self.dataset)} samples.")
        else:
            print(f"[TF-GRPO] Loading dataset from {parquet_path}...")
            df = pd.read_parquet(parquet_path)
            records = df.to_dict(orient="records")
            
            # 随机抽取 sample_size 道题
            # 如果原始数据不够，则全量使用
            actual_size = min(sample_size, len(records))
            self.dataset = random.sample(records, actual_size)
            print(f"[TF-GRPO] Randomly sampled {actual_size} problems. Dataset initialized.")
        # =================================================

        for ep in range(epochs):
            print(f"\n{'='*20} Epoch {ep+1}/{epochs} {'='*20}")
            
            # 遍历 dataset
            for idx, item in enumerate(tqdm(self.dataset)):
                
                # 数据解析
                q, gold = item["prompt"][0]["content"], item["reward_model"]["ground_truth"]

                # 1. 提取经验 (按 idx 顺序提取)
                current_experiences = self.experience_bank.get(idx, [])
                
                # 2. Rollout
                if current_experiences:
                    exp_str = "\n".join([f"{i+1}. {e}" for i, e in enumerate(current_experiences)])
                    system_content = (
                        "You are a math problem solver. "
                        "Please refer to the following historical experiences to solve this problem:\n"
                        f"{exp_str}\n\n"
                        "Think step by step."
                    )
                else:
                    system_content = "You are a math problem solver. Think step by step."

                messages = [{"role": "system", 
                             "content": system_content}, 
                            {"role": "user", 
                             "content": f"Problem: {q}\n"}]
                
                outputs = []
                for _ in range(self.group_size):
                    outputs.append(self.call_llm(messages, temperature=0.7))

                # 3. Advantage
                rewards = [self.check_correctness(o, gold) for o in outputs]
                advantages = self.compute_advantages(rewards)

                # 4. Batch Summarization (1次 API 调用)
                
                # 只要有不一致的Reward，或者为了积累经验，就进行总结
                # 这里建议始终进行，或者根据需求加 if 判断
                summary_result = self.batch_summarize(q, outputs)
                
                individual_summaries = summary_result.get("individual_summaries", [])
                overall_summary = summary_result.get("overall_summary", "")

                # 补齐数据（防止模型返回的列表长度不够）
                if len(individual_summaries) < self.group_size:
                    individual_summaries.extend([""] * (self.group_size - len(individual_summaries)))

                observations = []
                
                # A. 填入个体总结
                for i in range(self.group_size):
                    observations.append({
                        "summary": individual_summaries[i], 
                        "advantage": advantages[i]
                    })
                
                # B. 填入整体总结
                # 整体总结的 advantage 给平均分
                avg_advantage = statistics.mean(advantages) if advantages else 0.0
                observations.append({
                    "summary": f"[Global Insight] {overall_summary}", 
                    "advantage": avg_advantage
                })
                # 调用 Controller 更新
                updated_experiences = self.exp_controller(q, current_experiences, observations)
                self.experience_bank[idx] = updated_experiences
                print(f" Problem {idx}: Updated Experience Bank: {updated_experiences}")

                # Debug print
                if (idx + 1) % 10 == 0:
                    print(f" Problem {idx+1}: AvgReward={avg_advantage}")

            # 保存 Epoch 结果
            with open(f"experience_bank_epoch_{ep+1}.json", "w") as f:
                dump_data = [{"index": k, "experiences": v} for k, v in self.experience_bank.items()]
                json.dump(dump_data, f, indent=2)