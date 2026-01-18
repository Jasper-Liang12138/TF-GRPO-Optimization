import re
import os
import torch
import numpy as np
from typing import List
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class TF_GRPO:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        group_size=5,
        max_experiences=100,
        max_new_tokens=512,
        # Embedding 模型改为本地轻量级模型，用于计算相似度
        embedding_model_name: str = "all-MiniLM-L6-v2", 
    ):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
        self.group_size = group_size
        self.max_experiences = max_experiences
        self.max_new_tokens = max_new_tokens
        
        self.experience_bank = []
        self.exp_embeddings = []
        self.exp_embeddings_matrix = None

        # 初始化 Embedding 模型 (运行在 CPU 或 GPU 均可，显存占用极小)
        print(f"[TF-GRPO] Loading Embedding Model: {embedding_model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer(embedding_model_name, device=device)

    # ===================== 摘要推理 (改为调用 GPT-3.5) =====================
    def summarize_reasoning(self, reasoning: str) -> str:
        """
        使用 GPT-3.5 对推理过程进行摘要
        """
        reasoning = reasoning.strip()
        if not reasoning:
            return reasoning

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Summarize the following reasoning process concisely."},
                    {"role": "user", "content": reasoning}
                ],
                max_tokens=150,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Warning] Summarize failed: {e}")
            return reasoning

    # ===================== 计算文本 embedding (使用 SentenceTransformer) =====================
    def embed_func(self, text: str):
        # SentenceTransformer 直接返回 numpy array
        return self.embed_model.encode(text)

    # ===================== Prompt 构建 =====================
    def build_prompt(self, question: str) -> List[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": "You are a math problem solver. Solve the problem step by step and give the final answer."
            },
            {
                "role": "user",
                "content": f"Problem: {question}\n"
            }
        ]
        return messages

    def build_prompt_inference(self, question: str) -> List[dict[str, str]]:
        # 修改为 Chat 格式
        messages = [
            {
                "role": "system",
                "content": "You are a math problem solver. Please refer to experience to solve the problems."
            },
            {
                "role": "user",
                "content": f"Problem: {question}\n"
            }
        ]
        return messages

    def build_aqua_prompt(self, question_with_choices: str) -> List[dict[str, str]]:
        content = (
            "Solve the following multiple-choice math problem referring to experience.\n"
            "Pick exactly one option from {A, B, C, D, E}.\n"
            "Do NOT output the option text.\n"
            "Your output must end with exactly one line:\n"
            "Answer: <A/B/C/D/E>\n\n"
            f"Problem:\n{question_with_choices}\n"
        )
        return [{"role": "user", "content": content}]

    # ===================== 相似度检索 (保持逻辑不变) =====================
    def extract_similar_experiences(self, question: str, top_k) -> List[str]:
        if not self.experience_bank or self.exp_embeddings_matrix is None:
            return self.experience_bank[-top_k:]

        q_emb = self.embed_func(question).reshape(1, -1)
        sims = cosine_similarity(q_emb, self.exp_embeddings_matrix)[0]

        top_k = min(top_k, len(self.experience_bank))
        if top_k <= 0:
            return []

        top_indices = sims.argsort()[-top_k:][::-1]
        similarity_threshold = 0.1
        top_experiences = [
            self.experience_bank[i]
            for i in top_indices
            if sims[i] >= similarity_threshold
        ]
        return top_experiences

    # ===================== 生成 (调用 OpenAI API) =====================
    def batch_group_generate(self, prompts: List[dict[str, str]]) -> List[str]:
        """
        API 不支持像本地模型那样的 batch tensor 输入，
        这里输入如果是 List[List[Dict]] (多条不同的prompt)，需要循环调用。
        如果输入是 List[Dict] (单条 prompt)，直接调用。
        """
        # 兼容性处理：如果输入是单个 prompt (List[Dict])
        if isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], dict):
            single_prompt = prompts
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=single_prompt,
                    temperature=0.7,
                    max_tokens=self.max_new_tokens,
                    n=1 # 每次生成一条
                )
                return [response.choices[0].message.content]
            except Exception as e:
                print(f"API Error: {e}")
                return [""]
        
        # 暂时不支持一次传入多个不同的 prompt list，如果需要可在这里加循环
        return ["Error: Invalid prompt format"]

    # ===================== 辅助函数 (保持不变) =====================
    def extract_answer(self, text: str) -> str:
        patterns = [r"Answer\s*[:\-]?\s*([^\n$]+)"]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip().replace("$", "").replace("\\", "").strip()
                if re.match(r"^[\d\.\-/]+$", ans): return ans
        for l in reversed(text.splitlines()):
            l_clean = l.replace("$","").replace("\\","").strip()
            if re.match(r"^[\d\.\-/]+$", l_clean): return l_clean
        return ""

    def reasoning_quality(self, reasoning: str) -> float:
        if not reasoning: return 0.0
        text = reasoning.strip()
        score = 0.0
        math_ops = "+-*/=^"
        n_math = sum(text.count(op) for op in math_ops)
        score += min(n_math, 20) * 0.2
        if "step" in text.lower(): score += 1.0
        if len(text) > 50: score += 1.0
        return score

    def reward(self, output: str, gold: str) -> float:
        answer = self.extract_answer(output)
        answer_reward = float(answer == str(gold))
        reasoning_reward = self.reasoning_quality(output)
        alpha, beta = 0.7, 0.3
        return alpha * answer_reward + beta * reasoning_reward

    def select_best(self, outputs: List[str], gold: str) -> str:
        if not outputs: return None
        scores = [self.reward(o, gold) for o in outputs]
        if max(scores) > 0:
            return outputs[scores.index(max(scores))]
        return max(outputs, key=self.reasoning_quality)

    def extract_reasoning_from_output(self, prompt, output_text: str) -> str:
        # API 返回的内容就是 Assistant 的回复，不需要像本地模型那样去除 prompt
        reasoning = output_text
        # 清洗逻辑
        reasoning = re.sub(r"```.*?```", "", reasoning, flags=re.DOTALL)
        reasoning = re.sub(r"\[experience\s*\d+\].*?(?=\n\n|answer|final answer|$)", "", reasoning, flags=re.DOTALL | re.IGNORECASE)
        return reasoning.strip()

    def update_experience(self, problem: str, reasoning: str, gold: str):
        reasoning = reasoning.strip()
        if not reasoning: return
        
        # 使用 GPT-3.5 进行摘要
        summarized_reasoning = self.summarize_reasoning(reasoning)

        experience = (
            f"Problem:\n{problem}\n\n"
            f"Experience:\n{summarized_reasoning}\n\n"
            f"Gold Answer:\n{gold}"
        )

        self.experience_bank.append(experience)
        
        # 更新 embedding
        emb = self.embed_func(experience)
        self.exp_embeddings.append(emb)

        if len(self.experience_bank) > self.max_experiences:
            self.experience_bank.pop(0)
            self.exp_embeddings.pop(0)

        if self.exp_embeddings:
            self.exp_embeddings_matrix = np.vstack(self.exp_embeddings)
        else:
            self.exp_embeddings_matrix = None

    def load_experience_bank(self, experience_data: list):
        if not isinstance(experience_data, list): raise ValueError("Must be list")
        self.experience_bank = experience_data[-self.max_experiences:]
        self.exp_embeddings = []
        print(f"[TF-GRPO] Loading {len(self.experience_bank)} experiences...")
        for exp in tqdm(self.experience_bank):
            emb = self.embed_func(exp)
            self.exp_embeddings.append(emb)
        if self.exp_embeddings:
            self.exp_embeddings_matrix = np.vstack(self.exp_embeddings)

    def build_experience_from_dapo_epochs(self, parquet_path, sample_size=100, epochs=3):
        import pandas as pd
        
        df = pd.read_parquet(parquet_path)
        records = df.to_dict(orient="records")

        for ep in range(epochs):
            print(f"\n[TF-GRPO] Epoch {ep+1}/{epochs}")
            samples = random.sample(records, min(sample_size, len(records)))
            
            for sample_idx, item in enumerate(tqdm(samples)):
                q = item["prompt"][0]["content"]
                # 兼容 DAPO 数据集的 ground truth 路径
                a = item["reward_model"]["ground_truth"]

                base_prompt = self.build_prompt(q)
                top_experiences = self.extract_similar_experiences(q, self.group_size)
                
                outputs = []
                # 策略：如果无经验，直接跑一次；如果有经验，结合每一条经验分别跑一次
                if not top_experiences:
                    out = self.batch_group_generate(base_prompt)[0]
                    outputs.append(out)
                else:
                    for exp_idx, exp in enumerate(top_experiences, 1):
                        import copy
                        p = copy.deepcopy(base_prompt)
                        # 将经验插入到 User 消息中
                        p[1]["content"] += (
                            "\nUse the following reasoning experience internally:\n"
                            f"[Experience {exp_idx}]\n{exp}\n"
                        )
                        out = self.batch_group_generate(p)[0]
                        outputs.append(out)

                best = self.select_best(outputs, a)
                if best:
                    self.update_experience(q, best, a)

            # 保存中间结果
            with open(f"experience_bank_ep{ep}.json", "w") as f:
                json.dump(self.experience_bank, f, indent=2)