import re
import random
from typing import List
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TF_GRPO:
    def __init__(
        self,
        model,
        tokenizer,
        group_size=5,
        max_experiences=100,
        max_new_tokens=512,
        summarizer_model_name: str = "t5-base",
        summarizer_max_new_tokens: int = 128,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_experiences = max_experiences
        self.max_new_tokens = max_new_tokens
        self.experience_bank = []
        self.exp_embeddings = []
        # 预缓存好的经验库 embedding 矩阵，用于快速相似度计算，避免每次查询都 vstack
        self.exp_embeddings_matrix = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # ===================== 初始化 T5 摘要模型 =====================
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(
            summarizer_model_name
        )
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
            summarizer_model_name
        ).to(self.model.device)
        self.summarizer_max_new_tokens = summarizer_max_new_tokens

    # ===================== 摘要推理 =====================
    @torch.no_grad()
    def summarize_reasoning(self, reasoning: str) -> str:
        """
        使用 t5-base 对推理过程进行摘要，减少冗余后再写入 experience bank
        """
        reasoning = reasoning.strip()
        if not reasoning:
            return reasoning

        # T5 通常使用 "summarize:" 前缀
        input_text = "summarize: " + reasoning
        inputs = self.summarizer_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.summarizer_model.device)

        outputs = self.summarizer_model.generate(
            **inputs,
            max_new_tokens=self.summarizer_max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )
        summary = self.summarizer_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()
        return summary if summary else reasoning
    # ===================== 计算文本 embedding =====================
    def embed_func(self, text: str):
        # tokenization
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # 获取隐藏层表征
        with torch.no_grad():
            # 尝试获取 base_model 以确保能获取 hidden states
            base = None
            if hasattr(self.model, 'get_base_model'):
                try:
                    base = self.model.get_base_model()
                except Exception:
                    base = None
            if base is None:
                # fallback to common attributes
                base = getattr(self.model, 'base_model', self.model)
            
            # 通过 forward pass 获取隐藏层输出
            outputs = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 获取最后一层的隐藏状态
            last_hidden = getattr(outputs, 'last_hidden_state', None)
            if last_hidden is None:
                hidden_states = getattr(outputs, 'hidden_states', None)
                if hidden_states:
                    last_hidden = hidden_states[-1]  # [1, seq_len, hidden_size]
                else:
                    raise RuntimeError('模型未返回 last_hidden_state 或 hidden_states，无法计算隐藏层表征')
            
            # 使用 attention_mask 进行 mean pooling（只对有效 token 取平均）
            if attention_mask is not None:
                # 扩展 attention_mask 维度用于广播: [1, seq_len] -> [1, seq_len, 1]
                mask_expanded = attention_mask.unsqueeze(-1).float()
                # 将 padding 位置的 hidden state 置为 0
                masked_hidden = last_hidden * mask_expanded
                # 对有效 token 取平均
                sum_hidden = masked_hidden.sum(dim=1)  # [1, hidden_size]
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [1, 1]
                sent_emb = sum_hidden / seq_lengths  # [1, hidden_size]
            else:
                # 如果没有 attention_mask，直接对所有 token 取平均
                sent_emb = last_hidden.mean(dim=1)  # [1, hidden_size]
        
        return sent_emb.cpu().numpy()[0]  # 返回 1D np.array

    # ===================== pθ(y | x, E) =====================
    def build_prompt(self, question: str) -> List[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math problem solver. "
                    "Solve the problem step by step and give the final answer."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Problem:{question}\n"
                )
            }
        ]
        
        return messages

    def build_prompt_inference(self, question: str) -> str:
        prompt = (
            f"Please refer to experience to solve the problems:{question}.\n"
        )    
        return prompt

    def build_aqua_prompt(question_with_choices: str) -> str:
        """
        AQuA 是多选题，题干里通常包含 Answer Choices。
        强制模型只输出选项字母，避免输出空的 Answer: 或输出数值。
        """
        return (
            "Solve the following multiple-choice math problem referring to experience.\n"
            "Pick exactly one option from {A, B, C, D, E}.\n"
            "Answer: <A/B/C/D/E>\n\n"
            f"Problem:\n{question_with_choices}\n"
    )
    # ===================== rollout × G =====================
    def extract_similar_experiences(self, question: str, top_k) -> List[str]:
        """
        从经验库中抽取与当前问题最相似的 top_k 条经验。
        复杂度主要来自与所有经验的相似度计算，这里通过预缓存矩阵避免每次调用都 vstack。
        """
        # 经验库为空或 embedding 矩阵尚未构建时，fallback 为最近的几条
        if not self.experience_bank or self.exp_embeddings_matrix is None:
            return self.experience_bank[-top_k:]

        # 计算问题 embedding
        q_emb = self.embed_func(question).reshape(1, -1)

        # 计算与经验库的相似度 (1, N)，这里直接使用预缓存矩阵，避免每次 vstack
        sims = cosine_similarity(q_emb, self.exp_embeddings_matrix)[0]

        # 取 top_k
        top_k = min(top_k, len(self.experience_bank))
        if top_k <= 0:
            return []

        top_indices = sims.argsort()[-top_k:][::-1]

        # 相似度阈值（可根据分布再调参）
        similarity_threshold = 0.1
        top_experiences = [
            self.experience_bank[i]
            for i in top_indices
            if sims[i] >= similarity_threshold
        ]
        return top_experiences
        
    # ===================== generate =====================
    @torch.no_grad()
    def batch_group_generate(self, prompts: List[dict[str, str]]) -> List[str]:
        inputs = self.tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            max_length=1024
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=4,            # 束搜索数量（GPT-OSS-120B可设4-8，平衡质量/速度）
            num_beam_groups=2,      # 可选：分组束搜索，提升推理多样性（不破坏确定性）
            length_penalty=1.0,     # 长度惩罚，避免输出过短
            max_new_tokens=self.max_new_tokens,  # 仅通过 max_new_tokens 约束生成长度
            num_return_sequences=1,  # 每个输入只生成一条
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,  # 遇到 EOS token 时停止
        )

        # 直接解码输出
        return [
            self.tokenizer.decode(o[inputs["input_ids"].shape[-1]:], skip_special_tokens=True) 
            for o in outputs
        ]

    # ===================== GRPO =====================
    def extract_answer(self, text: str) -> str:
        patterns = [r"Answer\s*[:\-]?\s*([^\n$]+)"]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip()
                ans = ans.replace("$", "").replace("\\", "").strip()
                # 匹配数字或分数
                if re.match(r"^[\d\.\-/]+$", ans):
                    return ans
        # fallback: 最后一行纯数字或分数
        for l in reversed(text.splitlines()):
            l_clean = l.replace("$","").replace("\\","").strip()
            if re.match(r"^[\d\.\-/]+$", l_clean):
                return l_clean
        return ""  # 如果没有数字，返回空



    def reasoning_quality(self, reasoning: str) -> float:
        """
        针对数学推理的启发式打分：
        - 鼓励：合适长度、包含数学符号/数字、分步结构、逻辑连接词
        - 惩罚：过短/过长、模板/系统提示、HTML 垃圾、大量重复
        """
        if not reasoning:
            return 0.0

        text = reasoning.strip()
        if not text:
            return 0.0

        lower = text.lower()
        score = 0.0
        n_tokens = len(text.split())

        # ===== 2. 数学信号：运算符、等号、幂、LaTeX =====
        math_ops = "+-*/=^"
        n_math = sum(text.count(op) for op in math_ops)
        score += min(n_math, 20) * 0.2  # 上限防止极端情况
        score += text.count("\\") * 0.1  # LaTeX 轻微加分

        # 数字出现次数
        n_numbers = len(re.findall(r"\d", text))
        score += min(n_numbers, 50) * 0.05

        # ===== 3. 步骤结构和答案信号 =====
        step_patterns = re.findall(r"(step\s*\d+|第\s*\d+\s*步)", lower)
        score += len(step_patterns) * 0.5

        if ("answer:" in lower) or ("final answer" in lower) or ("答案" in lower):
            score += 1.0

        # ===== 4. 逻辑连接词（中英混合） =====
        logic_words = [
            # 英文
            "let", "assume", "then", "therefore", "hence",
            "because", "so", "implies", "we have", "consider",
            "case", "thus", "suppose",
            # 常见中文数学逻辑词
            "因为", "所以", "因此", "故", "可得", "得到", "综上",
            "令", "假设", "不妨设", "从而",
        ]
        for w in logic_words:
            cnt = lower.count(w)
            if cnt > 0:
                score += min(cnt, 10) * 0.3

        # ===== 5. 重复度惩罚：token 去重比率过低视为水文 / 重复 =====
        if n_tokens > 0:
            uniq_ratio = len(set(text.split())) / n_tokens
            if uniq_ratio < 0.3:
                score -= 2.0
            elif uniq_ratio < 0.2:
                score -= 4.0

        # ===== 6. 垃圾 / 模板内容惩罚 =====
        bad_snippets = [
            "<p>",
            "you are a reasoning assistant",
            "you are a helpful assistant",
            "the following are useful reasoning experiences",
            "as an ai language model",
        ]
        for s in bad_snippets:
            if s.lower() in lower:
                score -= 10.0

        return score

    def reward(self, output: str, gold: str) -> float:
        answer = self.extract_answer(output)
        answer_reward = float(answer == gold)  # EM
        reasoning_reward = self.reasoning_quality(output)
        # 可以调整权重
        alpha, beta = 0.7, 0.3
        return alpha * answer_reward + beta * reasoning_reward

    def select_best(self, outputs: List[str], gold: str) -> str:
        # 检查 outputs 是否为空
        if not outputs:
            return None
        
        # 先过滤明显无效的
        # valid_outputs = [o for o in outputs if self.is_valid_reasoning(o)]

        #if not valid_outputs:
            #return None  # ❗直接丢弃这个样本
        # 拆分 output
        #reasonings = self.split_output_into_reasonings(outputs)
        
        scores = [self.reward(o, gold) for o in outputs]

        if max(scores) > 0:
            return outputs[scores.index(max(scores))]

        # fallback：不是最长，而是「最像推理的」, 只看推理部分，不考虑EM
        return max(outputs, key=self.reasoning_quality)


    def extract_reasoning_from_output(
            self,
            prompt: List[dict[str, str]],
            output: List[dict[str, str]],
        ) -> str:
            """
            从 chat-style output(List[Dict]) 中提取模型生成的 reasoning
            - prompt: 输入给模型的 messages（仅用于语义对齐，不做字符串替换）
            - output: prompt + assistant 的完整 messages
            """
            # -------------------------------------------------
            # 0) 只提取 assistant 的新生成内容
            # -------------------------------------------------
            assistant_contents = [
                msg["content"]
                for msg in output
                if msg.get("role") == "assistant" and msg.get("content")
            ]

            if not assistant_contents:
                return ""

            reasoning = "\n".join(assistant_contents)

            # -------------------------------------------------
            # 1) 删除代码块（LaTeX / markdown / python）
            # -------------------------------------------------
            reasoning = re.sub(r"```.*?```", "", reasoning, flags=re.DOTALL)

            reasoning = re.sub(
                r"\\begin\{code\}.*?\\end\{code\}",
                "",
                reasoning,
                flags=re.DOTALL | re.IGNORECASE,
            )

            reasoning = re.sub(
                r"^(import |from |def |class |print\(|if __name__|return |    return ).*?(?=\n\n|\n[^\s]|\Z)",
                "",
                reasoning,
                flags=re.MULTILINE | re.DOTALL,
            )

            code_keywords = [
                r"^import .*$",
                r"^from .* import .*$",
                r"^def \w+\(.*\):.*$",
                r"^class \w+.*:.*$",
                r"^print\(.*\)$",
                r"^    return .*$",
            ]
            for pattern in code_keywords:
                reasoning = re.sub(pattern, "", reasoning, flags=re.MULTILINE | re.IGNORECASE)

            # -------------------------------------------------
            # 2) 删除系统指令 / prompt 痕迹
            # -------------------------------------------------
            system_prompts = [
                r"please solve the math problems?.*?\n",
                r"please solve the math problem and give an?.*?\n",
                r"please refer to experience to solve the problems?.*?\n",
                r"solve the following problem step by step.*?\n",
                r"solve the following math problem step by step.*?\n",
                r"the last line of your response should be.*?\n",
                r"remember to put your answer.*?\n",
                r"use the following reasoning experiences.*?\n",
                r"use at most \d+ tokens.*?\n",
                r"only output the final answer.*?\n",
                r"don't repeat the same reasoning.*?\n",
                r"output should be:.*?\n",
                r"your output must end with.*?\n",
                r"problem\s*:.*?\n",
            ]
            for pattern in system_prompts:
                reasoning = re.sub(pattern, "", reasoning, flags=re.IGNORECASE | re.DOTALL)

            # -------------------------------------------------
            # 3) 删除 Experience 块
            # -------------------------------------------------
            reasoning = re.sub(
                r"\[experience\s*\d+\].*?(?=\n\n|answer|final answer|$)",
                "",
                reasoning,
                flags=re.DOTALL | re.IGNORECASE,
            )

            reasoning = re.sub(
                r"\[experience\s*\d+\][^\n]*\n?",
                "",
                reasoning,
                flags=re.IGNORECASE,
            )

            # -------------------------------------------------
            # 4) 删除 HTML / LaTeX 标记
            # -------------------------------------------------
            reasoning = re.sub(r"<[^>]+>", " ", reasoning)
            reasoning = re.sub(r"\$[^\$]+\$", "", reasoning)
            reasoning = re.sub(r"\\[a-zA-Z]+\{[^\}]+\}", "", reasoning)

            # -------------------------------------------------
            # 5) 删除最终答案行（只保留 reasoning）
            # -------------------------------------------------
            reasoning = re.sub(
                r"(?:answer|final answer)\s*[:\-]?\s*[^\n]*(?:\n|$)",
                "",
                reasoning,
                flags=re.IGNORECASE,
            )

            # -------------------------------------------------
            # 6) 行级去重 + 删除问题文本
            # -------------------------------------------------
            lines = reasoning.split("\n")
            seen = set()
            unique_lines = []

            question_pattern = re.compile(
                r"^(how many|what is|find|calculate|solve|determine).*\?$",
                re.IGNORECASE,
            )

            for line in lines:
                s = line.strip()
                l = s.lower()

                if not s:
                    continue
                if question_pattern.match(l):
                    continue
                if l in seen:
                    continue
                if len(s) < 3:
                    continue

                seen.add(l)
                unique_lines.append(s)

            reasoning = "\n".join(unique_lines)

            # -------------------------------------------------
            # 7) 清理多余空白
            # -------------------------------------------------
            reasoning = re.sub(r"\n\s*\n\s*\n+", "\n\n", reasoning)
            reasoning = re.sub(r"[ \t]+", " ", reasoning)

            return reasoning.strip()


    def update_experience(self, problem: str, reasoning: str, gold: str):
        # 去掉空或无效内容
        reasoning = reasoning.strip()
        if not reasoning:
            return

        # 先用 T5 对推理进行摘要，减少冗余
        summarized_reasoning = self.summarize_reasoning(reasoning)

        # ========== 3. 统一写入格式 ==========
        experience = (
            f"Problem:\n{problem}\n\n"
            f"Experience:\n{summarized_reasoning}\n\n"
            f"Gold Answer:\n{gold}"
        )

        self.experience_bank.append(experience)
        
        # ========== 同步更新 embeddings ==========
        emb = self.embed_func(experience)
        self.exp_embeddings.append(emb)

        # ========== 4. 维护容量 ==========
        if len(self.experience_bank) > self.max_experiences:
            self.experience_bank.pop(0)
            self.exp_embeddings.pop(0)

        # ========== 5. 维护预缓存矩阵 ==========
        if self.exp_embeddings:
            self.exp_embeddings_matrix = np.vstack(self.exp_embeddings)
        else:
            self.exp_embeddings_matrix = None

    def is_valid_reasoning(self, text: str) -> bool:
        if not text:
            return False

        text = text.strip()
        bad_patterns = [
            "You are a reasoning assistant",
            "The following are useful reasoning experiences",
        ]
         # 只有：包含这两句之一 + 文本很短，才判不合规
        if len(text) < 100:
            for p in bad_patterns:
                if p in text:
                    return False
        return True

    # ===================== 从外部加载经验库 =====================
    def load_experience_bank(self, experience_data: list):
        """
        Load experience bank from external list of reasoning strings.

        Args:
            experience_data (list[str]): List of reasoning examples.
        """
        if not isinstance(experience_data, list):
            raise ValueError("experience_data must be a list of strings.")
        # 仅保留最近 max_experiences 条
        self.experience_bank = experience_data[-self.max_experiences:]

        # 同时计算 embedding
        self.exp_embeddings = []
        self.exp_embeddings_matrix = None
        print(f"[TF-GRPO] Loading {len(self.experience_bank)} experiences...")
        all_embs = []
        for exp in tqdm(self.experience_bank, desc="Embedding experiences"):
            emb = self.embed_func(exp)
            self.exp_embeddings.append(emb)
            all_embs.append(emb)

        # 构建一次性矩阵，后续相似度查询直接使用
        if all_embs:
            self.exp_embeddings_matrix = np.vstack(all_embs)
        else:
            self.exp_embeddings_matrix = None
        print("[TF-GRPO] Experience bank loaded.")

    # ===================== Build E from DAPO with Epochs =====================
    @torch.no_grad()
    def build_experience_from_dapo_epochs(
        self,
        parquet_path: str,
        sample_size=100,
        epochs=3,
    ):
        print_samples=100
        df = pd.read_parquet(parquet_path)
        records = df.to_dict(orient="records")

        for ep in range(epochs):
            print(f"\n[TF-GRPO] Epoch {ep+1}/{epochs}")

            samples = random.sample(
                records, min(sample_size, len(records))
            )

            for sample_idx, item in enumerate(tqdm(samples)):
                q, a = item["prompt"][0]["content"], item["reward_model"]["ground_truth"]

                base_prompt = self.build_prompt(q)
                top_experiences = self.extract_similar_experiences(q, self.group_size)
                outputs = []  # 用来收集每个经验生成的 output
                
                if not top_experiences:
                    # 如果没有相似经验，直接生成
                    out = self.batch_group_generate(base_prompt)[0]
                    reasoning = self.extract_reasoning_from_output(base_prompt, out)
                    if reasoning:
                        outputs.append(reasoning)
                else:
                    # 为每个经验单独构建 prompt 并生成
                    for exp_idx, exp in enumerate(top_experiences, 1):
                        # 每次从 base_prompt 开始，添加当前经验
                        prompt_with_exp = base_prompt
                        prompt_with_exp[1]["content"] += "Use the following reasoning experiences internally to help your solution, " \
                                          "but do NOT copy them verbatim into your answer."
                        prompt_with_exp[1]["content"] += f"[Experience {exp_idx}]\n{exp}\n\n"
                        
                        out = self.batch_group_generate(prompt_with_exp)[0]
                        reasoning = self.extract_reasoning_from_output(prompt_with_exp, out)
                        if reasoning:    
                            outputs.append(reasoning)

                best = self.select_best(outputs, a)
                if best is None:
                    continue   # 跳过该样本，不写入 bank

                self.update_experience(q, best, a)

                # 打印验证
                if sample_idx < print_samples:
                    print(f"\nSample {sample_idx+1}")
                    print("Question:", q)
                    #print("outputs:", outputs)
                    print("Selected reasoning:", best)
                    print("extracted answer:", self.extract_answer(best))
                    print("Gold answer:", a)

            experience_bank = self.experience_bank
            # 阶段性保存到文件
            with open(f"RLHF/experience_bank{ep}.json", "w", encoding="utf-8") as f:
                json.dump(experience_bank, f, ensure_ascii=False, indent=2)

        experience_bank = self.experience_bank
        # 保存到文件
        with open("RLHF/experience_bank.json", "w", encoding="utf-8") as f:
            json.dump(experience_bank, f, ensure_ascii=False, indent=2)

        
