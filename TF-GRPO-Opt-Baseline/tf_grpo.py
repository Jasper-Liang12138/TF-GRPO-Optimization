"""
tf_grpo.py — TF-GRPO 核心类（本地 Qwen2.5-Math-7B-Instruct 版）

逻辑完全对齐原版 tf_grpo_deepseek.py，仅将 OpenAI API 替换为本地模型推理。

原版流程（每道题每 epoch）：
  1. 从经验库取当前经验 → 注入 system prompt
  2. Rollout G 次
  3. 计算 composite reward（正确性 + 格式 + 过程质量）
  4. 计算 advantages（z-score）
  5. batch_summarize：用 LLM 对每条 rollout 生成摘要 + 整体摘要
  6. exp_controller：用 LLM 基于 advantage + 摘要更新经验库
  7. 保存经验库到 JSON

本地化改动：
  - call_llm → model.generate()，支持 chat_template
  - json_mode → 在 prompt 中显式要求 JSON，并做鲁棒解析
  - NPU 初始化放在最顶部（import torch_npu）
"""
from __future__ import annotations

import json
import os
import re
import random
import statistics
from fractions import Fraction
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 昇腾 NPU 初始化 ────────────────────────────────────────────────────────
try:
    import torch_npu                               # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    _NPU_AVAILABLE = True
except ImportError:
    _NPU_AVAILABLE = False


# ── JSON 鲁棒解析（本地模型不保证严格 JSON 输出） ──────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """
    尝试从模型输出中提取 JSON，兼容以下常见格式：
      1. 纯 JSON 字符串
      2. ```json ... ``` 代码块
      3. 夹杂说明文字，但含完整 {...} 块
    失败返回 None。
    """
    text = text.strip()

    # 1. 剥离 markdown 代码块标记（```json 或 ```）
    code_block = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if code_block:
        text = code_block.group(1).strip()

    def _try_parse(s: str) -> Optional[dict]:
        """尝试解析，若失败则去掉尾逗号后再试一次。"""
        try:
            return json.loads(s)
        except Exception:
            pass
        # 去掉对象/数组末尾的尾逗号（Qwen 常见输出问题）
        cleaned = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

    # 2. 直接解析
    result = _try_parse(text)
    if result is not None:
        return result

    # 3. 提取第一个完整 {...} 块（允许前后有说明文字）
    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if start == -1:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start != -1:
                candidate = text[start: i + 1]
                result = _try_parse(candidate)
                if result is not None:
                    return result
                start = -1  # 重置，继续找下一个块
    return None


# ─────────────────────────────────────────────────────────────────────────────

class TF_GRPO:
    """
    原版 TF-GRPO 算法的本地模型复刻版（Baseline）。

    与原版 API 版的主要对应关系：
      call_llm(messages, temperature, json_mode)
        原版：client.chat.completions.create(...)
        本版：tokenizer.apply_chat_template + model.generate()
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        group_size: int = 5,
        max_new_tokens: int = 2048,
        device: str = "npu:0",
        torch_dtype: str = "float16",
    ) -> None:
        self.model_name = model_name
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.device = torch.device(device)
        self.dtype = torch.float16 if torch_dtype == "float16" else torch.bfloat16

        # 经验库：key=题目索引，value=[{"text":str,"score":float},...]
        self.experience_bank: Dict[int, List[Dict[str, Any]]] = {}
        self.dataset: List[Dict] = []

        # ── 加载本地模型 ────────────────────────────────────────────────
        print(f"[TF-GRPO] 加载 tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[TF-GRPO] 加载模型: {model_name}  dtype={self.dtype}  device={device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print("[TF-GRPO] 模型加载完成。")

    # ===================== 1. 基础工具 & 奖励 =====================

    def extract_answer(self, text: str) -> str:
        """从模型输出中提取最终答案（与原版逻辑一致）。"""
        patterns = [
            r"\\boxed\{([^{}]+)\}",
            r"Answer\s*[:\-]?\s*([^\n$]+)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().replace("$", "").replace("\\", "").strip()
        for line in reversed(text.splitlines()):
            clean = line.replace("$", "").replace("\\", "").strip()
            if re.match(r"^-?[\d\.\-/]+$", clean):
                return clean
        return ""

    def check_format(self, text: str) -> float:
        """格式奖励：包含 \\boxed{} → 1.0，否则 0.0。"""
        return 1.0 if r"\boxed{" in text else 0.0

    def check_process_quality(self, text: str) -> float:
        """过程质量启发式打分（与原版一致）。"""
        if len(text) < 50:
            return 0.0
        score = 0.0
        if "$" in text or "\\" in text:
            score += 0.5
        step_words = ["step", "first", "therefore", "since", "because"]
        if any(w in text.lower() for w in step_words):
            score += 0.5
        return min(1.0, score)

    def check_correctness(self, output: str, gold: str) -> float:
        """数值/字符串正确性判断（与原版一致）。"""
        pred = self.extract_answer(output)
        if not pred:
            return 0.0
        try:
            def clean_num(s: str) -> float:
                s = s.replace(",", "")
                if "/" in s:
                    return float(Fraction(s))
                return float(s)
            return 1.0 if abs(clean_num(pred) - clean_num(str(gold))) < 1e-6 else 0.0
        except Exception:
            return 1.0 if pred.strip() == str(gold).strip() else 0.0

    def compute_composite_reward(self, output: str, gold: str) -> float:
        """
        复合奖励（与原版权重完全一致）：
          r = 1.0*correctness + 0.1*format + 0.1*process_quality
        """
        r_acc = self.check_correctness(output, gold)
        r_fmt = self.check_format(output)
        r_proc = self.check_process_quality(output)
        return r_acc * 1.0 + r_fmt * 0.1 + r_proc * 0.1

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """GRPO z-score 优势（与原版一致）。"""
        if not rewards:
            return []
        if len(rewards) == 1:
            return [0.0]
        mean_r = statistics.mean(rewards)
        std_r = statistics.stdev(rewards)
        if std_r == 0:
            return [0.0] * len(rewards)
        return [(r - mean_r) / (std_r + 1e-8) for r in rewards]

    # ===================== 2. 本地 LLM 调用封装 =====================

    def call_llm(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """
        本地推理替代原版 API 调用。

        json_mode=True 时在 prompt 末尾追加 JSON 格式约束指令，
        并截短 max_new_tokens 避免过度生成。
        """
        # 若 json_mode，在最后一条 user 消息末尾追加格式提醒
        if json_mode:
            messages = list(messages)  # 浅拷贝，不改原列表
            last = messages[-1].copy()
            last["content"] = last["content"] + (
                "\n\nIMPORTANT: Output valid JSON ONLY. "
                "No explanation, no markdown code block, no extra text. "
                "Start your response directly with { and end with }."
            )
            messages[-1] = last
            # JSON 调用用低温，减少随机性
            temperature = min(temperature, 0.2)

        # 应用 chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
        ).to(self.device)

        max_tok = 2048 if json_mode else self.max_new_tokens

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的 token
        new_ids = output_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ===================== 3. 核心算子（与原版逻辑一致） =====================

    def batch_summarize(
        self, question: str, outputs: List[str]
    ) -> Dict[str, Any]:
        """
        对 G 条 rollout 分别生成摘要 + 整体摘要。
        原版：call_llm(..., json_mode=True)
        本版：同样调用 call_llm + _extract_json 鲁棒解析。
        """
        # 截断过长输出以适应上下文
        MAX_OUT_CHARS = 800
        MAX_Q_CHARS = 300
        traces_text = ""
        for i, out in enumerate(outputs):
            traces_text += f"--- [Attempt {i+1}] ---\n{out[:MAX_OUT_CHARS]}...\n\n"

        q_short = question[:MAX_Q_CHARS] + ("..." if len(question) > MAX_Q_CHARS else "")
        prompt = (
            f"Problem: {q_short}\n\n"
            f"Here are {len(outputs)} different attempts to solve this problem:\n"
            f"{traces_text}\n"
            "--- TASK ---\n"
            "1. Analyze each attempt INDIVIDUALLY: Summarize method and errors using about 2-4 sentences.\n"
            "2. Analyze ALL attempts COLLECTIVELY: Provide overall insight.\n\n"
            "--- OUTPUT FORMAT ---\n"
            "Return JSON ONLY (no other text):\n"
            "{\n"
            '  "individual_summaries": ["summary 1", "summary 2", ...],\n'
            '  "overall_summary": "overall insight"\n'
            "}"
        )
        messages = [{"role": "user", "content": prompt}]
        res = self.call_llm(messages, temperature=0.7, json_mode=True)

        data = _extract_json(res)
        if data and isinstance(data.get("individual_summaries"), list):
            return data
        return {
            "individual_summaries": ["Summary failed"] * len(outputs),
            "overall_summary": "Summary failed",
        }

    def exp_controller(
        self,
        question: str,
        old_exps: List[Dict[str, Any]],
        observations: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        基于 advantage + 摘要更新经验库（与原版逻辑一致）。
        原版：call_llm(..., json_mode=True) → 解析 JSON
        本版：同样，但用 _extract_json 鲁棒解析。
        """
        target_count = len(observations)
        default_item: Dict[str, Any] = {
            "text": "Think clearly and step by step.",
            "score": 0.0,
        }

        # 格式化旧经验（与原版一致）
        if old_exps:
            formatted_old = []
            for i, exp in enumerate(old_exps):
                key = "overall summary" if i == len(old_exps) - 1 else f"attempt{i+1}"
                formatted_old.append({key: exp.get("text", ""), "score": exp.get("score", 0.0)})
            e_text = json.dumps(formatted_old, indent=1)
        else:
            e_text = "[] (No prior experience)"

        obs_text = ""
        for i, obs in enumerate(observations):
            prefix = f"[Attempt {i+1}]" if i != len(observations) - 1 else "[Overall Summary]"
            obs_text += (
                f"{prefix}\n"
                f"Advantage Score: {obs['advantage']:.4f}\n"
                f"Summary: {obs['summary']}\n\n"
            )

        MAX_Q_CHARS = 300
        q_short = question[:MAX_Q_CHARS] + ("..." if len(question) > MAX_Q_CHARS else "")
        system_prompt = "You are a meta-learning optimizer for math reasoning."
        user_prompt = (
            f"Current Problem: {q_short}\n\n"
            f"=== OLD EXPERIENCE BANK (With Advantage Scores) ===\n{e_text}\n\n"
            f"=== NEW OBSERVATIONS (G={self.group_size} rollouts + 1 global) ===\n{obs_text}\n\n"
            "=== INSTRUCTIONS ===\n"
            "Refine the Experience Bank based on the new Advantage Scores.\n"
            f"IMPORTANT: You MUST output exactly {target_count} experience items.\n"
            "1. **KEEP/MODIFY**: Retain or improve experiences from high-advantage attempts.\n"
            "2. **DELETE**: Remove experiences linked to negative advantages.\n"
            "3. **ADD**: Extract new insights from high-advantage attempts.\n"
            "4. **ASSIGN SCORE**: For each experience, assign a 'score' equal to the Advantage of the attempt it was derived from.\n\n"
            "Output strictly valid JSON ONLY (no other text):\n"
            "{\n"
            "  \"experiences\": [\n"
            "    {\"text\": \"Use method X...\", \"score\": 0.85},\n"
            "    {\"text\": \"Avoid error Y...\", \"score\": -0.5}\n"
            "  ]\n"
            "}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        res = self.call_llm(messages, temperature=0.7, json_mode=True)
        data = _extract_json(res)
        # 若首次解析失败，重试一次
        if data is None:
            res = self.call_llm(messages, temperature=0.7, json_mode=True)
            data = _extract_json(res)

        try:
            if data is None:
                raise ValueError("JSON 解析失败")

            new_exps = data.get("experiences", [])

            # 格式清洗（与原版一致）
            cleaned: List[Dict] = []
            for item in new_exps:
                if isinstance(item, dict) and "text" in item:
                    if "score" not in item:
                        item["score"] = 0.0
                    cleaned.append(item)
                elif isinstance(item, str):
                    cleaned.append({"text": item, "score": 0.0})

            # 长度强制对齐（与原版一致）
            cur = len(cleaned)
            if cur == target_count:
                return cleaned
            elif cur > target_count:
                return cleaned[:target_count]
            else:
                missing = target_count - cur
                if len(old_exps) >= target_count:
                    cleaned.extend(old_exps[cur:target_count])
                else:
                    filler = cleaned[-1] if cleaned else default_item
                    cleaned.extend([filler.copy() for _ in range(missing)])
                return cleaned

        except Exception as e:
            print(f"[Exp Controller Error] {e}")
            # 规则兜底：直接把 advantage>0 的 observation summary 存为经验，
            # 比全部 fallback 成默认值更有实际意义
            return self._rule_based_experiences(observations, old_exps, target_count, default_item)

    # ===================== 4. 主训练循环 =====================

    def train_loop(
        self,
        parquet_path: Optional[str] = None,
        epochs: int = 3,
        sample_size: int = 100,
        output_dir: str = ".",
    ) -> None:
        """
        主训练循环（与原版逻辑完全一致，新增 output_dir 参数）。

        数据源支持：
          (a) parquet 文件（原版格式）：prompt[0]["content"] + reward_model["ground_truth"]
          (b) HuggingFace GSM8K：question + answer
          (c) 使用 self.dataset（已提前加载）
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── 数据加载 ────────────────────────────────────────────────────
        if self.dataset:
            print(f"[TF-GRPO] 使用已有数据集，共 {len(self.dataset)} 条。")
        elif parquet_path and os.path.exists(parquet_path):
            print(f"[TF-GRPO] 从 Parquet 加载: {parquet_path}")
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            records = df.to_dict(orient="records")
            actual_size = min(sample_size, len(records))
            self.dataset = random.sample(records, actual_size)
            print(f"[TF-GRPO] 随机采样 {actual_size} 条。")
        else:
            print("[TF-GRPO] 未指定 parquet_path，从 HuggingFace 加载 GSM8K ...")
            from datasets import load_dataset
            raw = load_dataset("openai/gsm8k", "main", split="train")
            records = list(raw)
            actual_size = min(sample_size, len(records))
            self.dataset = random.sample(records, actual_size)
            print(f"[TF-GRPO] 随机采样 {actual_size} 条 GSM8K。")

        # ── 训练循环 ────────────────────────────────────────────────────
        for ep in range(epochs):
            print(f"\n{'='*20} Epoch {ep+1}/{epochs} {'='*20}")

            for idx, item in enumerate(tqdm(self.dataset)):
                # 兼容两种数据格式（与原版逻辑一致）
                # pandas 读 parquet 后 prompt 列为 numpy.ndarray，直接索引即可
                if "reward_model" in item:
                    # DAPO-Math 格式
                    q = str(item["prompt"][0]["content"])
                    gold = str(item["reward_model"]["ground_truth"])
                else:
                    # GSM8K 格式
                    q = item.get("question", item.get("problem", ""))
                    gold = self._extract_gsm8k_gold(item.get("answer", ""))

                # 首题打印题目前 120 字，确认读取正确
                if idx == 0 and ep == 0:
                    print(f"[DEBUG] q[:120] = {q[:120]!r}")
                    print(f"[DEBUG] gold = {gold!r}")

                # 1. 提取当前经验 → 构造 messages（始终重新构造，与原版一致）
                current_exps = self.experience_bank.get(idx, [])

                if current_exps:
                    exp_lines = []
                    for i, e in enumerate(current_exps):
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

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Problem: {q}\n"},
                ]

                # 2. Rollout G 次（与原版一致）
                outputs: List[str] = []
                for _ in range(self.group_size):
                    outputs.append(self.call_llm(messages, temperature=0.7))

                # 3. Advantage（与原版一致）
                rewards = [self.compute_composite_reward(o, str(gold)) for o in outputs]
                advantages = self.compute_advantages(rewards)

                # 4. Summarize & Update（与原版一致）
                summary_result = self.batch_summarize(q, outputs)
                indiv_sums = summary_result.get("individual_summaries", [])
                overall_sum = summary_result.get("overall_summary", "")

                if len(indiv_sums) < self.group_size:
                    indiv_sums.extend([""] * (self.group_size - len(indiv_sums)))

                # batch_summarize 失败时用 rollout 原文（截断）替代，
                # 保证 observations 始终有实质内容传入 exp_controller
                FAILED = {"", "Summary failed"}
                for i in range(self.group_size):
                    if indiv_sums[i].strip() in FAILED:
                        indiv_sums[i] = outputs[i][:300] if outputs[i] else ""

                observations: List[Dict] = []
                for i in range(self.group_size):
                    observations.append({"summary": indiv_sums[i], "advantage": advantages[i]})

                avg_adv = statistics.mean(advantages) if advantages else 0.0
                observations.append(
                    {"summary": f"[Global] {overall_sum}", "advantage": avg_adv}
                )

                updated_exps = self.exp_controller(q, current_exps, observations)
                self.experience_bank[idx] = updated_exps
                print(f"problem{idx+1}: {updated_exps}")

                if (idx + 1) % 10 == 0:
                    avg_r = sum(rewards) / len(rewards) if rewards else 0.0
                    print(f" Problem {idx+1}: AvgReward={avg_r:.4f}")

            # ── 保存经验库（与原版格式完全一致） ──────────────────────
            print(f"Saving Experience Bank for Epoch {ep+1}...")
            save_data = []
            for idx, exps in self.experience_bank.items():
                if idx < len(self.dataset):
                    item = self.dataset[idx]
                    if "reward_model" in item:
                        prob_text = str(item["prompt"][0]["content"])
                    else:
                        prob_text = item.get("question", item.get("problem", "Unknown"))
                else:
                    prob_text = "Unknown Problem"

                formatted_exps = []
                for i, e in enumerate(exps):
                    key_name = "overall summary" if i == len(exps) - 1 else f"attempt{i+1}"
                    formatted_exps.append({
                        key_name: e.get("text", ""),
                        "score": e.get("score", 0.0),
                    })

                save_data.append({"problem": prob_text, "experiences": formatted_exps})

            save_path = os.path.join(output_dir, f"experience_bank_epoch_{ep+1}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"[TF-GRPO] 经验库已保存 → {save_path}")

    # ── 工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rule_based_experiences(
        observations: List[Dict],
        old_exps: List[Dict],
        target_count: int,
        default_item: Dict,
    ) -> List[Dict]:
        """
        LLM JSON 生成失败时的规则兜底：
        按 advantage 降序取 summary，不论正负（全为 0 时也照常存入），
        只跳过空文本和明确失败标记。
        """
        INVALID = {"", "Summary failed", "[Global] Summary failed"}
        sorted_obs = sorted(observations, key=lambda x: x["advantage"], reverse=True)
        result: List[Dict] = []
        for obs in sorted_obs:
            text = obs["summary"].strip()
            if text not in INVALID:
                result.append({"text": text[:300], "score": round(obs["advantage"], 4)})
            if len(result) >= target_count:
                break
        # 用旧经验补齐
        for exp in old_exps:
            if len(result) >= target_count:
                break
            result.append(exp)
        # 最后用默认值补足
        while len(result) < target_count:
            result.append(default_item.copy())
        return result[:target_count]

    @staticmethod
    def _extract_gsm8k_gold(answer_text: str) -> str:
        """从 GSM8K answer 字段提取 #### 后的数字。"""
        m = re.search(r"####\s*([\-\d,\.]+)", answer_text)
        if m:
            return m.group(1).replace(",", "").strip()
        return answer_text.strip()
