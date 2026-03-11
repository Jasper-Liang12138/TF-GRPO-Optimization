from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any


def load_json(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_stats(path: str) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, list):
        return {"exists": False, "count": 0, "correct": 0, "accuracy": None}
    count = len(data)
    correct = sum(1 for item in data if item.get("flag") is True)
    return {
        "exists": True,
        "count": count,
        "correct": correct,
        "accuracy": (correct / count) if count else None,
    }


def experience_stats(path: str) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, list):
        return {"exists": False, "cases": 0, "avg_items": 0.0}
    item_counts = [len(item.get("experiences", [])) for item in data]
    avg_items = sum(item_counts) / len(item_counts) if item_counts else 0.0
    return {"exists": True, "cases": len(data), "avg_items": avg_items}


def format_accuracy(stats: dict[str, Any]) -> str:
    acc = stats.get("accuracy")
    if acc is None:
        return "N/A"
    return f"{acc:.4f} ({stats['correct']}/{stats['count']})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--group_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--exp_bank_path", required=True)
    parser.add_argument("--tfgrpo_result", required=True)
    parser.add_argument("--zeroshot_result", required=True)
    parser.add_argument("--master_log", required=True)
    parser.add_argument("--build_log", required=True)
    parser.add_argument("--tfgrpo_log", required=True)
    parser.add_argument("--zeroshot_log", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    exp_stats = experience_stats(args.exp_bank_path)
    tf_stats = result_stats(args.tfgrpo_result)
    zs_stats = result_stats(args.zeroshot_result)
    delta = None
    if tf_stats["accuracy"] is not None and zs_stats["accuracy"] is not None:
        delta = tf_stats["accuracy"] - zs_stats["accuracy"]

    lines = [
        f"# TF-GRPO Baseline Report ({args.run_tag})",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model: `{args.model_path}`",
        f"- Device: `{args.device}`",
        f"- Train data: `{args.train_data}`",
        f"- Test data: `{args.test_data}`",
        f"- Sample size: `{args.sample_size}`",
        f"- Group size: `{args.group_size}`",
        f"- Epochs: `{args.epochs}`",
        f"- Max new tokens: `{args.max_new_tokens}`",
        "",
        "## Outputs",
        "",
        f"- Experience bank: `{args.exp_bank_path}`",
        f"- TF-GRPO result: `{args.tfgrpo_result}`",
        f"- Zero-shot result: `{args.zeroshot_result}`",
        f"- Master log: `{args.master_log}`",
        f"- Build log: `{args.build_log}`",
        f"- TF-GRPO eval log: `{args.tfgrpo_log}`",
        f"- Zero-shot eval log: `{args.zeroshot_log}`",
        "",
        "## Experience Bank",
        "",
        f"- Generated: `{exp_stats['exists']}`",
        f"- Cases: `{exp_stats['cases']}`",
        f"- Average items per case: `{exp_stats['avg_items']:.2f}`",
        "",
        "## Evaluation",
        "",
        f"- TF-GRPO accuracy: {format_accuracy(tf_stats)}",
        f"- Zero-shot accuracy: {format_accuracy(zs_stats)}",
        f"- Accuracy delta (TF-GRPO - Zero-shot): {delta:.4f}" if delta is not None else "- Accuracy delta (TF-GRPO - Zero-shot): N/A",
        "",
        "## Notes",
        "",
        "- Experience-controller JSON generation is still imperfect on 7B, so some experiences may come from rule-based fallback compaction.",
        "- This report is generated from saved artifacts and logs after the run.",
    ]

    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
