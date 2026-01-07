# libero_vqa/eval_libero_vqa.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


# =========================
# Handler interface
# =========================
class VQAHandler:
    name = "base"

    def can_handle(self, qid: str) -> bool:
        raise NotImplementedError

    def build_user_text(self, q: Dict[str, Any]) -> str:
        raise NotImplementedError

    def normalize_gt(self, q: Dict[str, Any]) -> str:
        raise NotImplementedError

    def normalize_pred(self, pred_text: str, q: Dict[str, Any]) -> str:
        raise NotImplementedError

    def score(self, gt: str, pred: str, q: Dict[str, Any]) -> bool:
        return gt == pred


# =========================
# Measurement handler
# =========================
class MeasurementHandler(VQAHandler):
    name = "measurement"

    def can_handle(self, qid: str) -> bool:
        return qid in {"M1", "M2", "M3"}

    def build_user_text(self, q: Dict[str, Any]) -> str:
        # plain instruction text (messages already provide role/image)
        return q["question"] + "\nAnswer concisely."

    def normalize_gt(self, q: Dict[str, Any]) -> str:
        return q["answer"].strip()

    def normalize_pred(self, pred_text: str, q: Dict[str, Any]) -> str:
        txt = pred_text.strip()

        qid = q["id"]
        if qid in {"M1", "M2"}:
            if re.search(r"\byes\b", txt, re.I):
                return "Yes"
            if re.search(r"\bno\b", txt, re.I):
                return "No"
            return ""

        if qid == "M3":
            m = re.search(r"\bA\s*([1-5])\b", txt, re.I)
            if m:
                return f"A{m.group(1)}"
            return ""

        return txt


# =========================
# Registry
# =========================
class HandlerRegistry:
    def __init__(self, handlers: List[VQAHandler]):
        self.handlers = handlers

    def get(self, qid: str) -> VQAHandler | None:
        for h in self.handlers:
            if h.can_handle(qid):
                return h
        return None


# =========================
# Pipeline helpers (Qwen2.5-VL style)
# =========================
def build_messages(pil_image: Image.Image, user_text: str, image_key: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", image_key: pil_image},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def run_pipeline(
    pipe,
    pil_image: Image.Image,
    user_text: str,
    *,
    image_key: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        do_sample=(temperature > 0),
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    # try {"image": PIL}
    try:
        messages = build_messages(pil_image, user_text, image_key=image_key)
        out = pipe(text=messages, **gen_kwargs)
        return out[0]["generated_text"]
    except Exception as e:
        raise RuntimeError(f"pipeline failed: {repr(e)}")


# =========================
# Main evaluation
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--image_key", type=str, default="agentview_image")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--save_json", type=str, default=None)
    args = p.parse_args()

    # dtype
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(args.dtype, None)

    # pipeline (official Qwen2.5-VL style)
    pipe = pipeline(
        task="image-text-to-text",
        model=args.model_id,
        trust_remote_code=True,
        device=0 if args.device.startswith("cuda") else -1,
        dtype=torch_dtype,
    )

    ds = load_from_disk(args.dataset_dir)

    registry = HandlerRegistry(
        handlers=[
            MeasurementHandler(),
        ]
    )

    total = 0
    correct = 0
    by_qid = {}

    results_dump = []

    for row in tqdm(ds, desc="Evaluating"):
        img_arr = row[args.image_key]
        if img_arr is None:
            continue

        if not isinstance(img_arr, np.ndarray):
            img_arr = np.asarray(img_arr, dtype=np.uint8)

        pil_image = Image.fromarray(img_arr)

        for q in row["questions"]:
            qid = q["id"]
            handler = registry.get(qid)
            if handler is None:
                continue

            user_text = handler.build_user_text(q)

            pred_text = run_pipeline(
                pipe,
                pil_image,
                user_text,
                image_key="image",  # official key for Qwen2.5-VL
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            gt = handler.normalize_gt(q)
            pred = handler.normalize_pred(pred_text, q)
            ok = handler.score(gt, pred, q)

            total += 1
            correct += int(ok)
            by_qid.setdefault(qid, {"total": 0, "correct": 0})
            by_qid[qid]["total"] += 1
            by_qid[qid]["correct"] += int(ok)

            if args.save_json:
                results_dump.append(
                    {
                        "suite": row.get("suite"),
                        "task": row.get("task"),
                        "episode_id": row.get("episode_id"),
                        "step_idx": row.get("step_idx"),
                        "qid": qid,
                        "question": q["question"],
                        "gt": gt,
                        "pred": pred,
                        "raw_pred": pred_text,
                        "correct": ok,
                    }
                )

    summary = {
        "accuracy": correct / max(total, 1),
        "total": total,
        "by_qid": {k: v["correct"] / max(v["total"], 1) for k, v in by_qid.items()},
    }

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(
                {"summary": summary, "results": results_dump},
                f,
                indent=2,
            )
        print("Saved detailed results to:", args.save_json)


if __name__ == "__main__":
    main()
