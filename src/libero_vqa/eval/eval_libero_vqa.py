# libero_vqa/eval_libero_vqa.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


# =========================
# Image utils
# =========================
def _ensure_uint8_hwc(img: Any) -> np.ndarray:
    """
    Accepts:
      - np.ndarray (H,W,C) or (C,H,W) or list-like
    Returns:
      uint8 HWC (H,W,3)
    """
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape={arr.shape}")

    # CHW -> HWC
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # dtype -> uint8
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(np.max(arr)) if arr.size else 0.0
            if mx <= 1.5:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 1ch -> 3ch
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got shape={arr.shape}")

    return arr


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
        return q["question"] + "\nAnswer concisely."

    def normalize_gt(self, q: Dict[str, Any]) -> str:
        return str(q["answer"]).strip()

    def normalize_pred(self, pred_text: str, q: Dict[str, Any]) -> str:
        txt = (pred_text or "").strip()

        qid = str(q.get("id", ""))
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

    def get(self, qid: str) -> Optional[VQAHandler]:
        for h in self.handlers:
            if h.can_handle(qid):
                return h
        return None


# =========================
# Inference backends
# =========================
class InferBackend:
    name = "base"

    def infer(self, pil_image: Image.Image, user_text: str) -> str:
        raise NotImplementedError


@dataclass
class PipelineBackend(InferBackend):
    """
    Uses transformers pipeline('image-text-to-text') in the Qwen2.5-VL style.
    """

    pipe: Any
    image_key: str = "image"
    max_new_tokens: int = 32
    temperature: float = 0.0
    name: str = "pipeline"

    def _build_messages(self, pil_image: Image.Image, user_text: str):
        # Qwen2.5-VL expects {"type":"image","image": PIL} NOT {"url":...} for local PIL.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", self.image_key: pil_image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

    def infer(self, pil_image: Image.Image, user_text: str) -> str:
        gen_kwargs = dict(
            max_new_tokens=int(self.max_new_tokens),
            return_full_text=False,
            do_sample=(float(self.temperature) > 0.0),
        )
        if float(self.temperature) > 0.0:
            gen_kwargs["temperature"] = float(self.temperature)

        messages = self._build_messages(pil_image, user_text)
        out = self.pipe(text=messages, **gen_kwargs)
        # pipeline outputs like: [{"generated_text": "..."}]
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return str(out[0].get("generated_text", "")).strip()
        return str(out).strip()


@dataclass
class CustomFnBackend(InferBackend):
    """
    Wraps your pre-written infer_fn(model_id, pil_image, user_text, **kwargs) -> str
    """

    infer_fn: Callable[..., str]
    model_id: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    name: str = "custom_fn"

    def infer(self, pil_image: Image.Image, user_text: str) -> str:
        return str(
            self.infer_fn(
                model_id=self.model_id,
                pil_image=pil_image,
                user_text=user_text,
                max_new_tokens=int(self.max_new_tokens),
                temperature=float(self.temperature),
            )
        ).strip()


def _try_make_pipeline_backend(
    *,
    model_id: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
) -> Optional[PipelineBackend]:
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, None)

    dev = 0 if str(device).startswith("cuda") else -1

    try:
        pipe = pipeline(
            task="image-text-to-text",
            model=model_id,
            trust_remote_code=True,
            device=dev,
            dtype=torch_dtype,
        )
        backend = PipelineBackend(
            pipe=pipe,
            image_key="image",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # smoke test (fast, catches "messages format" / unsupported pipeline models)
        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        _ = backend.infer(dummy, "Answer with exactly: Yes")
        return backend
    except Exception:
        return None


def _try_make_custom_backend(
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
) -> Optional[CustomFnBackend]:
    """
    Looks for your pre-written infer registry, but doesn't hard-require it.

    Supported layout:
      libero_vqa.infer_registry.get_infer_fn(model_id) -> Callable
    """
    import libero_vqa.eval.infer_registry as ir

    infer_fn = None

    try:
        infer_fn = ir.get_infer_fn(model_id)
    except Exception:
        infer_fn = None

    if infer_fn is None:
        return None

    return CustomFnBackend(
        infer_fn=infer_fn,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def get_backend(
    *,
    model_id: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
) -> InferBackend:
    """
    Priority:
      1) pipeline backend if it can be created AND passes smoke test
      2) custom infer_fn backend (your mapping)
    """
    pb = _try_make_pipeline_backend(
        model_id=model_id,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    if pb is not None:
        print(f"[Infer] Using pipeline backend for: {model_id}")
        return pb

    cb = _try_make_custom_backend(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    if cb is not None:
        print(f"[Infer] Using custom infer_fn backend for: {model_id}")
        return cb

    raise RuntimeError(
        "No available inference backend.\n"
        f"- pipeline('image-text-to-text') failed for model_id={model_id}\n"
        "- and no custom infer_fn registry entry was found.\n"
        "Create libero_vqa/infer_registry.py with get_infer_fn(model_id) or INFER_FNS dict."
    )


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

    backend = get_backend(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    ds = load_from_disk(args.dataset_dir)

    registry = HandlerRegistry(
        handlers=[
            MeasurementHandler(),
        ]
    )

    total = 0
    correct = 0
    by_qid: Dict[str, Dict[str, int]] = {}
    results_dump: List[Dict[str, Any]] = []

    for row in tqdm(ds, desc=f"Evaluating ({backend.name})"):
        if args.image_key not in row or row[args.image_key] is None:
            continue

        img_arr = _ensure_uint8_hwc(row[args.image_key])
        pil_image = Image.fromarray(img_arr)

        questions = row.get("questions", [])
        if not isinstance(questions, list):
            continue

        for q in questions:
            if not isinstance(q, dict):
                continue
            qid = str(q.get("id", ""))
            handler = registry.get(qid)
            if handler is None:
                continue

            user_text = handler.build_user_text(q)

            try:
                pred_text = backend.infer(pil_image, user_text)
            except Exception as e:
                pred_text = f"[INFER_ERROR] {repr(e)}"

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
                        "question": q.get("question", ""),
                        "gt": gt,
                        "pred": pred,
                        "raw_pred": pred_text,
                        "correct": ok,
                    }
                )

    summary = {
        "model_id": args.model_id,
        "backend": backend.name,
        "accuracy": correct / max(total, 1),
        "total": total,
        "by_qid": {k: (v["correct"] / max(v["total"], 1)) for k, v in by_qid.items()},
        "by_qid_counts": by_qid,
    }

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(
            json.dumps(
                {"summary": summary, "results": results_dump},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("Saved detailed results to:", args.save_json)


if __name__ == "__main__":
    main()
