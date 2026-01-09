# libero_vqa/eval_openai_batch.py
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from datasets import load_from_disk
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


# -------------------------
# utilities
# -------------------------
def _ensure_uint8_hwc(img: Any) -> np.ndarray:
    """
    Accepts:
      - np.ndarray (H,W,C) or (C,H,W)
      - list-like
    Returns uint8 HWC.
    """
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape={arr.shape}")

    # if CHW -> HWC
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # common: already (H,W,3)
    if arr.dtype != np.uint8:
        # if float in [0,1] or [0,255], robustly map
        if np.issubdtype(arr.dtype, np.floating):
            # assume 0..1 if max<=1.5
            mx = float(np.max(arr)) if arr.size else 0.0
            if mx <= 1.5:
                arr = np.clip(arr * 255.0, 0, 255)
            else:
                arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        else:
            # int types (your error was int64)
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    # enforce 3 channels
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got shape={arr.shape}")
    return arr


def _img_to_base64_jpeg(img_arr: Any, quality: int = 90) -> str:
    arr = _ensure_uint8_hwc(img_arr)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_output_text_from_responses_body(body: Dict[str, Any]) -> str:
    """
    Try best-effort extraction from Responses API payload.

    Priority:
      1) body["output_text"] (if present)
      2) body["output"][...]["content"][...]["text"] / output_text
      3) body["choices"][0]["message"]["content"] (chat.completions-like fallback)
      4) stringify
    """
    if not isinstance(body, dict):
        return str(body)

    # 1) direct convenience field
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()

    # 2) canonical responses output array
    out = body.get("output")
    if isinstance(out, list):
        texts: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    # common shapes
                    if part.get("type") in ("output_text", "text") and isinstance(
                        part.get("text"), str
                    ):
                        texts.append(part["text"])
                    # sometimes: {"type":"output_text","text":"..."} already covered
            # sometimes: {"type":"message","role":"assistant","content":[...]}
        if texts:
            return "\n".join(texts).strip()

    # 3) chat.completions-like fallback (some wrappers may return this)
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"].strip()

    # 4) fallback
    return json.dumps(body, ensure_ascii=False)


def _normalize_pred_answer(text: str) -> str:
    """
    Normalize model output to compare with your dataset's target answers.
    Handles:
      - "Yes"/"No"
      - "A1".."A5"
      - Extra punctuation/whitespace
    """
    t = (text or "").strip()

    # common patterns: "Answer: Yes" / "Answer: A2"
    low = t.lower()
    if "answer" in low:
        # keep substring after "answer"
        idx = low.find("answer")
        t = t[idx:]
        # split by ':' if exists
        if ":" in t:
            t = t.split(":", 1)[1].strip()

    # take first token-like chunk
    # (but keep A2 exactly)
    t = t.strip().replace(".", " ").replace(",", " ").replace("\n", " ")
    toks = [x for x in t.split(" ") if x]
    if not toks:
        return ""

    first = toks[0].strip()

    # normalize yes/no
    f_low = first.lower()
    if f_low in ("yes", "y", "true"):
        return "Yes"
    if f_low in ("no", "n", "false"):
        return "No"

    # normalize A1..A5
    # tolerate "a2)" "A2)" etc
    f = first.upper().strip(") ]}")
    if f.startswith("A") and len(f) >= 2 and f[1].isdigit():
        return "A" + f[1]
    return first


# -------------------------
# dataset flattening
# -------------------------
@dataclass
class FlatExample:
    custom_id: str
    row_idx: int
    q_idx: int
    question_id: str
    question: str
    gold: str
    image_b64: str
    meta: Dict[str, Any]


def iter_flat_examples(
    ds,
    *,
    image_key: str,
    max_rows: Optional[int] = None,
    jpeg_quality: int = 90,
) -> Iterable[FlatExample]:
    n = len(ds) if max_rows is None else min(len(ds), int(max_rows))
    for i in tqdm(range(n), desc="Preparing requests", unit="row"):
        row = ds[i]

        if image_key not in row:
            raise KeyError(
                f"image_key='{image_key}' not in dataset row keys={list(row.keys())}"
            )

        img_b64 = _img_to_base64_jpeg(row[image_key], quality=jpeg_quality)

        questions = row.get("questions")
        if not isinstance(questions, list):
            continue

        for qi, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            qid = str(q.get("id", f"Q{qi}"))
            qtext = str(q.get("question", ""))
            gold = str(q.get("answer", ""))

            # custom_id: unique in batch
            custom_id = f"row{i:07d}_q{qi:02d}_{qid}"

            meta = {
                "suite": row.get("suite"),
                "task": row.get("task"),
                "episode_id": row.get("episode_id"),
                "step_idx": row.get("step_idx"),
                "question_id": qid,
                # objects/oracle are useful for later slicing
                "objects": row.get("objects"),
                "oracle": row.get("oracle"),
            }

            yield FlatExample(
                custom_id=custom_id,
                row_idx=i,
                q_idx=qi,
                question_id=qid,
                question=qtext,
                gold=gold,
                image_b64=img_b64,
                meta=meta,
            )


# -------------------------
# batch creation / polling
# -------------------------
def build_batch_jsonl(
    out_jsonl: Path,
    examples: List[FlatExample],
    *,
    model: str,
    max_output_tokens: int = 1024,
    system_prompt: str = (
        "You are evaluating a robotics VQA benchmark. "
        "Answer as concisely as possible.\n"
        "Rules:\n"
        "- For yes/no questions, output exactly: Yes or No\n"
        "- For multiple-choice questions, output exactly: A1, A2, A3, A4, or A5\n"
        "Do not output any extra words."
    ),
):
    """
    Batch API JSONL:
      { custom_id, method, url, body }
    Endpoint:
      /v1/responses
    Body:
      messages: [ {role:system,...}, {role:user, content:[input_text,input_image]} ]
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for ex in examples:
            body = {
                "model": model,
                "max_output_tokens": int(max_output_tokens),
                "input": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": ex.question},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{ex.image_b64}",
                            },
                        ],
                    },
                ],
            }

            task = {
                "custom_id": ex.custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def upload_and_create_batch(
    client: OpenAI,
    jsonl_path: Path,
    *,
    completion_window: str = "24h",
) -> str:
    batch_file = client.files.create(file=jsonl_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    return batch.id


def poll_batch_until_done(
    client: OpenAI,
    batch_id: str,
    *,
    poll_s: float = 10.0,
) -> Dict[str, Any]:
    while True:
        b = client.batches.retrieve(batch_id)
        status = getattr(b, "status", None) or b.get("status")
        if status in ("completed", "failed", "cancelled", "expired"):
            # convert to dict-like
            if hasattr(b, "model_dump"):
                return b.model_dump()
            return dict(b)
        time.sleep(float(poll_s))


def download_batch_results(
    client: OpenAI,
    batch_info: Dict[str, Any],
    out_jsonl: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Batch output file (and error file) are file IDs.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    output_file_id = batch_info.get("output_file_id")
    error_file_id = batch_info.get("error_file_id")

    out_path = None
    err_path = None

    if output_file_id:
        content = client.files.content(output_file_id).content
        out_path = out_jsonl
        out_path.write_bytes(content)

    if error_file_id:
        content = client.files.content(error_file_id).content
        err_path = out_jsonl.with_suffix(".errors.jsonl")
        err_path.write_bytes(content)

    return out_path, err_path


# -------------------------
# scoring
# -------------------------
def score_results(
    result_jsonl: Path,
    gold_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    gold_map[custom_id] = {gold, meta...}
    result lines look like cookbook:
      {
        "custom_id": "...",
        "response": { "status_code": 200, "body": { ... } }
      }
    (same envelope even when using /v1/responses)  [oai_citation:4‡OpenAI Cookbook](https://cookbook.openai.com/examples/batch_processing)
    """
    total = 0
    correct = 0
    by_qid: Dict[str, Dict[str, int]] = {}

    rows: List[Dict[str, Any]] = []

    with result_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            resp_env = obj.get("response") or {}
            body = resp_env.get("body") or {}
            status_code = resp_env.get("status_code", 200)

            resp_status = body.get("status", None)

            if status_code != 200 or resp_status != "completed":
                text = ""
            else:
                text = _extract_output_text_from_responses_body(body)

            pred = _normalize_pred_answer(text)

            gold_item = gold_map.get(cid, {})
            gold = _normalize_pred_answer(gold_item.get("gold", ""))

            ok = bool(pred == gold and gold != "")
            total += 1
            correct += int(ok)

            qid = str(gold_item.get("question_id", "UNK"))
            if qid not in by_qid:
                by_qid[qid] = {"total": 0, "correct": 0}
            by_qid[qid]["total"] += 1
            by_qid[qid]["correct"] += int(ok)

            rows.append(
                {
                    "custom_id": cid,
                    "pred_raw": text,
                    "pred": pred,
                    "gold": gold,
                    "correct": ok,
                    "question_id": qid,
                    "meta": gold_item.get("meta", {}),
                }
            )

    acc = (correct / total) if total else 0.0
    by_qid_acc = {
        qid: (v["correct"] / v["total"] if v["total"] else 0.0)
        for qid, v in by_qid.items()
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "by_question_id": by_qid,
        "by_question_id_accuracy": by_qid_acc,
        "rows": rows,
    }


# -------------------------
# main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="OpenAI model id")
    p.add_argument("--dataset_dir", type=str, required=True, help="HF save_to_disk dir")
    p.add_argument(
        "--image_key",
        type=str,
        default="agentview_image",
        help="dataset image column key",
    )

    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_questions", type=int, default=None)  # optional truncation
    p.add_argument("--jpeg_quality", type=int, default=90)

    # responses params
    p.add_argument("--max_output_tokens", type=int, default=32)

    # batch params
    p.add_argument("--completion_window", type=str, default="24h")
    p.add_argument("--poll_s", type=float, default=10.0)

    # outputs
    p.add_argument(
        "--work_dir",
        type=str,
        default="./openai_batch_eval",
        help="where to write jsonl + results",
    )
    p.add_argument("--save_json", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="only write input jsonl")

    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    ds = load_from_disk(args.dataset_dir)

    # flatten
    examples: List[FlatExample] = []
    for ex in iter_flat_examples(
        ds,
        image_key=args.image_key,
        max_rows=args.max_rows,
        jpeg_quality=args.jpeg_quality,
    ):
        examples.append(ex)
        if args.max_questions is not None and len(examples) >= int(args.max_questions):
            break

    if not examples:
        raise RuntimeError("No examples produced. Check dataset schema / image_key.")

    # write batch jsonl
    input_jsonl = work_dir / "batch_input.jsonl"
    build_batch_jsonl(
        input_jsonl,
        examples,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
    )

    # gold mapping for scoring
    gold_map = {
        ex.custom_id: {
            "gold": ex.gold,
            "question_id": ex.question_id,
            "meta": ex.meta,
        }
        for ex in examples
    }
    (work_dir / "gold_map.json").write_text(
        json.dumps(gold_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Wrote batch input: {input_jsonl} ({len(examples)} requests)")

    if args.dry_run:
        print("[DRY RUN] Not submitting batch.")
        return

    # submit batch
    client = OpenAI()

    batch_id = upload_and_create_batch(
        client, input_jsonl, completion_window=args.completion_window
    )
    print(f"[OK] Created batch: {batch_id}")

    # poll
    batch_info = poll_batch_until_done(client, batch_id, poll_s=args.poll_s)
    status = batch_info.get("status")
    print(f"[OK] Batch status: {status}")

    (work_dir / "batch_info.json").write_text(
        json.dumps(batch_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # download results
    result_jsonl = work_dir / "batch_output.jsonl"
    out_path, err_path = download_batch_results(client, batch_info, result_jsonl)
    if out_path:
        print(f"[OK] Downloaded results: {out_path}")
    if err_path:
        print(f"[WARN] Downloaded errors:  {err_path}")

    if status != "completed":
        print("[WARN] Batch did not complete successfully. See batch_info/errors.")
        return

    # score
    report = score_results(out_path, gold_map)
    print(
        f"[RESULT] accuracy={report['accuracy']:.4f} ({report['correct']}/{report['total']})"
    )
    print("[RESULT] by_question_id_accuracy:", report["by_question_id_accuracy"])

    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] Saved report to: {args.save_json}")
    else:
        (work_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] Saved report to: {work_dir / 'report.json'}")


if __name__ == "__main__":
    main()
