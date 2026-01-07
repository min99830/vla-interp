#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm


# -------------------------
# helpers
# -------------------------
def _ensure_uint8_hwc(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={arr.shape}")

    # CHW -> HWC
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(arr.max()) if arr.size else 0.0
            if mx <= 1.5:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    # 1ch -> 3ch
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got shape={arr.shape}")
    return arr


def _load_eval(eval_json: Path) -> Dict[str, Any]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"results": data}
    if isinstance(data, dict) and "results" in data:
        return data
    raise ValueError("eval_json must be dict with 'results' or a list of result dicts.")


def _safe_fname(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def _find_row_index(
    ds, suite: str, task: str, episode_id: str, step_idx: int
) -> Optional[int]:
    # O(N) scan; fastest is to store row_idx in eval output.
    for i in range(len(ds)):
        r = ds[i]
        if (
            r.get("suite") == suite
            and r.get("task") == task
            and r.get("episode_id") == episode_id
            and int(r.get("step_idx")) == int(step_idx)
        ):
            return i
    return None


def _wrap(s: str, width: int) -> str:
    # break_long_words=False가 훨씬 보기 좋음 (moka_pot_1 같은 토큰 유지)
    return textwrap.fill(str(s), width=width, break_long_words=False)


# -------------------------
# pretty rendering
# -------------------------
def _draw_header(ax, header_left: str, header_right: str):
    ax.axis("off")
    ax.set_facecolor("#f6f7f9")
    ax.text(0.01, 0.78, header_left, va="top", ha="left", fontsize=11, weight="bold")
    ax.text(0.99, 0.78, header_right, va="top", ha="right", fontsize=10)
    ax.plot([0.01, 0.99], [0.05, 0.05], lw=1)


def _draw_table(ax, rows: List[Dict[str, Any]], *, model_label: str, q_wrap: int = 60):
    """
    rows: each has qid, question, pred, gt, correct
    """
    ax.axis("off")
    ax.set_facecolor("white")

    # Column positions (axes fraction)
    x_qid = 0.03  # 살짝 오른쪽으로 (ID가 너무 붙어보이던 것 보정)
    x_q = 0.12
    x_pred = 0.73
    x_gt = 0.90
    x_ok = 0.97

    # header (정렬: Question만 left, 나머지는 center)
    y_header = 0.880
    ax.text(x_qid, y_header, "ID", va="center", ha="center", fontsize=10, weight="bold")
    ax.text(
        x_q, y_header, "Question", va="center", ha="left", fontsize=10, weight="bold"
    )
    ax.text(
        x_pred,
        y_header,
        model_label,
        va="center",
        ha="center",
        fontsize=10 if len(model_label) <= 10 else 9,
        weight="bold",
    )
    ax.text(x_gt, y_header, "GT", va="center", ha="center", fontsize=10, weight="bold")
    ax.text(x_ok, y_header, "", va="center", ha="center", fontsize=10, weight="bold")

    ax.plot([0.02, 0.99], [0.915, 0.915], lw=1)  # header underline

    # rows
    y = 0.865  # ✅ 표 전체를 살짝 아래로 내려서 이미지랑 간격 더 자연스럽게
    base_line_h = 0.070  # 줄 높이 살짝 키워서 center 정렬이 안정적

    for idx, r in enumerate(rows):
        qid = str(r.get("qid", "Q"))
        q = _wrap(r.get("question", ""), q_wrap)

        pred = str(r.get("raw_pred", r.get("pred", ""))).strip()
        gt = str(r.get("gt", "")).strip()
        ok = bool(r.get("correct", False))

        q_lines = q.count("\n") + 1
        row_h = base_line_h * max(1, q_lines)

        # alternate shading
        if idx % 2 == 1:
            ax.add_patch(
                plt.Rectangle((0.02, y - row_h + 0.01), 0.97, row_h, alpha=0.04)
            )

        # ✅ 모두 va="center"로 통일해서 행 정렬 깔끔하게
        ax.text(
            x_qid,
            y - row_h / 2,
            qid,
            va="center",
            ha="center",
            fontsize=10,
            weight="bold",
        )

        ax.text(
            x_q,
            y - row_h / 2,
            q,
            va="center",
            ha="left",
            fontsize=10,
            linespacing=1.25,  # ✅ M3 줄바꿈 가독성
        )

        ax.text(x_pred, y - row_h / 2, pred, va="center", ha="center", fontsize=10)
        ax.text(x_gt, y - row_h / 2, gt, va="center", ha="center", fontsize=10)

        mark = "✓" if ok else "✗"
        ax.text(
            x_ok,
            y - row_h / 2,
            mark,
            va="center",
            ha="center",
            fontsize=13,
            weight="bold",
            color=("green" if ok else "red"),
        )

        # separator
        ax.plot([0.02, 0.99], [y - row_h, y - row_h], lw=0.7, alpha=0.5)
        y -= row_h + 0.03

        if y < 0.06:
            ax.text(
                0.5,
                0.02,
                "(truncated)",
                ha="center",
                va="bottom",
                fontsize=9,
                alpha=0.6,
            )
            break


def _ellipsize(s: str, max_len: int = 80) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def make_pretty_figure(
    img_main: np.ndarray,
    table_rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    title_left: str,
    title_right: str,
    model_label: str,
    dpi: int = 200,
    img_aux: Optional[np.ndarray] = None,
):
    fig = plt.figure(figsize=(8.5, 10.5), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 5.4, 3.7])

    ax_header = fig.add_subplot(gs[0])
    _draw_header(
        ax_header,
        _ellipsize(title_left, 90),
        _ellipsize(title_right, 90),
    )

    ax_img = fig.add_subplot(gs[1])
    ax_img.axis("off")
    ax_img.set_aspect("auto")

    if img_aux is None:
        ax_img.imshow(img_main)
    else:
        gap = 8
        h = max(img_main.shape[0], img_aux.shape[0])
        w = img_main.shape[1] + img_aux.shape[1] + gap
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: img_main.shape[0], : img_main.shape[1]] = img_main
        canvas[
            : img_aux.shape[0],
            img_main.shape[1] + gap : img_main.shape[1] + gap + img_aux.shape[1],
        ] = img_aux
        ax_img.imshow(canvas)

        ax_img.text(
            0.01,
            0.99,
            "agentview",
            transform=ax_img.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            alpha=0.8,
        )
        ax_img.text(
            0.52,
            0.99,
            "eye_in_hand",
            transform=ax_img.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            alpha=0.8,
        )

    ax_tbl = fig.add_subplot(gs[2])
    _draw_table(ax_tbl, table_rows, model_label=model_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# -------------------------
# main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--eval_json", type=str, required=True)
    p.add_argument("--image_key", type=str, default="agentview_image")
    p.add_argument("--aux_image_key", type=str, default="eye_in_hand_image")
    p.add_argument("--show_both_cams", action="store_true")
    p.add_argument("--out_dir", type=str, default="./viz_out_pretty")
    p.add_argument("--model_label", type=str, default="model")
    p.add_argument("--max_items", type=int, default=80)
    p.add_argument("--only_incorrect", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    ds = load_from_disk(args.dataset_dir)
    eval_data = _load_eval(Path(args.eval_json))
    results: List[Dict[str, Any]] = eval_data.get("results", [])

    groups: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        key = (
            str(r.get("suite")),
            str(r.get("task")),
            str(r.get("episode_id")),
            int(r.get("step_idx")),
        )
        groups[key].append(r)

    keys = list(groups.keys())
    if args.only_incorrect:
        keys = [
            k for k in keys if any(not bool(x.get("correct", False)) for x in groups[k])
        ]
    keys.sort(key=lambda k: (k[0], k[1], k[2], k[3]))
    keys = keys[: int(args.max_items)]

    qid_order = {"M1": 0, "M2": 1, "M3": 2}

    out_dir = Path(args.out_dir)
    missed = 0

    for i, key in enumerate(tqdm(keys, desc="Visualizing", unit="ts")):
        suite, task, episode_id, step_idx = key
        items = sorted(
            groups[key], key=lambda x: qid_order.get(str(x.get("qid", "")), 999)
        )

        row_i = _find_row_index(ds, suite, task, episode_id, step_idx)
        if row_i is None:
            missed += 1
            continue

        row = ds[row_i]
        img_main = _ensure_uint8_hwc(row[args.image_key])

        img_aux = None
        if args.show_both_cams and args.aux_image_key in row:
            try:
                img_aux = _ensure_uint8_hwc(row[args.aux_image_key])
            except Exception:
                img_aux = None

        objs = row.get("objects") or {}
        objA = objs.get("A") if isinstance(objs, dict) else None
        objB = objs.get("B") if isinstance(objs, dict) else None

        title_left = f"{suite} / {task}"
        right_parts = [f"{episode_id}  step={step_idx}"]
        if objA and objB:
            right_parts.append(f"A={objA}, B={objB}")
        title_right = "   |   ".join(right_parts)

        fname = _safe_fname(f"{i:04d}_{suite}_{task}_{episode_id}_s{step_idx}.png")
        out_path = out_dir / fname

        make_pretty_figure(
            img_main=img_main,
            img_aux=img_aux,
            table_rows=items,
            out_path=out_path,
            title_left=title_left,
            title_right=title_right,
            model_label=args.model_label,
            dpi=int(args.dpi),
        )

    print(f"Saved {len(keys) - missed} figures to: {out_dir}")
    if missed:
        print(f"[WARN] {missed} timesteps couldn't be matched back to dataset rows.")


if __name__ == "__main__":
    main()
