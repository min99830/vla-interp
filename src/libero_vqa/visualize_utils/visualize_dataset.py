#!/usr/bin/env python3
# libero_vqa/gradio_viewer.py
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

SHORTCUT_JS = """<script>
(function () {
  function isTypingTarget(el) {
    if (!el) return false;
    const tag = (el.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || el.isContentEditable;
  }

  function clickGradioButton(elemId) {
    const root = document.getElementById(elemId);
    if (!root) return false;
    if (root.tagName && root.tagName.toLowerCase() === "button") { root.click(); return true; }
    const btn = root.querySelector("button");
    if (btn) { btn.click(); return true; }
    return false;
  }

  window.addEventListener("keydown", function (e) {
    if (isTypingTarget(document.activeElement)) return;
    if (e.isComposing) return;

    if (e.key === "ArrowLeft") {
      if (clickGradioButton("prev_btn")) e.preventDefault();
    } else if (e.key === "ArrowRight") {
      if (clickGradioButton("next_btn")) e.preventDefault();
    }
  }, { passive: false });
})();
</script>"""


# -------------------------
# utils
# -------------------------
def _ensure_uint8_hwc(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={arr.shape}")

    # CHW -> HWC
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # dtype -> uint8
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


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _scan_eval_files(eval_dir: Optional[str]) -> List[str]:
    """
    Return a list of eval file paths (strings), sorted.
    We accept:
      - report.json (dict with "results") or raw list json
      - optionally *.jsonl if you want later (keep json only for now)
    """
    if not eval_dir:
        return []
    d = Path(eval_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"eval_dir not found or not a dir: {d}")

    files = sorted([p for p in d.glob("*.json") if p.is_file()])
    return [str(p) for p in files]


def _normalize_eval_to_results(report_like: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert report.json style -> results style.

    Input row example:
      {
        "custom_id": "...",
        "pred_raw": "Yes",
        "pred": "Yes",
        "gold": "No",
        "correct": false,
        "question_id": "M1",
        "meta": {
          "suite": "...",
          "task": "...",
          "episode_id": "...",
          "step_idx": 5,
          "objects": {...},
          "oracle": {...}
        }
      }

    Output result item (viewer expects):
      {
        "suite","task","episode_id","step_idx",
        "qid","question","gt","pred","raw_pred","correct"
      }
    """
    out: Dict[str, Any] = {"results": []}
    rows = report_like.get("rows", [])
    for r in rows:
        if not isinstance(r, dict):
            continue
        meta = r.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        qid = _safe_str(r.get("question_id") or meta.get("question_id") or r.get("qid"))
        suite = _safe_str(meta.get("suite") or r.get("suite"))
        task = _safe_str(meta.get("task") or r.get("task"))
        episode_id = _safe_str(meta.get("episode_id") or r.get("episode_id"))
        step_idx = int(
            meta.get("step_idx")
            if meta.get("step_idx") is not None
            else (r.get("step_idx") or 0)
        )

        # viewer table columns
        gt = _safe_str(r.get("gold") if "gold" in r else r.get("gt"))
        pred = _safe_str(r.get("pred"))
        raw_pred = _safe_str(
            r.get("pred_raw") if "pred_raw" in r else r.get("raw_pred", pred)
        )
        ok = bool(r.get("correct", False))

        # question text may not exist in report.json rows
        qtext = _safe_str(r.get("question", ""))

        out["results"].append(
            {
                "suite": suite,
                "task": task,
                "episode_id": episode_id,
                "step_idx": step_idx,
                "qid": qid,
                "question": qtext,  # might be "", that's okay
                "gt": gt,
                "pred": pred,
                "raw_pred": raw_pred,
                "correct": ok,
                # optionally keep useful extra fields (won't show unless you extend UI)
                "objects": meta.get("objects", None),
                "oracle": meta.get("oracle", None),
                "custom_id": r.get("custom_id", None),
            }
        )
    return out


def _load_eval_any(eval_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Supports:
      A) { "results": [ ... ] } or raw list -> {"results": list}
      B) report.json style:
         { "rows": [ {pred_raw,gold,question_id,meta:{suite,task,episode_id,step_idx,...}}, ... ], ... }
         -> converted into {"results": [...]}
    """
    if not eval_path or eval_path == "(none)":
        return None

    p = Path(eval_path)
    if not p.exists():
        raise FileNotFoundError(f"eval file not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))

    # Case: raw list
    if isinstance(data, list):
        return {"results": data}

    if not isinstance(data, dict):
        raise ValueError("eval file must be a JSON dict or list")

    # Case A: already has results
    if "results" in data and isinstance(data["results"], list):
        return data

    # Case B: report.json style with "rows"
    if "rows" in data and isinstance(data["rows"], list):
        return _normalize_eval_to_results(data)

    raise ValueError(
        "Unsupported eval format. Expect either {'results':[...]}, a raw list, or report.json {'rows':[...]}"
    )


def _group_eval_results(
    eval_data: Dict[str, Any],
) -> Dict[Tuple[str, str, str, int], List[Dict[str, Any]]]:
    """
    Groups by (suite, task, episode_id, step_idx).
    Expects each result has: suite, task, episode_id, step_idx, qid(or question_id), question, gt, pred/raw_pred, correct.
    """
    groups: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in eval_data.get("results", []):
        key = (
            _safe_str(r.get("suite")),
            _safe_str(r.get("task")),
            _safe_str(r.get("episode_id")),
            int(r.get("step_idx", 0)),
        )
        rr = dict(r)
        if "qid" not in rr and "question_id" in rr:
            rr["qid"] = rr["question_id"]
        groups[key].append(rr)
    return groups


def _index_dataset(
    ds,
) -> Tuple[
    List[Tuple[str, str, str, int]],
    Dict[Tuple[str, str, str, int], int],
    Dict[str, List[str]],
]:
    """
    Builds:
      - keys list (suite, task, episode_id, step_idx) in dataset order
      - key->row_idx map
      - suite->tasks map

    ✅ Optimized: select_columns + to_dict (avoid decoding images)
    """
    cols = ["suite", "task", "episode_id", "step_idx"]
    tbl = ds.select_columns([c for c in cols if c in ds.column_names])
    d = tbl.to_dict()

    n = len(tbl)
    keys: List[Tuple[str, str, str, int]] = []
    key2idx: Dict[Tuple[str, str, str, int], int] = {}
    suite2tasks: Dict[str, set] = defaultdict(set)

    suites = d.get("suite", [""] * n)
    tasks = d.get("task", [""] * n)
    eps = d.get("episode_id", [""] * n)
    steps = d.get("step_idx", [0] * n)

    for i in tqdm(range(n), desc="Indexing dataset", unit="row"):
        suite = _safe_str(suites[i])
        task = _safe_str(tasks[i])
        episode_id = _safe_str(eps[i])
        step_idx = int(steps[i]) if steps[i] is not None else 0

        k = (suite, task, episode_id, step_idx)
        keys.append(k)
        key2idx[k] = i
        if suite and task:
            suite2tasks[suite].add(task)

    suite2tasks_sorted = {s: sorted(list(ts)) for s, ts in suite2tasks.items()}
    return keys, key2idx, suite2tasks_sorted


def _make_questions_table(row: Dict[str, Any]) -> List[List[Any]]:
    out: List[List[Any]] = []
    qs = row.get("questions", [])
    if not isinstance(qs, list):
        return out

    order = {"M1": 0, "M2": 1, "M3": 2}
    qs = sorted(qs, key=lambda q: order.get(_safe_str(q.get("id")), 999))

    for q in qs:
        qid = _safe_str(q.get("id"))
        qtext = _safe_str(q.get("question"))
        ans = _safe_str(q.get("answer"))
        choices = q.get("choices", None)
        choices_s = "" if choices is None else _safe_str(choices)
        out.append([qid, qtext, ans, choices_s])
    return out


def _make_eval_table(items: List[Dict[str, Any]]) -> List[List[Any]]:
    out: List[List[Any]] = []
    if not items:
        return out
    order = {"M1": 0, "M2": 1, "M3": 2}
    items = sorted(items, key=lambda r: order.get(_safe_str(r.get("qid")), 999))

    for r in items:
        qid = _safe_str(r.get("qid"))
        q = _safe_str(r.get("question"))
        pred_raw = _safe_str(r.get("raw_pred", r.get("pred", "")))
        gt = _safe_str(r.get("gt"))
        ok = bool(r.get("correct", False))
        out.append([qid, q, pred_raw, gt, "✓" if ok else "✗"])
    return out


def _row_summary(row: Dict[str, Any]) -> str:
    suite = _safe_str(row.get("suite"))
    task = _safe_str(row.get("task"))
    episode_id = _safe_str(row.get("episode_id"))
    step_idx = _safe_str(row.get("step_idx"))

    objs = row.get("objects") or {}
    A = B = None
    if isinstance(objs, dict):
        A = objs.get("A")
        B = objs.get("B")

    s = f"{suite} / {task} / {episode_id} / step={step_idx}"
    if A and B:
        s += f"   |   A={A}, B={B}"
    return s


def _maybe_filter_keys(
    keys: List[Tuple[str, str, str, int]],
    *,
    suite: str,
    task: str,
    episode_regex: str,
    step_min: int,
    step_max: int,
    only_in_eval: bool,
    only_incorrect: bool,
    eval_groups: Optional[Dict[Tuple[str, str, str, int], List[Dict[str, Any]]]],
) -> List[Tuple[str, str, str, int]]:
    suite = suite.strip()
    task = task.strip()
    episode_regex = episode_regex.strip()

    prog = re.compile(episode_regex) if episode_regex else None

    out = []
    for k in keys:
        s, t, ep, st = k
        if suite and s != suite:
            continue
        if task and t != task:
            continue
        if prog is not None and not prog.search(ep):
            continue
        if st < int(step_min) or st > int(step_max):
            continue

        if eval_groups is not None:
            in_eval = k in eval_groups
            if only_in_eval and not in_eval:
                continue
            if only_incorrect:
                if not in_eval:
                    continue
                items = eval_groups.get(k, [])
                if not any(not bool(x.get("correct", False)) for x in items):
                    continue

        out.append(k)
    return out


# -------------------------
# app
# -------------------------
def build_app(
    dataset_dir: str,
    *,
    image_key: str = "agentview_image",
    aux_image_key: str = "eye_in_hand_image",
    eval_dir: Optional[str] = None,
):
    ds = load_from_disk(dataset_dir)
    keys, key2idx, suite2tasks = _index_dataset(ds)

    # ✅ scan eval files once
    eval_files = _scan_eval_files(eval_dir) if eval_dir else []
    eval_choices = ["(none)"] + eval_files
    default_eval = "(none)"

    # eval state (selected file -> groups)
    eval_groups0 = None

    all_suites = sorted(list(suite2tasks.keys()))
    default_suite = all_suites[0] if all_suites else ""
    default_task = (
        suite2tasks[default_suite][0]
        if default_suite and suite2tasks.get(default_suite)
        else ""
    )

    step_vals = [k[3] for k in keys] or [0]
    step_min0, step_max0 = int(min(step_vals)), int(max(step_vals))

    def refresh_tasks(suite: str):
        ts = suite2tasks.get(suite, [])
        first = ts[0] if ts else ""
        return gr.Dropdown(choices=ts, value=first)

    def load_eval_selected(eval_path: str):
        """
        Returns:
          - eval_groups state (dict or None)
          - helper status string
        """
        if not eval_path or eval_path == "(none)":
            return None, "Eval overlay: (none)"
        data = _load_eval_any(eval_path)
        groups = _group_eval_results(data)
        return groups, f"Eval overlay: {eval_path}  (groups={len(groups)})"

    def apply_filters(
        suite: str,
        task: str,
        episode_regex: str,
        step_min: int,
        step_max: int,
        only_in_eval: bool,
        only_incorrect: bool,
        eval_groups: Optional[Dict[Tuple[str, str, str, int], List[Dict[str, Any]]]],
    ):
        filtered = _maybe_filter_keys(
            keys,
            suite=suite,
            task=task,
            episode_regex=episode_regex,
            step_min=step_min,
            step_max=step_max,
            only_in_eval=only_in_eval,
            only_incorrect=only_incorrect,
            eval_groups=eval_groups,
        )
        idx = 0
        status = f"Filtered: {len(filtered)} timesteps"
        return filtered, idx, status

    def render_one(
        filtered_keys: List[List[Any]] | List[Tuple[str, str, str, int]],
        idx: int,
        eval_groups: Optional[Dict[Tuple[str, str, str, int], List[Dict[str, Any]]]],
    ):
        if not filtered_keys:
            return None, None, "No items (check filters).", [], []

        idx = int(idx)
        idx = max(0, min(idx, len(filtered_keys) - 1))
        k = tuple(filtered_keys[idx])

        row_i = key2idx.get(k, None)
        if row_i is None:
            return None, None, f"Missing row for key={k}", [], []

        row = ds[row_i]
        img_main = _ensure_uint8_hwc(row[image_key]) if image_key in row else None
        img_aux = (
            _ensure_uint8_hwc(row[aux_image_key])
            if aux_image_key in row and row.get(aux_image_key) is not None
            else None
        )

        header = _row_summary(row)
        q_table = _make_questions_table(row)

        e_table: List[List[Any]] = []
        if eval_groups is not None:
            e_items = eval_groups.get(k, [])
            e_table = _make_eval_table(e_items)

        footer = f"{header}\n[{idx+1}/{len(filtered_keys)}] row_idx={row_i}"
        return img_main, img_aux, footer, q_table, e_table

    def next_item(filtered_keys, idx):
        if not filtered_keys:
            return idx
        return min(int(idx) + 1, len(filtered_keys) - 1)

    def prev_item(filtered_keys, idx):
        if not filtered_keys:
            return idx
        return max(int(idx) - 1, 0)

    with gr.Blocks(
        title="LIBERO VQA Dataset Viewer", theme=gr.themes.Soft(), head=SHORTCUT_JS
    ) as demo:
        gr.Markdown(
            f"""
# LIBERO VQA HF Dataset Viewer
- dataset_dir: `{dataset_dir}`
- image_key: `{image_key}`, aux_image_key: `{aux_image_key}`
- eval_dir: `{eval_dir or "(none)"}`
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                suite_dd = gr.Dropdown(
                    label="Suite",
                    choices=all_suites,
                    value=default_suite,
                    interactive=True,
                )
                task_dd = gr.Dropdown(
                    label="Task",
                    choices=suite2tasks.get(default_suite, []),
                    value=default_task,
                    interactive=True,
                )

                # ✅ eval selector (from directory)
                eval_dd = gr.Dropdown(
                    label="Eval overlay (choose a json in eval_dir)",
                    choices=eval_choices,
                    value=default_eval,
                    interactive=True,
                )
                eval_status = gr.Markdown("Eval overlay: (none)")

                episode_re = gr.Textbox(
                    label="Episode regex (optional)",
                    placeholder="e.g. demo_0|demo_1",
                    value="",
                )
                with gr.Row():
                    step_min = gr.Slider(
                        label="step_min",
                        minimum=step_min0,
                        maximum=step_max0,
                        value=step_min0,
                        step=1,
                    )
                    step_max = gr.Slider(
                        label="step_max",
                        minimum=step_min0,
                        maximum=step_max0,
                        value=step_max0,
                        step=1,
                    )

                with gr.Row():
                    only_in_eval = gr.Checkbox(
                        label="Only timesteps present in eval overlay",
                        value=False,
                        interactive=True,
                    )
                    only_incorrect = gr.Checkbox(
                        label="Only incorrect (needs eval overlay)",
                        value=False,
                        interactive=True,
                    )

                apply_btn = gr.Button("Apply filters", variant="primary")
                status = gr.Markdown("")

                filtered_state = gr.State([])  # list of keys
                idx_state = gr.State(0)
                eval_groups_state = gr.State(eval_groups0)

                with gr.Row():
                    prev_btn = gr.Button("◀ Prev", elem_id="prev_btn")
                    next_btn = gr.Button("Next ▶", elem_id="next_btn")

                jump = gr.Slider(
                    label="Jump index",
                    minimum=0,
                    maximum=0,
                    value=0,
                    step=1,
                    interactive=True,
                )

            with gr.Column(scale=2):
                info = gr.Textbox(label="Info", value="", lines=3)
                with gr.Row():
                    img_main = gr.Image(
                        label="Main camera", type="numpy", interactive=False
                    )
                    img_aux = gr.Image(
                        label="Aux camera", type="numpy", interactive=False
                    )

                gr.Markdown("### Dataset questions (gold)")
                q_df = gr.Dataframe(
                    headers=["ID", "Question", "Answer", "Choices"],
                    datatype=["str", "str", "str", "str"],
                    row_count=3,
                    col_count=(4, "fixed"),
                    wrap=True,
                    interactive=False,
                )

                gr.Markdown("### Eval results (overlay, optional)")
                e_df = gr.Dataframe(
                    headers=["ID", "Question", "Pred(raw)", "GT", "Correct"],
                    datatype=["str", "str", "str", "str", "str"],
                    row_count=3,
                    col_count=(5, "fixed"),
                    wrap=True,
                    interactive=False,
                )

        # suite -> task refresh
        suite_dd.change(fn=refresh_tasks, inputs=[suite_dd], outputs=[task_dd])

        # ✅ eval dropdown -> load overlay + re-render current view (without changing filter)
        def _on_eval_change(
            eval_path,
            filtered_keys,
            idx,
            suite,
            task,
            episode_regex,
            smin,
            smax,
            only_eval,
            only_inc,
        ):
            groups, st = load_eval_selected(eval_path)

            # if filters depend on eval (only_in_eval/only_incorrect), recompute filtered list
            fkeys, new_idx, filt_status = apply_filters(
                suite, task, episode_regex, smin, smax, only_eval, only_inc, groups
            )
            img1, img2, footer, qt, et = render_one(fkeys, new_idx, groups)
            jmax = max(0, len(fkeys) - 1)
            return (
                groups,
                st,
                fkeys,
                new_idx,
                filt_status,
                gr.Slider(minimum=0, maximum=jmax, value=new_idx, step=1),
                img1,
                img2,
                footer,
                qt,
                et,
            )

        eval_dd.change(
            fn=_on_eval_change,
            inputs=[
                eval_dd,
                filtered_state,
                idx_state,
                suite_dd,
                task_dd,
                episode_re,
                step_min,
                step_max,
                only_in_eval,
                only_incorrect,
            ],
            outputs=[
                eval_groups_state,
                eval_status,
                filtered_state,
                idx_state,
                status,
                jump,
                img_main,
                img_aux,
                info,
                q_df,
                e_df,
            ],
        )

        def _apply_and_render(
            suite, task, episode_regex, smin, smax, only_eval, only_inc, eval_groups
        ):
            fkeys, idx, st = apply_filters(
                suite, task, episode_regex, smin, smax, only_eval, only_inc, eval_groups
            )
            jmax = max(0, len(fkeys) - 1)
            img1, img2, footer, qt, et = render_one(fkeys, idx, eval_groups)
            return (
                fkeys,
                idx,
                st,
                gr.Slider(minimum=0, maximum=jmax, value=idx, step=1),
                img1,
                img2,
                footer,
                qt,
                et,
            )

        apply_btn.click(
            fn=_apply_and_render,
            inputs=[
                suite_dd,
                task_dd,
                episode_re,
                step_min,
                step_max,
                only_in_eval,
                only_incorrect,
                eval_groups_state,
            ],
            outputs=[
                filtered_state,
                idx_state,
                status,
                jump,
                img_main,
                img_aux,
                info,
                q_df,
                e_df,
            ],
        )

        def _render_from_jump(filtered_keys, j, eval_groups):
            img1, img2, footer, qt, et = render_one(filtered_keys, int(j), eval_groups)
            return int(j), img1, img2, footer, qt, et

        jump.change(
            fn=_render_from_jump,
            inputs=[filtered_state, jump, eval_groups_state],
            outputs=[idx_state, img_main, img_aux, info, q_df, e_df],
        )

        def _nav_prev(filtered_keys, idx, eval_groups):
            new_idx = prev_item(filtered_keys, idx)
            img1, img2, footer, qt, et = render_one(filtered_keys, new_idx, eval_groups)
            return new_idx, gr.Slider(value=new_idx), img1, img2, footer, qt, et

        def _nav_next(filtered_keys, idx, eval_groups):
            new_idx = next_item(filtered_keys, idx)
            img1, img2, footer, qt, et = render_one(filtered_keys, new_idx, eval_groups)
            return new_idx, gr.Slider(value=new_idx), img1, img2, footer, qt, et

        prev_btn.click(
            fn=_nav_prev,
            inputs=[filtered_state, idx_state, eval_groups_state],
            outputs=[idx_state, jump, img_main, img_aux, info, q_df, e_df],
        )
        next_btn.click(
            fn=_nav_next,
            inputs=[filtered_state, idx_state, eval_groups_state],
            outputs=[idx_state, jump, img_main, img_aux, info, q_df, e_df],
        )

        # initial render
        demo.load(
            fn=_apply_and_render,
            inputs=[
                suite_dd,
                task_dd,
                episode_re,
                step_min,
                step_max,
                only_in_eval,
                only_incorrect,
                eval_groups_state,
            ],
            outputs=[
                filtered_state,
                idx_state,
                status,
                jump,
                img_main,
                img_aux,
                info,
                q_df,
                e_df,
            ],
        )

    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_dir", type=str, required=True, help="HF load_from_disk dir"
    )
    p.add_argument("--image_key", type=str, default="agentview_image")
    p.add_argument("--aux_image_key", type=str, default="eye_in_hand_image")

    # ✅ NEW: directory with json eval files
    p.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help="optional dir that contains eval json files (each must be list or dict with 'results')",
    )

    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    app = build_app(
        dataset_dir=args.dataset_dir,
        image_key=args.image_key,
        aux_image_key=args.aux_image_key,
        eval_dir=args.eval_dir,
    )
    app.launch(
        server_name=args.host, server_port=int(args.port), share=bool(args.share)
    )


if __name__ == "__main__":
    main()
