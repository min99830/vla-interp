# libero_vqa/eval/infer_registry.py
__all__ = []

from typing import Callable, Dict, Optional

INFER_REGISTRY: Dict[str, Callable] = {}  # Should be lazy infer_fns to avoid overheads
_BACKEND_IMPORTED = False


def register_infer_fn(model_id: str, fn: Callable, *, overwrite: bool = False):
    if not overwrite and model_id in INFER_REGISTRY:
        raise KeyError(f"infer_fn already registered: {model_id}")
    INFER_REGISTRY[model_id] = fn


def _ensure_backends_imported() -> None:
    global _BACKEND_IMPORTED
    if _BACKEND_IMPORTED:
        return

    from libero_vqa.eval import infer_backends

    _BACKEND_IMPORTED = True


def get_infer_fn(model_id: str) -> Optional[Callable]:
    _ensure_backends_imported()

    if model_id in INFER_REGISTRY:
        return INFER_REGISTRY[model_id]

    # heuristic fallback
    for k, fn in INFER_REGISTRY.items():
        if k in model_id:
            return fn
    return None
