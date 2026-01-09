# libero_vqa/problem_utils.py
from __future__ import annotations

import random

DEFAULT_REGION_SUFFIXES = ("_region",)


def _extract_entities_from_literal(lit):
    """
    lit example:
      ['on', 'wine_bottle_1', 'main_table_wine_bottle_region']
      ['open', 'wooden_cabinet_1_middle_region']
    returns:
      predicate, args(list[str])
    """
    if not isinstance(lit, (list, tuple)) or len(lit) < 2:
        return None, []
    pred = lit[0]
    args = [a for a in lit[1:] if isinstance(a, str)]
    return pred, args


def extract_problem_entities(parsed_problem: dict):
    """
    returns:
      dict with:
        - goal_args: list[str]
        - init_objects: list[str] (non-region)
        - init_regions: list[str] (region-like)
        - all_args: list[str]
    """
    init = parsed_problem.get("initial_state", []) or []
    goal = parsed_problem.get("goal_state", []) or []

    goal_args = []
    init_objects = []
    init_regions = []
    all_args = []

    for lit in goal:
        pred, args = _extract_entities_from_literal(lit)
        for a in args:
            all_args.append(a)
            goal_args.append(a)

    for lit in init:
        pred, args = _extract_entities_from_literal(lit)
        for a in args:
            all_args.append(a)
            if a.endswith(DEFAULT_REGION_SUFFIXES):
                init_regions.append(a)
            else:
                init_objects.append(a)

    # uniq preserve order
    def uniq(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {
        "goal_args": uniq(goal_args),
        "init_objects": uniq(init_objects),
        "init_regions": uniq(init_regions),
        "all_args": uniq(all_args),
    }


def choose_objA_objB(
    parsed_problem: dict,
    *,
    rng: random.Random,
    strategy: str = "goal_vs_init",
):
    """
    strategy:
      - "goal_vs_init": A from goal args, B from init_objects (fallback to all args)
      - "random": A,B random from init_objects
    """
    ent = extract_problem_entities(parsed_problem)

    init_objs = ent["init_objects"]
    goal_args = ent["goal_args"]
    all_args = ent["all_args"]

    if strategy == "random":
        pool = init_objs or [a for a in all_args if not a.endswith("_region")]
        if len(pool) < 2:
            return None, None
        A, B = rng.sample(pool, 2)
        return A, B

    # goal_vs_init
    A_pool = goal_args or init_objs or all_args
    B_pool = init_objs or [a for a in all_args if a not in A_pool]

    if not A_pool:
        return None, None

    A = rng.choice(A_pool)

    # pick B != A
    B_candidates = [b for b in (B_pool or all_args) if b != A]
    if not B_candidates:
        return A, None
    B = rng.choice(B_candidates)

    return A, B
