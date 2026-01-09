# libero_vqa/oracle_utils.py
from __future__ import annotations

import numpy as np


def get_all_names(model):
    names = {
        "body": [model.body(i).name for i in range(model.nbody)],
        "site": [model.site(i).name for i in range(model.nsite)],
        "geom": [model.geom(i).name for i in range(model.ngeom)],
    }
    return names


def name2id(model, kind: str, name: str) -> int:
    """
    kind: 'body' | 'site' | 'geom'
    """
    fn = getattr(model, f"{kind}_name2id", None)
    if fn is not None:
        return int(fn(name))

    names = get_all_names(model)[kind]
    return int(names.index(name))


def get_body_pos(model, data, body_name: str):
    bid = name2id(model, "body", body_name)
    return data.body_xpos[bid].copy()


def get_site_pos(model, data, site_name: str):
    sid = name2id(model, "site", site_name)
    return data.site_xpos[sid].copy()


def get_geom_pos(model, data, geom_name: str):
    gid = name2id(model, "geom", geom_name)
    return data.geom_xpos[gid].copy()


def get_gripper_pos(model, data, eef_site_name="gripper0_grip_site"):
    return get_site_pos(model, data, eef_site_name)


def resolve_object_entity(model, obj_query: str, prefer=("body", "site", "geom")):
    """
    obj_query: ex) "mug", "drawer_handle", "cube"
    return: (kind, name)
    """
    names = get_all_names(model)
    q = obj_query.lower()

    # 1) exact match
    for kind in prefer:
        for n in names.get(kind, []):
            if n and n.lower() == q:
                return kind, n

    # 2) substring match
    hits = []
    for kind in prefer:
        for n in names.get(kind, []):
            if n and q in n.lower():
                hits.append((kind, n))

    if hits:
        hits.sort(key=lambda kn: (len(kn[1]), kn[1]))
        return hits[0]

    raise RuntimeError(f"'{obj_query}' not found in model.")


def get_object_pos(model, data, obj_query: str, resolved=None):
    if resolved is None:
        resolved = resolve_object_entity(model, obj_query)
    kind, name = resolved
    if kind == "body":
        return get_body_pos(model, data, name)
    if kind == "site":
        return get_site_pos(model, data, name)
    if kind == "geom":
        return get_geom_pos(model, data, name)
    raise ValueError(resolved)


def body_geoms(model, body_id: int):
    adr = model.body_geomadr[body_id]
    num = model.body_geomnum[body_id]
    return list(range(int(adr), int(adr + num)))


def unwrap_mujoco(model, data):
    """
    robosuite binding_utils 래퍼(MjModel/MjData)에서
    mujoco.mj_geomDistance가 요구하는 raw struct를 꺼낸다.

    - robosuite: model._model, data._data가 존재하는 경우가 많음
    - 혹은 이미 mujoco.MjModel/MjData일 수도 있음
    """
    raw_model = getattr(model, "_model", model)
    raw_data = getattr(data, "_data", data)
    return raw_model, raw_data


def geom_distance(model, data, geom1: int, geom2: int, distmax=1e9):
    import mujoco

    raw_model, raw_data = unwrap_mujoco(model, data)

    fromto = np.zeros((2, 3), dtype=np.float64)
    d = mujoco.mj_geomDistance(
        raw_model, raw_data, int(geom1), int(geom2), float(distmax), fromto
    )
    return float(d), fromto.copy()


def get_gripper_geom_ids_from_names(model, gripper_geom_names):
    ids = []
    name_set = set(gripper_geom_names)
    for gid in range(model.ngeom):
        n = model.geom(gid).name
        if n is None:
            continue
        if n in name_set:
            ids.append(int(gid))
    if len(ids) == 0:
        raise RuntimeError(
            "No gripper geoms matched your whitelist names. "
            "Print candidate geom names containing 'gripper'/'finger' to verify naming."
        )
    return ids


def get_closest_distance(
    model,
    data,
    obj_query: str,
    *,
    gripper_geom_names=None,
    prefer=("body",),
    distmax=1e9,
):
    """
    Fast + robust:
      - subtree geoms supported (wrapper bodies)
      - caches subtree + geom ids per model instance
    Returns:
      (d_gap_m, (g1, g2), fromto)
    """
    # -------------------------
    # gripper geoms
    # -------------------------
    if gripper_geom_names is None:
        gripper_geom_names = [
            "gripper0_hand_visual",
            "gripper0_hand_collision",
            "gripper0_finger1_visual",
            "gripper0_finger1_collision",
            "gripper0_finger1_pad_collision",
            "gripper0_finger2_visual",
            "gripper0_finger2_collision",
            "gripper0_finger2_pad_collision",
        ]

    gripper_geom_ids = get_gripper_geom_ids_from_names(model, gripper_geom_names)
    if len(gripper_geom_ids) == 0:
        raise RuntimeError("No gripper geoms found. Check gripper_geom_names.")

    # -------------------------
    # resolve object (BODY)
    # -------------------------
    kind, obj_name = resolve_object_entity(model, obj_query, prefer=prefer)
    if kind != "body":
        raise RuntimeError(
            f"get_closest_distance expects BODY for obj_query='{obj_query}', "
            f"but resolved to {kind}:{obj_name}"
        )

    root_bid = int(model.body_name2id(obj_name))

    # -------------------------
    # build / fetch model-level caches
    # -------------------------
    # attach caches to model instance (works with robosuite binding objects)
    cache = getattr(model, "_closest_dist_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_closest_dist_cache", cache)

    # 1) children adjacency (build once)
    children = cache.get("children", None)
    if children is None:
        children = [[] for _ in range(model.nbody)]
        parentid = model.body_parentid
        for bid in range(model.nbody):
            pid = int(parentid[bid])
            if pid >= 0:
                children[pid].append(bid)
        cache["children"] = children

    # 2) subtree cache: root_bid -> list[bids]
    subtree_cache = cache.get("subtree", None)
    if subtree_cache is None:
        subtree_cache = {}
        cache["subtree"] = subtree_cache

    # 3) geom cache: root_bid -> list[gids]
    geom_cache = cache.get("geom_ids", None)
    if geom_cache is None:
        geom_cache = {}
        cache["geom_ids"] = geom_cache

    # -------------------------
    # get object geom ids (cached)
    # -------------------------
    obj_geom_ids = geom_cache.get(root_bid, None)
    if obj_geom_ids is None:
        # compute subtree bids (cached)
        bids = subtree_cache.get(root_bid, None)
        if bids is None:
            bids = []
            stack = [root_bid]
            while stack:
                b = stack.pop()
                bids.append(b)
                stack.extend(children[b])
            subtree_cache[root_bid] = bids

        # collect geoms from subtree
        gids = []
        for bid in bids:
            gids.extend(body_geoms(model, int(bid)))

        # unique + stable order
        obj_geom_ids = sorted(set(int(g) for g in gids))
        geom_cache[root_bid] = obj_geom_ids

    if len(obj_geom_ids) == 0:
        raise RuntimeError(
            f"Failed to compute closest distance for obj_query='{obj_query}'. "
            f"Resolved body='{obj_name}' has no geoms (including subtree)."
        )

    # -------------------------
    # brute-force closest distance
    # -------------------------
    best_d = float(distmax)
    best_pair = None
    best_fromto = None

    for g1 in gripper_geom_ids:
        for g2 in obj_geom_ids:
            d_raw, fromto = geom_distance(model, data, g1, g2, distmax=distmax)
            if d_raw < best_d:
                best_d = d_raw
                best_pair = (int(g1), int(g2))
                best_fromto = fromto

    if best_pair is None or best_fromto is None:
        raise RuntimeError(
            f"Failed to compute closest distance for obj_query='{obj_query}'."
        )

    d_gap = max(float(best_d), 0.0)
    return d_gap, best_pair, best_fromto
