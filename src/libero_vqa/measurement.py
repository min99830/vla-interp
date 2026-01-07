# libero_vqa/measurement.py
from __future__ import annotations

import random
import re

from libero_vqa import oracle_utils

MEASUREMENT_TEMPLATES = [
    ("M1", "Is {OBJ_A} closer to the gripper than {OBJ_B}?"),
    ("M2", "Is the gripper within {TH}cm of {OBJ_A}?"),
    (
        "M3",
        "What is the height difference between {OBJ_A} and the gripper (in cm)?\n"
        "A1. {C1}\nA2. {C2}\nA3. {C3}\nA4. {C4}\nA5. {C5}",
    ),
]


# ----------------------
# pretty object names (display only)
# ----------------------
_DEFAULT_ALIASES = {
    # --- common furniture / appliances ---
    "wooden_cabinet": "wooden cabinet",
    "cabinet": "cabinet",
    "drawer": "drawer",
    "microwave": "microwave",
    "stove": "stove",
    "flat_stove": "flat stove",
    "wine_rack": "wine rack",
    "main_table": "table",
    # --- objects (keep color / material words!) ---
    "moka_pot": "moka pot",
    "frypan": "frying pan",
    "frying_pan": "frying pan",
    "wine_bottle": "wine bottle",
    "plate": "plate",
    "book": "book",
    "caddy": "caddy",
    "basket": "basket",
    "tray": "tray",
    # bowls / mugs with color kept
    "bowl": "bowl",
    "black_bowl": "black bowl",
    "white_bowl": "white bowl",
    "akita_black_bowl": "akita black bowl",  # dataset-specific; keep "black"
    "mug": "mug",
    "white_mug": "white mug",
    "red_mug": "red mug",
    "yellow_and_white_mug": "yellow and white mug",
    # foods (keep descriptors)
    "cream_cheese": "cream cheese",
    "cream_cheese_box": "cream cheese box",
    "butter": "butter",
    "milk": "milk",
    "orange_juice": "orange juice",
    "chocolate_pudding": "chocolate pudding",
    "alphabet_soup": "alphabet soup",
    "tomato_sauce": "tomato sauce",
    "bbq_sauce": "BBQ sauce",
    "ketchup": "ketchup",
    "salad_dressing": "salad dressing",
}


def pretty_obj_name(obj_id: str, aliases: dict[str, str] | None = None) -> str:
    """
    Convert LIBERO entity id -> more natural text for question rendering.

    Examples:
      moka_pot_1 -> moka pot
      chefmate_8_frypan_1 -> frying pan
      wooden_cabinet_1_middle_region -> cabinet (fallback-ish)
    """
    s = str(obj_id)

    # strip common vendor/prefix patterns
    s = re.sub(r"^chefmate_\d+_", "", s)
    s = re.sub(r"^main_table_", "", s)

    # strip region suffix
    s = re.sub(r"_region$", "", s)

    # strip trailing numeric instance id
    s = re.sub(r"_\d+$", "", s)

    # normalize underscores
    s = s.replace("_", " ").strip()

    # apply alias: match on "base tokens" without spaces
    # ex: "moka pot" -> key "moka_pot" originally
    key = s.replace(" ", "_")
    amap = aliases or _DEFAULT_ALIASES
    if key in amap:
        return amap[key]

    # also allow prefix alias: flat_stove_1_cook -> "flat stove cook" -> key "flat_stove_cook"
    # try progressively shortening by trailing tokens
    toks = key.split("_")
    for k in range(len(toks), 1, -1):
        kk = "_".join(toks[:k])
        if kk in amap:
            return amap[kk]

    return s


class MeasurementModule:
    def __init__(
        self,
        *,
        objA="cabinet_middle",
        objB="wine",
        th_cm=10,
        gripper_geom_names=None,
        # sampling
        # every_k=10,
        min_gap=5,
        dist_bin_cm=2.0,
        dz_bin_cm=2.0,
        max_steps=25,
        # mcq
        mcq_step=1.0,
        mcq_spread=10.0,
    ):
        self.objA = objA
        self.objB = objB
        self.th_cm = float(th_cm)
        self.gripper_geom_names = gripper_geom_names

        # self.every_k = int(every_k)
        self.min_gap = int(min_gap)
        self.dist_bin_cm = float(dist_bin_cm)
        self.dz_bin_cm = float(dz_bin_cm)
        self.max_steps = int(max_steps)

        self.mcq_step = float(mcq_step)
        self.mcq_spread = float(mcq_spread)

        # ✅ task/module-lifetime: semantic question-set de-dup (DO NOT clear per-episode)
        # signature = (objA_id, objB_id, closer, within, qdz)
        self.seen_question_sets = set()

        self.reset_episode()

    def reset_episode(self):
        self.last_keep = -(10**9)
        self.kept = 0
        self.seen_sig = set()
        self.seen_m12 = {}

    def set_objects(self, objA: str, objB: str):
        self.objA = objA
        self.objB = objB

    # ----------------------
    # oracle
    # ----------------------
    def _measure(self, env):
        model = env.sim.model
        data = env.sim.data

        dA_m, pairA, fromtoA = oracle_utils.get_closest_distance(
            model,
            data,
            self.objA,
            gripper_geom_names=self.gripper_geom_names,
            prefer=("body",),
        )
        dB_m, pairB, fromtoB = oracle_utils.get_closest_distance(
            model,
            data,
            self.objB,
            gripper_geom_names=self.gripper_geom_names,
            prefer=("body",),
        )

        # ✅ height difference: use absolute magnitude for "difference"
        dz_cm = abs(float((fromtoA[1][2] - fromtoA[0][2]) * 100.0))

        return {
            "distA_cm": float(dA_m * 100.0),
            "distB_cm": float(dB_m * 100.0),
            "within": bool(dA_m <= self.th_cm / 100.0),
            "closer": bool(dA_m < dB_m),
            "dz_cm": float(dz_cm),
            "fromto": fromtoA,  # numpy (2,3)
            "pair": pairA,  # (g1,g2)
            "pairB": pairB,
            "fromtoB": fromtoB,
        }

    # ----------------------
    # sampler
    # ----------------------
    def _q(self, x, bin_size):
        return round(float(x) / bin_size) * bin_size

    def _is_hard(self, sig):
        # 1) threshold boundary
        th_edge = abs(sig["distA_cm"] - self.th_cm) <= 2.0
        # 2) closer ambiguity
        diff_edge = abs(sig["distA_cm"] - sig["distB_cm"]) <= 2.0
        return th_edge or diff_edge

    def _question_set_signature(self, sig):
        """
        Semantic (M1/M2/M3-meaning) signature, excluding presentation noise (MCQ shuffle).
        NOTE: keep obj ids (not display names) for stability.
        """
        qdz = self._q(sig["dz_cm"], self.dz_bin_cm)
        return (self.objA, self.objB, bool(sig["closer"]), bool(sig["within"]), qdz)

    def _should_keep(self, step_idx, sig):
        if self.kept >= self.max_steps:
            return False
        if (step_idx - self.last_keep) < self.min_gap:
            return False

        within = bool(sig["within"])
        closer = bool(sig["closer"])

        qdistA = self._q(sig["distA_cm"], self.dist_bin_cm)
        qdz = self._q(sig["dz_cm"], self.dz_bin_cm)
        qdiff = self._q(abs(sig["distA_cm"] - sig["distB_cm"]), self.dist_bin_cm)

        signature = (within, closer, qdistA, qdiff, qdz)

        # ✅ module-lifetime semantic de-dup
        qset_sig = self._question_set_signature(sig)
        if qset_sig in self.seen_question_sets:
            return False

        # 1) full signature de-dup (episode-local)
        if signature in self.seen_sig:
            return False

        # 2) M1/M2 de-dup pressure (episode-local)
        m12 = (within, closer)
        c = self.seen_m12.get(m12, 0)
        if c >= 1 and (not self._is_hard(sig)):
            return False

        # keep
        self.seen_question_sets.add(qset_sig)
        self.seen_sig.add(signature)
        self.seen_m12[m12] = c + 1
        self.last_keep = step_idx
        self.kept += 1
        return True

    # ----------------------
    # VQA generation
    # ----------------------
    def _make_mcq(self, true_val_cm):
        tv = round(float(true_val_cm) / self.mcq_step) * self.mcq_step
        vals = {tv}
        while len(vals) < 5:
            v = (
                round(
                    (tv + random.uniform(-self.mcq_spread, self.mcq_spread))
                    / self.mcq_step
                )
                * self.mcq_step
            )
            vals.add(v)
        vals = list(vals)
        random.shuffle(vals)
        return vals, vals.index(tv)

    def _make_vqas(self, sig):
        vqas = []

        A_disp = pretty_obj_name(self.objA)
        B_disp = pretty_obj_name(self.objB)

        # M1
        vqas.append(
            {
                "question_id": "M1",
                "question": MEASUREMENT_TEMPLATES[0][1].format(
                    OBJ_A=A_disp, OBJ_B=B_disp
                ),
                "answer": "Yes" if sig["closer"] else "No",
            }
        )

        # M2
        vqas.append(
            {
                "question_id": "M2",
                "question": MEASUREMENT_TEMPLATES[1][1].format(
                    OBJ_A=A_disp, TH=int(self.th_cm)
                ),
                "answer": "Yes" if sig["within"] else "No",
            }
        )

        # M3
        choices, idx = self._make_mcq(sig["dz_cm"])
        vqas.append(
            {
                "question_id": "M3",
                "question": MEASUREMENT_TEMPLATES[2][1].format(
                    OBJ_A=A_disp,
                    C1=choices[0],
                    C2=choices[1],
                    C3=choices[2],
                    C4=choices[3],
                    C5=choices[4],
                ),
                "answer": f"A{idx+1}",
                "choices": choices,
            }
        )

        return vqas, {"A": A_disp, "B": B_disp}

    def process_step(self, env, ctx):
        if self.objA is None or self.objB is None:
            return None
        sig = self._measure(env)
        if not self._should_keep(int(ctx["step_idx"]), sig):
            return None

        questions, objects_display = self._make_vqas(sig)

        pack = {
            "objects": {"A": self.objA, "B": self.objB},
            "objects_display": objects_display,
            "oracle": {
                "distA_cm": sig["distA_cm"],
                "distB_cm": sig["distB_cm"],
                "within": sig["within"],
                "closer": sig["closer"],
                "dz_cm": sig["dz_cm"],
                "pairA": list(sig["pair"]) if sig["pair"] is not None else None,
                "fromtoA": (
                    sig["fromto"].tolist() if sig["fromto"] is not None else None
                ),
                "pairB": list(sig["pairB"]) if sig.get("pairB") is not None else None,
                "fromtoB": (
                    sig["fromtoB"].tolist() if sig.get("fromtoB") is not None else None
                ),
                "th_cm": self.th_cm,
            },
            "questions": [
                {
                    "id": q["question_id"],
                    "question": q["question"],
                    "answer": q["answer"],
                    **({"choices": q["choices"]} if "choices" in q else {}),
                }
                for q in questions
            ],
        }
        return pack
