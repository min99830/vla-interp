# libero_vqa/build_dataset.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import libero.libero.utils.utils as libero_utils
import robosuite
from libero.libero import get_default_path_dict
from libero.libero.envs import TASK_MAPPING  # noqa
from tqdm import tqdm

from libero_vqa import oracle_utils
from libero_vqa.hf_writer import HFWriter
from libero_vqa.measurement import MeasurementModule
from libero_vqa.problem_utils import choose_objA_objB
from libero_vqa.scenario_replayer import ScenarioReplayer


# -------------------------
# discovery
# -------------------------
def iter_demo_paths(libero_root: Path, suites: list[str] | None = None):
    """
    Yields:
      <libero_root>/<suite>/<task>_demo.hdf5
    """
    libero_root = Path(libero_root)
    if not libero_root.exists():
        raise FileNotFoundError(f"LIBERO root not found: {libero_root}")

    if not suites:
        suite_dirs = [p for p in libero_root.iterdir() if p.is_dir()]
    else:
        suite_dirs = [(libero_root / s) for s in suites if (libero_root / s).is_dir()]

    for suite_dir in sorted(suite_dirs):
        for h5_path in sorted(suite_dir.glob("*_demo.hdf5")):
            yield h5_path


def parse_suite_task_from_dataset_path(dataset_path: Path) -> tuple[str, str]:
    suite = dataset_path.parent.name
    fname = dataset_path.name
    if not fname.endswith("_demo.hdf5"):
        raise ValueError(f"Expected '*_demo.hdf5' but got: {fname}")
    task = fname.replace("_demo.hdf5", "")
    return suite, task


# -------------------------
# env build
# -------------------------
def load_demo_hdf5(dataset_path: Path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {dataset_path}")
    return h5py.File(str(dataset_path), "r")


def resolve_bddl_path_from_hdf5(demo_hdf5) -> str:
    bddl_file_name = demo_hdf5["data"].attrs["bddl_file_name"]
    default_bddl_root = get_default_path_dict()["bddl_files"]
    suffix = bddl_file_name.split("libero/libero/bddl_files/")[1]
    return str(Path(default_bddl_root) / suffix)


def build_env_from_demo_hdf5(
    demo_hdf5,
    *,
    camera_height=256,
    camera_width=256,
    control_freq=20,
):
    env_name = demo_hdf5["data"].attrs["env_name"]
    env_kwargs = json.loads(demo_hdf5["data"].attrs["env_args"])

    problem_info = json.loads(demo_hdf5["data"].attrs["problem_info"])
    problem_name = problem_info["problem_name"]
    language_instruction = problem_info.get("language_instruction", "")

    bddl_path = resolve_bddl_path_from_hdf5(demo_hdf5)

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=True,
        camera_names=["robot0_eye_in_hand", "agentview"],
        robots=["Panda"],
        controller_configs=robosuite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        reward_shaping=True,
        control_freq=int(control_freq),
        camera_heights=int(camera_height),
        camera_widths=int(camera_width),
        camera_segmentations=None,
    )

    # remove possibly embedded keys (mirrors your original snippet)
    env_args = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": bddl_path,
        "env_kwargs": env_kwargs,
    }
    for key in list(env_args.keys()):
        env_kwargs.pop(key, None)

    env = TASK_MAPPING[problem_name](**env_kwargs)
    return env, problem_name, env_name, language_instruction


def list_episodes(demo_hdf5, max_episodes: int | None = None):
    eps = list(demo_hdf5["data"].keys())
    if max_episodes is not None:
        eps = eps[: int(max_episodes)]
    return eps


# -------------------------
# parsed_problem -> (A,B)
# -------------------------
def _extract_entities_from_literal(lit):
    if not isinstance(lit, (list, tuple)) or len(lit) < 2:
        return None, []
    pred = lit[0]
    args = [a for a in lit[1:] if isinstance(a, str)]
    return pred, args


def extract_problem_entities(parsed_problem: dict):
    init = parsed_problem.get("initial_state", []) or []
    goal = parsed_problem.get("goal_state", []) or []

    goal_args = []
    init_objects = []
    init_regions = []
    all_args = []

    for lit in goal:
        _, args = _extract_entities_from_literal(lit)
        for a in args:
            all_args.append(a)
            goal_args.append(a)

    for lit in init:
        _, args = _extract_entities_from_literal(lit)
        for a in args:
            all_args.append(a)
            if a.endswith("_region"):
                init_regions.append(a)
            else:
                init_objects.append(a)

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


def resolve_to_body_query(model, query: str) -> str | None:
    """
    Measurement oracle currently expects BODY-resolvable queries.
    parsed_problem often yields *_region or part names. We try a ladder:

    1) original
    2) strip trailing "_region"
    3) progressively drop trailing tokens after '_' until something resolves as body
    """
    if query is None:
        return None

    candidates = []
    candidates.append(query)

    if query.endswith("_region"):
        candidates.append(query[: -len("_region")])

    # token-drop ladder: wooden_cabinet_1_middle_region -> wooden_cabinet_1_middle -> wooden_cabinet_1 -> wooden_cabinet
    base = candidates[-1]
    parts = base.split("_")
    for k in range(len(parts) - 1, 0, -1):
        candidates.append("_".join(parts[:k]))

    # unique preserve order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    for c in uniq:
        try:
            # just test if it resolves
            oracle_utils.resolve_object_entity(model, c, prefer=("body",))
            return c
        except Exception:
            continue

    return None


# -------------------------
# runner
# -------------------------
def run_collection(
    *,
    libero_root: Path,
    output_dir: Path,
    suites: list[str] | None = None,
    max_tasks: int | None = None,
    # replay options
    cap_index: int = 5,
    mismatch_tol: float = 0.01,
    max_episodes_per_task: int | None = None,
    max_steps_per_episode: int | None = None,
    # env options
    camera_height: int = 256,
    camera_width: int = 256,
    control_freq: int = 20,
    # A/B selection options
    ab_strategy: str = "goal_vs_init",
    ab_seed: int = 0,
    objA_override: str | None = None,
    objB_override: str | None = None,
    # measurement options
    th_cm: float = 10.0,
    gripper_geom_names: list[str] | None = None,
    # sampling knobs (redundancy control)
    min_gap: int = 5,
    dist_bin_cm: float = 2.0,
    dz_bin_cm: float = 2.0,
    max_keep_steps: int = 25,
    # writer
    keep_images: bool = True,
    verbose: bool = True,
):
    libero_root = Path(libero_root)
    output_dir = Path(output_dir)

    rng = random.Random(int(ab_seed))
    writer = HFWriter(keep_images=keep_images)

    demo_paths = list(iter_demo_paths(libero_root, suites=suites))
    if max_tasks is not None:
        demo_paths = demo_paths[: int(max_tasks)]

    if verbose:
        print(
            f"[run_collection] Found {len(demo_paths)} demo files under {libero_root}"
        )

    for dataset_path in tqdm(demo_paths, desc="LIBERO Tasks", unit="task"):
        suite, task = parse_suite_task_from_dataset_path(dataset_path)

        demo_hdf5 = None
        env = None
        try:
            demo_hdf5 = load_demo_hdf5(dataset_path)
            env, problem_name, env_name, language_instruction = (
                build_env_from_demo_hdf5(
                    demo_hdf5,
                    camera_height=camera_height,
                    camera_width=camera_width,
                    control_freq=control_freq,
                )
            )

            parsed_problem = getattr(env, "parsed_problem", None)
            if parsed_problem is None:
                raise RuntimeError("env.parsed_problem not found")

            model = env.sim.model

            replayer = ScenarioReplayer(
                env=env,
                demo_hdf5=demo_hdf5,
                suite=suite,
                task=task,
                cap_index=int(cap_index),
                mismatch_tol=float(mismatch_tol),
            )

            measurement = MeasurementModule(
                objA=None,
                objB=None,
                th_cm=th_cm,
                gripper_geom_names=gripper_geom_names,
                min_gap=min_gap,
                dist_bin_cm=dist_bin_cm,
                dz_bin_cm=dz_bin_cm,
                max_steps=max_keep_steps,
            )

            episodes = list_episodes(demo_hdf5, max_episodes=max_episodes_per_task)

            seen_pairs: set[tuple[str, str]] = set()

            for ep in tqdm(
                episodes, desc=f"{suite}/{task} episodes", unit="ep", leave=False
            ):
                try:
                    # ---------- episode-wise A/B ----------
                    A_res = B_res = None
                    for _ in range(20):
                        A_auto, B_auto = choose_objA_objB(
                            parsed_problem, rng=rng, strategy=ab_strategy
                        )
                        A = objA_override or A_auto
                        B = objB_override or B_auto
                        if A is None or B is None:
                            continue
                        if (A, B) in seen_pairs:
                            continue

                        A_res = resolve_to_body_query(model, A)
                        B_res = resolve_to_body_query(model, B)
                        if A_res and B_res:
                            seen_pairs.add((A, B))
                            break

                    if A_res is None or B_res is None:
                        raise RuntimeError("Failed to resolve unique (A,B)")

                    measurement.set_objects(A_res, B_res)
                    measurement.reset_episode()

                    try:
                        for ctx in replayer.iter_steps(
                            ep, max_steps=max_steps_per_episode
                        ):
                            pack = measurement.process_step(env, ctx)
                            if pack is not None:
                                writer.add_pack(ctx, pack)

                    except ValueError as ve:
                        if verbose:
                            print(
                                f"[WARN] ValueError during step -> skip episode: {repr(ve)}"
                            )
                        continue
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] step failed -> skip episode: {repr(e)}")
                        continue

                except Exception as e:
                    if verbose:
                        print(f"[WARN] Skip episode {suite}/{task}/{ep}: {repr(e)}")
                    continue

        except Exception as e:
            print(f"[WARN] Failed on task {suite}/{task}: {repr(e)}")

        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if demo_hdf5 is not None:
                try:
                    demo_hdf5.close()
                except Exception:
                    pass

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    ds = writer.save(str(output_dir))
    if verbose:
        print("\nSaved dataset to:", output_dir)
        print(ds)
    return ds


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--libero_root", type=str, required=True, help="LIBERO-datasets root directory"
    )
    p.add_argument(
        "--output_dir", type=str, required=True, help="HF dataset save_to_disk dir"
    )

    # filter
    p.add_argument(
        "--suites", type=str, default="", help="comma-separated suites (empty=all)"
    )
    p.add_argument("--max_tasks", type=int, default=None)

    # episodes/steps
    p.add_argument("--max_episodes_per_task", type=int, default=None)
    p.add_argument("--max_steps_per_episode", type=int, default=None)

    # replay
    p.add_argument("--cap_index", type=int, default=5)
    p.add_argument("--mismatch_tol", type=float, default=0.01)

    # env
    p.add_argument("--camera_height", type=int, default=256)
    p.add_argument("--camera_width", type=int, default=256)
    p.add_argument("--control_freq", type=int, default=20)

    # A/B auto selection
    p.add_argument(
        "--ab_strategy",
        type=str,
        default="goal_vs_init",
        choices=["goal_vs_init", "random"],
    )
    p.add_argument("--ab_seed", type=int, default=0)
    p.add_argument(
        "--objA",
        type=str,
        default=None,
        help="override objA (otherwise auto from parsed_problem)",
    )
    p.add_argument(
        "--objB",
        type=str,
        default=None,
        help="override objB (otherwise auto from parsed_problem)",
    )

    # measurement
    p.add_argument("--th_cm", type=float, default=10.0)

    # sampling knobs
    # p.add_argument("--every_k", type=int, default=10)
    p.add_argument("--min_gap", type=int, default=5)
    p.add_argument("--dist_bin_cm", type=float, default=2.0)
    p.add_argument("--dz_bin_cm", type=float, default=2.0)
    p.add_argument("--max_keep_steps", type=int, default=25)

    # writer
    p.add_argument("--no_images", action="store_true")

    args = p.parse_args()
    suites = (
        [s.strip() for s in args.suites.split(",") if s.strip()]
        if args.suites
        else None
    )

    run_collection(
        libero_root=Path(args.libero_root),
        output_dir=Path(args.output_dir),
        suites=suites,
        max_tasks=args.max_tasks,
        max_episodes_per_task=args.max_episodes_per_task,
        max_steps_per_episode=args.max_steps_per_episode,
        cap_index=args.cap_index,
        mismatch_tol=args.mismatch_tol,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        control_freq=args.control_freq,
        ab_strategy=args.ab_strategy,
        ab_seed=args.ab_seed,
        objA_override=args.objA,
        objB_override=args.objB,
        th_cm=args.th_cm,
        # every_k=args.every_k,
        min_gap=args.min_gap,
        dist_bin_cm=args.dist_bin_cm,
        dz_bin_cm=args.dz_bin_cm,
        max_keep_steps=args.max_keep_steps,
        keep_images=(not args.no_images),
        verbose=True,
    )


if __name__ == "__main__":
    main()
