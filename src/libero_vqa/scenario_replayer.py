# libero_vqa/scenario_replayer.py
import time

import libero.libero.utils.utils as libero_utils
import numpy as np


class ScenarioReplayer:
    def __init__(self, env, demo_hdf5, suite, task, cap_index=0, mismatch_tol=0.01):
        self.env = env
        self.h5 = demo_hdf5
        self.suite = suite
        self.task = task
        self.cap_index = cap_index
        self.mismatch_tol = mismatch_tol

    def _safe_reset(self, max_retries=5):
        last_exc = None
        for i in range(max_retries):
            try:
                self.env.reset()
                return
            except Exception as e:
                last_exc = e
                print(f"[reset] failed ({i+1}/{max_retries}): {e}")
                time.sleep(0.2 * (i + 1))

        raise RuntimeError("env.reset failed repeatedly") from last_exc

    def setup_episode(self, ep):
        model_xml = self.h5[f"data/{ep}"].attrs["model_file"]
        model_xml = libero_utils.postprocess_model_xml(model_xml, {})

        states = self.h5[f"data/{ep}/states"][()]
        actions = self.h5[f"data/{ep}/actions"][()]

        self._safe_reset()
        self.env.reset_from_xml_string(model_xml)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(states[0])
        self.env.sim.forward()

        return states, actions

    def iter_steps(self, ep, max_steps=None):
        states, actions = self.setup_episode(ep)
        n = actions.shape[0]
        if max_steps is not None:
            n = min(n, max_steps)

        for j in range(n):
            obs, reward, done, info = self.env.step(actions[j])

            err = None
            if j < n - 1:
                playback = self.env.sim.get_state().flatten()
                err = float(np.linalg.norm(states[j + 1] - playback))
                if err > self.mismatch_tol:
                    info = dict(info)
                    info["state_mismatch"] = err

            if j < self.cap_index:
                continue

            yield {
                "suite": self.suite,
                "task": self.task,
                "episode_id": ep,
                "step_idx": j,
                "obs": obs,
                "reward": reward,
                "done": done,
                "info": info,
                "state_error": err,
            }
