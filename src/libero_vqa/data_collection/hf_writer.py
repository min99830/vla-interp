# libero_vqa/hf_writer.py
from datasets import Dataset


class HFWriter:
    def __init__(self, keep_images=True):
        self.keep_images = keep_images
        self.rows = []

    def _flip_img(self, img):
        if img is None:
            return None
        # img shape: (H, W, C)
        return img[::-1, :, :]

    def add_pack(self, ctx, pack: dict):
        row = {
            "suite": ctx.get("suite"),
            "task": ctx.get("task"),
            "episode_id": ctx.get("episode_id"),
            "step_idx": int(ctx.get("step_idx")),
            "objects": pack["objects"],  # dict
            "oracle": pack["oracle"],  # dict
            "questions": pack["questions"],  # list[dict]
            "state_error": (
                float(ctx["state_error"])
                if ctx.get("state_error") is not None
                else None
            ),
        }

        if self.keep_images:
            obs = ctx.get("obs", {})
            row["agentview_image"] = self._flip_img(obs.get("agentview_image", None))
            row["eye_in_hand_image"] = self._flip_img(
                obs.get("robot0_eye_in_hand_image", None)
            )

        self.rows.append(row)

    def save(self, path: str):
        ds = Dataset.from_list(self.rows)
        ds.save_to_disk(path)
        return ds
