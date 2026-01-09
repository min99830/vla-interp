# libero_vqa/eval/infer_backends/internvl.py
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from libero_vqa.eval.infer_registry import register_infer_fn

MODEL_IDS = [
    "OpenGVLab/InternVL3_5-1B-Instruct",
    "OpenGVLab/InternVL3_5-2B-Instruct",
    "OpenGVLab/InternVL3_5-4B-Instruct",
    "OpenGVLab/InternVL3_5-8B-Instruct",
    "OpenGVLab/InternVL3-1B-Instruct",
    "OpenGVLab/InternVL3-2B-Instruct",
    "OpenGVLab/InternVL3-8B-Instruct",
]


def _make_lazy_infer_fn(model_id: str):
    """
    Returns an infer_fn that lazily loads processor/model
    on first call.
    """
    processor = None
    model = None

    def _load():
        nonlocal processor, model
        if model is not None:
            return

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()

    @torch.inference_mode()
    def infer_fn(
        pil_image: Image.Image,
        user_text: str,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        **_,
    ) -> str:
        _load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = dict(max_new_tokens=int(max_new_tokens))
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            gen_kwargs.update(do_sample=False)

        out = model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        return processor.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

    return infer_fn


for model_id in MODEL_IDS:
    register_infer_fn(model_id, _make_lazy_infer_fn(model_id))


if __name__ == "__main__":
    import requests
    from PIL import Image

    url = "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "Describe the objects in the image."

    infer_fn = _make_lazy_infer_fn(MODEL_IDS[0])
    answer = infer_fn(image, prompt, max_new_tokens=64)
    print("Answer:", answer)
