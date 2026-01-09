# libero_vqa/eval/infer_backends/internvl.py
from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(pil_image: Image.Image, input_size=448, max_num=12) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    images = dynamic_preprocess(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def _make_lazy_infer_fn(model_id: str):
    """
    Returns an infer_fn that lazily loads processor/model
    on first call.
    """
    tokenizer = None
    model = None

    def _load():
        nonlocal tokenizer, model
        if model is not None:
            return

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
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

        pixel_values = load_image(pil_image)
        gen_kwargs = dict(max_new_tokens=int(max_new_tokens))
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            gen_kwargs.update(do_sample=False)

        response = model.chat(tokenizer, pixel_values, user_text, gen_kwargs)

        return response

    return infer_fn


for model_id in MODEL_IDS:
    register_infer_fn(model_id, _make_lazy_infer_fn(model_id))


# if __name__ == "__main__":
#     import requests
#     from PIL import Image

#     url = "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"
#     image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     prompt = "Describe the objects in the image."

#     infer_fn = get_infer_fn(MODEL_IDS[0])
#     answer = infer_fn(image, prompt, max_new_tokens=64)
#     print("Answer:", answer)
