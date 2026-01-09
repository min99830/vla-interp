#!/bin/bash

python -m libero_vqa.eval.eval_libero_vqa \
    --model_id google/gemma-3-4b-it \
    --dataset_dir "/mnt/hdd/processed/measurement_vqa_all_hf/" \
    --max_new_tokens 1024 \
    --save_json ./runs/eval/libero_vqa/all/gemma-3-4b-it.json

python -m libero_vqa.eval.eval_libero_vqa \
    --model_id Qwen/Qwen3-VL-4B-Instruct \
    --dataset_dir "/mnt/hdd/processed/measurement_vqa_all_hf/" \
    --max_new_tokens 1024 \
    --save_json ./runs/eval/libero_vqa/all/qwen-3-vl-4b-instruct.json

python -m libero_vqa.eval.eval_libero_vqa \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_dir "/mnt/hdd/processed/measurement_vqa_all_hf/" \
    --max_new_tokens 1024 \
    --save_json ./runs/eval/libero_vqa/all/qwen-2.5-vl-3b-instruct.json

python -m libero_vqa.eval.eval_libero_vqa \
    --model_id OpenGVLab/InternVL3_5-4B-Instruct\
    --dataset_dir "/mnt/hdd/processed/measurement_vqa_all_hf/" \
    --max_new_tokens 1024 \
    --save_json ./runs/eval/libero_vqa/all/internvl3-5-4b.json

python -m libero_vqa.eval.eval_libero_vqa \
    --model_id OpenGVLab/InternVL3_5-2B-Instruct\
    --dataset_dir "/mnt/hdd/processed/measurement_vqa_all_hf/" \
    --max_new_tokens 1024 \
    --save_json ./runs/eval/libero_vqa/all/internvl3-5-2b.json

python -m libero_vqa.eval.eval_libero_vqa \
    --model_id OpenGVLab/InternVL3-2B-Instruct \
    --dataset_dir "/mnt/hdd/processed/measurement_vqa_all_hf/" \
    --max_new_tokens 1024 \
    --save_json ./runs/eval/libero_vqa/all/internvl3-2b.json