# #!/bin/bash

# python -m libero_vqa.build_dataset \
#   --libero_root /mnt/hdd/LIBERO-datasets \
#   --output_dir  /mnt/hdd/processed/measurement_vqa_all_hf

python -m libero_vqa.build_dataset \
  --libero_root /mnt/hdd/LIBERO-datasets \
  --output_dir  /mnt/hdd/processed/measurement_vqa_all_hf \
  --max_episodes_per_task 3

# python -m libero_vqa.build_dataset \
#   --libero_root /mnt/hdd/LIBERO-datasets \
#   --output_dir  /mnt/hdd/processed/measurement_vqa_debug_hf \
#   --max_tasks 5 \
#   --max_episodes_per_task 2 \
#   --max_steps_per_episode 80

# python -m libero_vqa.build_dataset \
#   --libero_root /mnt/hdd/LIBERO-datasets \
#   --output_dir  /mnt/hdd/processed/measurement_vqa_override_hf \
#   --objA wooden_cabinet_1 \
#   --objB wine_bottle_1