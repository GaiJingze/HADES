#!/bin/bash

INPUT_DIR='./eval/evaluate/results/gen_results/{}/black_box/toxic'
IMAGE_SOURCE_DIR='./dataset/black_box/images'
IMAGE_DEST_DIR='./dataset/white_box/init_images'
SCENARIO='Violence' 
ATTACK_MODEL='llava'

MODEL_PATH="liuhaotian/llava-v1.5-13b"
SAVE_DIR="./dataset/white_box/images"
INPUT_PATH="./eval/evaluate/results/gen_results/llava/black_box/toxic"
IMAGE_DIR="./dataset/black_box/images"
MODE='toxic'

# Run the initial image generation script
python white_box/generate_init_image.py \
  --input-dir "$INPUT_DIR" \
  --image-source-dir "$IMAGE_SOURCE_DIR" \
  --image-dest-dir "$IMAGE_DEST_DIR" \
  --scenario "$SCENARIO" \
  --attack-model "$ATTACK_MODEL"

# Now run the white box attack script
python white_box/white_box_attack.py \
  --model_path "$MODEL_PATH" \
  --attack_model "$ATTACK_MODEL" \
  --save_dir "$SAVE_DIR" \
  --input_path "$INPUT_PATH" \
  --image-dir "$IMAGE_DIR" \
  --scenario "$SCENARIO" \
  --mode "$MODE"

echo "Scripts for initial image generation and white-box attack executed."
