#!/bin/bash

# 1. Check if an argument is provided
if [ -z "$1" ]; then
  echo "❌ Error: No dataset JSON provided."
  echo "Usage: bash $0 <relative_path_to_json>"
  echo "Example: bash $0 Attribute_test_sampling/CelebV-Text_Appearance-Caption.json"
  exit 1
fi

# 2. Assign the argument to the variable
TESTING_DATASET_NAME="$1"

# 3. Configuration & Paths
export TESTING_MODEL_NAME="./checkpoint-xxxx"
export FPS_MAX_FRAMES=16
export VIDEO_MAX_PIXELS=100352

# Define Base Paths (Modify these if needed)
BASE_DATASET_DIR="YOUR_DATASET_DIR"
BASE_CKPT_DIR="YOUR_CKPT_DIR"
BASE_RESULT_DIR="./reults"

# Construct the full result path
RESULT_PATH="${BASE_RESULT_DIR}/${TESTING_MODEL_NAME}/${TESTING_DATASET_NAME}"

echo "=========================== STARTING INFERENCE ==========================="
echo "Dataset: ${TESTING_DATASET_NAME}"
echo "Output:  ${RESULT_PATH}"

# 4. Check if result already exists
if [ -f "$RESULT_PATH" ]; then
    echo "⚠️  Result path already exists, skipping..."
    echo "=========================== FINISHED =================================="
    exit 0
else
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$RESULT_PATH")"
fi

# 5. Run Inference
echo "🚀 Running infer..."

CUDA_VISIBLE_DEVICES=0 swift infer \
    --val_dataset "${BASE_DATASET_DIR}/${TESTING_DATASET_NAME}" \
    --ckpt_dir "${BASE_CKPT_DIR}/${TESTING_MODEL_NAME}" \
    --result_path "${RESULT_PATH}" \
    --infer_backend vllm \
    --gpu_memory_utilization 0.85 \
    --torch_dtype bfloat16 \
    --max_new_tokens 2048 \
    --streaming False \
    --max_batch_size 4 \
    --attn_impl flash_attn \
    --limit_mm_per_prompt '{"image": 0, "video": 1}' \
    --max_model_len 49152

echo "=========================== FINISHED =================================="
echo ""
