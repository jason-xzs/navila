#!/bin/bash

# Usage:
#   n_node=1 GPUS_PER_NODE=1 MASTER_PORT=29501 MASTER_ADDR=127.0.0.1 CURRENT_RANK=0 \
#   bash scripts/train/sft_8frames_realnav.sh

OUTPUT="/home/nvme01/public_data/xzs_data/checkpoints/navila-8b-realnav-sft"
MODEL_PATH="/home/nvme04/public_data/xzs_data/navila_base"

# Safe defaults for single-node training.
n_node=${n_node:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
CURRENT_RANK=${CURRENT_RANK:-0}

# Low-memory defaults.
# Set FULL_FINETUNE=1 to tune language + vision + projector.
FULL_FINETUNE=${FULL_FINETUNE:-0}
if [[ "$FULL_FINETUNE" == "1" ]]; then
    TUNE_LANGUAGE_MODEL=True
    TUNE_VISION_TOWER=True
    TUNE_MM_PROJECTOR=True
else
    TUNE_LANGUAGE_MODEL=False
    TUNE_VISION_TOWER=False
    TUNE_MM_PROJECTOR=True
fi

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}

# Ensure repo root is discoverable for python imports when launched by torchrun.
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

torchrun --nnodes=$n_node --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    -m llava.train.train_mem \
    --longvila_sampler True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --version llama_3 \
    --seed 10 \
    --data_mixture realnav \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower $TUNE_VISION_TOWER \
    --tune_mm_projector $TUNE_MM_PROJECTOR \
    --tune_language_model $TUNE_LANGUAGE_MODEL \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 100 \
    --fps 0.0 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb



# CUDA_VISIBLE_DEVICES=3 FULL_FINETUNE=1 n_node=1 GPUS_PER_NODE=1 MASTER_PORT=29551 MASTER_ADDR=127.0.0.1 CURRENT_RANK=0 bash scripts/train/sft_8frames_realnav.sh
# CUDA_VISIBLE_DEVICES=1,3,2 FULL_FINETUNE=1 n_node=1 GPUS_PER_NODE=3 MASTER_PORT=29551 MASTER_ADDR=127.0.0.1 CURRENT_RANK=0 bash scripts/train/sft_8frames_realnav.sh