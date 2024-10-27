INPUT_DIR="examples/person_k+hair_c"
OUTPUT_DIR="outputs/magictailor"

LR1=1e-4
LR2=1e-5
STEP1=200
STEP2=300
LORA_RANK=32
LA=1e-2
ALPHA=0.5
GAMMA=32
ED=0.99
LP=0.2

python train.py \
    --seed 0 \
    --mixed_precision fp16 \
    --dataloader_num_workers 8 \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
    --instance_data_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --scale_lr \
    --gsam_repo_dir Grounded-Segment-Anything \
    --phase1_train_steps $STEP1 \
    --phase2_train_steps $STEP2 \
    --phase1_learning_rate $LR1 \
    --phase2_learning_rate $LR2 \
    --lora_rank $LORA_RANK \
    --alpha $ALPHA \
    --gamma $GAMMA \
    --ema_decay $ED \
    --lambda_attention $LA \
    --lambda_preservation $LP \
    --placeholder_token "<v>" \
