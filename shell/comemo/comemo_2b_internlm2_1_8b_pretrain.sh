set -x

GPUS=${GPUS:-8}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34817
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/comemo/comemo_chat_pretrain'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  comemo/train/comemo_chat_train.py \
  --mixin_config_path "/path/to/mixin_config"\
  --vision_path "./pretrained/InternViT-300M-448px" \
  --llm_path "./pretrained/internlm2-chat-7b" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/internvl_1_2_finetune.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --unfreeze_mixin True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --max_steps 10000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 10 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  --use_packed_ds True \
  --num_images_expected 48 \
  --max_packed_tokens 16384 \
  --max_buffer_size 20 \
  --log_freq 1000 \
  --strict_mode False \
  --replacement False \
  --allow_overflow False \
  --remove_unused_columns False \
  --loss_reduction "square" \
  --loss_reduction_all_gather True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
