NUM=4
model_path="meta-llama/Llama-3.1-8B-Instruct"
port_addr=11468

deepspeed --master_port=$port_addr --num_gpus=${NUM} ./train.py \
    --report_to "tensorboard" \
    --data_path ./dataset.json \
    --model_name_or_path $model_path \
    --output_dir ./output \
    --model_max_length 3096 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy no \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ./ds_config.json \
    --bf16 True \