model_name="qwen3-14b"
dataset_name="HotpotQA"
model_path="Qwen/Qwen3-14B"

python generate.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --dataset_name ${dataset_name} \
    --output_dir "./modeloutput"

python cot_extraction.py \
    --cot_extractor_model_path bebr2/RACE-CoT-Extractor-Llama-8B \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --data_path "./modeloutput" \
    --output_dir "./modeloutput" \
    --gpu_ids "0"


python main.py \
    --dataset ${dataset_name} \
    --model ${model_name} \
    --data_dir ./modeloutput \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --nli_model potsawee/deberta-v3-large-mnli \
    --llm_model  bebr2/RACE-CoT-Extractor-Llama-8B \
    --sindex_threshold 0.9 \
    --output ${model_name}_race_score.json \
    --gpu 

python eval.py ${model_name} ${dataset_name}