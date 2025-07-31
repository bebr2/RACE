python generate.py \
    --model_name "ds7b" \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --dataset_name "NQ" \
    --output_dir "./modeloutput"

python cot_extraction.py \
    --cot_extractor_model_path "/path/to/cot/extractor/or/meta-llama/Llama-3.1-8B-Instruct" \
    --model_name "ds7b" \
    --dataset_name "NQ" \
    --data_path "./modeloutput" \
    --output_dir "./modeloutput" \
    --gpu_ids "0"


python RACE.py \
    --dataset "NQ" \
    --model "ds7b" \
    --data_dir ./modeloutput \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --nli_model potsawee/deberta-v3-large-mnli \
    --llm_model  /path/to/cot/extractor/or/meta-llama/Llama-3.1-8B-Instruct \
    --gpu 