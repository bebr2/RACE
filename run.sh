python RACE.py \
    --dataset "NQ" \
    --model "qwen7b" \
    --data_dir ./modeloutput \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --nli_model potsawee/deberta-v3-large-mnli \
    --llm_model  /path/to/cot/extractor/or/meta-llama/Llama-3.1-8B-Instruct \
    --gpu 