
We provide the implementation for computing RACE scores and AUROC metrics.

### 🔧 Pipeline

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, run the setup script:

```bash
source run.sh
```

In this bash script, we evaluate **DeepSeek-Distill-Qwen7B** on **NQ-Open**. First run:

```bash
python generate.py \
    --model_name "ds7b" \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --dataset_name "NQ" \
    --output_dir "./modeloutput"
```

to generate the model outputs. Then run:

```bash
python cot_extraction.py \
    --cot_extractor_model_path "/path/to/cot/extractor/or/meta-llama/Llama-3.1-8B-Instruct" \
    --model_name "ds7b" \
    --dataset_name "NQ" \
    --data_path "./modeloutput" \
    --output_dir "./modeloutput" \
    --gpu_ids "0"
```

to extract the CoT from the model reasoning. Make sure to replace `/path/to/cot/extractor/or/meta-llama/Llama-3.1-8B-Instruct` with the actual path to your CoT extractor model. Then run:

```bash
python RACE.py \
    --dataset "NQ" \
    --model "ds7b" \
    --data_dir ./modeloutput \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --nli_model potsawee/deberta-v3-large-mnli \
    --llm_model  /path/to/cot/extractor/or/meta-llama/Llama-3.1-8B-Instruct \
    --gpu 
```

to compute the RACE score, as well as the SINdex score.

Finally, evaluate the model outputs:

```bash
python eval.py
```

### 📌 Notes

* The `llm_model` argument in `RACE.py` specifies the proxy model used to compute the attention score and LNPE.
  In our paper, we use **CoT Extractor** as a lightweight proxy to reduce deployment costs.
  Alternatively, you can use other models such as `Llama3.1-8B-Instruct`.

### 🔥Training CoT Extractor

To train the CoT Extractor, you can

```bash
cd train

source sft.sh
```

This will train the CoT Extractor model on the filtered CoT sum dataset (`train/dataset.json`).

### 📁 Provided Files

Under `./modeloutput`, we include the first 100 outputs from **DeepSeek-Distill-Qwen7B** on the **Natural Questions (NQ)** dataset:

* `judge.json`: Hallucination labels (`1` indicates hallucination).
* `result.json`, `sample_result.json`: The model's main output and sampled output, respectively.
* `summary_result.json`, `summary_sample_result.json`: Outputs processed by the CoT Extractor.

Under `./annotated`, we randomly select 250 results from two models and two datasets for annotation to investigate the consistency of LLM-as-Judges with human annotators in the hallucination labeling task (`annotation_results.json`). In this subset, the LLM achieves an accuracy of 0.98, with a Kappa value of 0.9405 indicating its agreement with human annotations. We also manually annotated the outputs of the CoT Extractor (`cot_annotated.json`).