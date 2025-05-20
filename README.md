
We provide the implementation for computing RACE scores and AUROC metrics.

### 🔧 Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, run the setup script:

```bash
source run.sh
```

Finally, evaluate the model outputs:

```bash
python eval.py
```

### 📌 Notes

* The `llm_model` argument in `RACE.py` specifies the proxy model used to compute the attention score and LNPE.
  In our paper, we use **CoT Extractor** as a lightweight proxy to reduce deployment costs.
  Alternatively, you can use other models such as `Llama3.1-8B-Instruct`.

### 📁 Provided Files

Under `./modeloutput`, we include the first 100 outputs from **DeepSeek-Distill-Qwen7B** on the **Natural Questions (NQ)** dataset:

* `judge.json`: Hallucination labels (`1` indicates hallucination).
* `result.json`, `sample_result.json`: The model's main output and sampled output, respectively.
* `summary_result.json`, `summary_sample_result.json`: Outputs processed by the CoT Extractor.