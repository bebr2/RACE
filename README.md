
# RACE: Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models

📢 **News: this work has been accepted at AAAI 2026 and selected for an Oral presentation!**

If you find our project interesting or helpful, we would appreciate it if you could give us a star! Your support is a tremendous encouragement to us!

Welcome to the official GitHub repository for *RACE: Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models*.

This work is the first to reveal that prior black-box hallucination detection methods are fundamentally flawed when applied to Large Reasoning Models (LRMs).
RACE pioneers the direction of black-box hallucination detection for LRMs by jointly evaluating both reasoning traces and final answers.

In this repository, we provide the implementation for computing RACE scores and AUROC metrics.

## Getting Started

### 🐳 Install Environment

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

And then install `en-core-web-trf-3.8.0` from https://spacy.io/models/en#en_core_web_trf

### RACE API
 
The core implementation of **RACE** is located in [`RACE.py`](./RACE.py). 
To get started, first initialize the scorer: 

```python
from RACE import RACEScorer

race_scorer = RACEScorer(
    embedding_model_path="sentence-transformers/all-MiniLM-L6-v2",
    nli_model_path="potsawee/deberta-v3-large-mnli",
    llm_model_path="bebr2/RACE-CoT-Extractor-Llama-8B",
    use_gpu=True,
    sindex_threshold=0.9
)
```

You can then compute the hallucination score for a single example using:

```python
race_score = race_scorer.calculate_race_score(main_data, sample_data)
```

Both `main_data` and `sample_data` should be Python dictionaries.
RACE supports **two convenient input formats**:

#### 1️⃣ With Pre-Extracted CoTs

If the reasoning steps have already been extracted:

```python
main_data = {
    "question": str,
    "think": str,              # reasoning part (e.g., content between <think>...</think>)
    "final_answer": str,
    "cots": List[str]          # extracted reasoning steps, ordered list
}

sample_data = {
    "final_answer": List[str],
    "cots": List[List[str]]    # extracted CoTs for each sampled answer
}
```

For example:

```python
main_data = {
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "think": "Okay, so I need to figure out if Scott Derrickson and Ed Wood were the same nationality. I'm not entirely sure who each of them is, so I'll start by recalling what I know about them.\n\nFirst, I think Ed Wood is an actor. I remember hearing his name in the context of some movies. Maybe \"The Lord of the Rings\" or something similar? I think he's from the UK because I've heard him speak in British accents and I believe he's associated with British productions.\n\nNow, Scott Derrickson. I'm not as familiar with him. I think he might be an actor too, but I'm not sure about his nationality. I'll try to remember any movies or shows he's been in. Maybe \"The Social Network\"? Wait, isn't that directed by David Fincher? Or was that someone else? Hmm, I'm a bit confused there. Maybe I should look up some of his work. I think he's been in \"The Devil's Advocate\" and \"The Intern\" as well. I'm not sure about his nationality, though.\n\nWait, I think I heard somewhere that Scott Derrickson is of mixed ethnicity. Maybe he's half-African American and half-Caribbean? That could mean he's British by blood but has some African heritage. If that's the case, then he might be British, similar to Ed Wood.\n\nBut I'm not entirely certain. I should double-check. Ed Wood is definitely British, as he's been in many British films and has a strong British accent. For Scott Derrickson, I think he's of mixed descent, possibly with some British influence, but I'm not sure if he's considered British by nationality. Nationality can be tricky because of mixed heritage, but often in such cases, the person's place of birth is considered. If he was born in the UK, even with mixed ethnicity, he'd be British.\n\nSo, putting it together, both are likely British, even if their nationalities are a bit more complex due to mixed heritage. Therefore, they might share the same nationality, which is British.\n",
        "final_answer": "\n\nScott Derrickson and Ed Wood share the same nationality, which is British. Ed Wood is primarily recognized as a British actor known for his roles in productions like \"The Lord of the Rings.\" Scott Derrickson, while of mixed ethnicity, is also of British origin, despite his mixed heritage, and is known for his work in films such as \"The Devil's Advocate\" and \"The Intern.\" Thus, both individuals are considered British due to their place of birth and cultural background.",
        "cots": [
            "Ed Wood is a British actor known for his roles in productions like \"The Lord of the Rings.\"",
            "Scott Derrickson is of mixed ethnicity but has British origins.",
            "Considering their place of birth and cultural background, both are considered British.",
            "Therefore, Scott Derrickson and Ed Wood share the same nationality, which is British."
        ]
}

sample_data = {
        "final_answer": [
            "\n\nScott Derrickson and Ed Wood are not of the same nationality. Ed Wood is an American director, best known for directing the 1994 Bollywood film \"Tata Sinha's Family,\" which was distributed internationally but produced in India. On the other hand, Scott Derrickson is an English actor, and while he worked on \"Tata Sinha's Family,\" which was produced in India, there is no explicit information confirming his nationality. Therefore, based on the available information, it is clear that they are from different countries.",
            "\n\nAfter considering the information and attempting to deduce the nationalities based on the names, here's the conclusion:\n\n**Final Answer:**\n\nEd Wood is known to be American. As for Scott Derrickson, there is some suggestion that his last name, Derrickson, might indicate a connection to the U.S., particularly given similar sounding places like Derrick, Texas. However, without additional information or context about Scott Derrickson, it is not definitively possible to confirm his nationality. Therefore, it is uncertain whether both individuals share the same nationality.",
            "\n\nYes, both Scott Derrickson and Ed Wood are of American nationality. Scott Derrickson, despite having a name common in the U.S., was primarily raised in the United States and has a substantial American background, including directing films like \"The Englishman.\" Ed Wood, being a renowned filmmaker, is definitively American. Therefore, they share the same nationality.",
            "\n\nNo, Scott Derrickson and Ed Wood are not of the same nationality. Ed Wood is an American director known for films like \"Fell into the Fire,\" while Scott Derrickson is a Polish director, active in the video game industry, notably on projects like \"The Witcher 3.\"",
            "\n\nScott Derrickson and Ed Wood were both of American nationality."
        ],
        "cots": [
            [
                "Ed Wood is an American director, known for directing the 1994 Bollywood film \"Tata Sinha's Family,\" which was produced in India.",
                "Scott Derrickson is an English actor and director, who worked on \"Tata Sinha's Family.\"",
                "There is no explicit information confirming Scott Derrickson's nationality as American.",
                "Therefore, based on the available information, Scott Derrickson and Ed Wood are not of the same nationality."
            ],
            [
                "Ed Wood is known to be American.",
                "Scott Derrickson's last name, Derrickson, may suggest a connection to the U.S. due to similar sounding places like Derrick, Texas.",
                "Without additional information or context about Scott Derrickson, his nationality cannot be definitively confirmed.",
                "Therefore, it is uncertain whether both individuals share the same nationality."
            ],
            [
                "Ed Wood is an American filmmaker.",
                "Scott Derrickson has a substantial American upbringing and has directed films like \"The Englishman.\"",
                "Given their American backgrounds, both Scott Derrickson and Ed Wood are American.",
                "Therefore, they share the same nationality."
            ],
            [
                "Ed Wood is an American director known for films like \"Fell into the Fire.\"",
                "Scott Derrickson is a Polish director active in the video game industry, including working on \"The Witcher 3.\"",
                "Since Ed Wood is American and Scott Derrickson is Polish, they are not of the same nationality."
            ],
            [
                "Ed Wood was an American filmmaker.",
                "Scott Derrickson was also an American filmmaker.",
                "Since both were American filmmakers, they were of the same nationality."
            ]
        ]
    }
``` 

#### 2️⃣ Without Pre-Extracted CoTs

If no extracted reasoning is available, RACE will automatically summarize reasoning steps using the CoT Extractor:

```python
main_data = {
    "question": str,
    "think": str,
    "final_answer": str
}

sample_data = {
    "think": List[str],
    "final_answer": List[str]
}
```
 
> 💡 **Note:** We recommend running RACE on at least one 80 GB GPU for efficient CoT extraction and consistency evaluation. 


### 🔧 Pipeline



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