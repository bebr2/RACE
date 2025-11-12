
# RACE: Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models

📢 **News: this work has been accepted at AAAI 2026 and selected for an Oral presentation!**

- **Arxiv Link**: https://arxiv.org/abs/2506.04832

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

### 🍕 RACE API
 
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

For example:

```python
```python
main_data = {
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "think": "Okay, so I need to figure out if Scott Derrickson and Ed Wood were the same nationality. I'm not entirely sure who each of them is, so I'll start by recalling what I know about them.\n\nFirst, I think Ed Wood is an actor. I remember hearing his name in the context of some movies. Maybe \"The Lord of the Rings\" or something similar? I think he's from the UK because I've heard him speak in British accents and I believe he's associated with British productions.\n\nNow, Scott Derrickson. I'm not as familiar with him. I think he might be an actor too, but I'm not sure about his nationality. I'll try to remember any movies or shows he's been in. Maybe \"The Social Network\"? Wait, isn't that directed by David Fincher? Or was that someone else? Hmm, I'm a bit confused there. Maybe I should look up some of his work. I think he's been in \"The Devil's Advocate\" and \"The Intern\" as well. I'm not sure about his nationality, though.\n\nWait, I think I heard somewhere that Scott Derrickson is of mixed ethnicity. Maybe he's half-African American and half-Caribbean? That could mean he's British by blood but has some African heritage. If that's the case, then he might be British, similar to Ed Wood.\n\nBut I'm not entirely certain. I should double-check. Ed Wood is definitely British, as he's been in many British films and has a strong British accent. For Scott Derrickson, I think he's of mixed descent, possibly with some British influence, but I'm not sure if he's considered British by nationality. Nationality can be tricky because of mixed heritage, but often in such cases, the person's place of birth is considered. If he was born in the UK, even with mixed ethnicity, he'd be British.\n\nSo, putting it together, both are likely British, even if their nationalities are a bit more complex due to mixed heritage. Therefore, they might share the same nationality, which is British.\n",
    "final_answer": "\n\nScott Derrickson and Ed Wood share the same nationality, which is British. Ed Wood is primarily recognized as a British actor known for his roles in productions like \"The Lord of the Rings.\" Scott Derrickson, while of mixed ethnicity, is also of British origin, despite his mixed heritage, and is known for his work in films such as \"The Devil's Advocate\" and \"The Intern.\" Thus, both individuals are considered British due to their place of birth and cultural background."
}

sample_data = {
    "think": [
        "Okay, so I'm trying to figure out if Scott Derrickson and Ed Wood are of the same nationality. Let's see, I know that Ed Wood was a really famous director, especially for the 1994 movie \"Tata Sinha's Family.\" Wait, that movie was directed by Ed Wood and it was a big hit in many countries, I think. But was it the USA or somewhere else? I'm a bit fuzzy on that.\n\nThen there's Scott Derrickson. I think he's an English actor, maybe from the UK? I've heard his name in the context of TV shows and movies. I believe he worked with big directors, so maybe he's worked with Ed Wood? If they were both working together, would that mean they're from the same country?\n\nWait, Ed Wood is from the United States, right? He's an American director. But the movie \"Tata Sinha's Family\" is a Bollywood film, so it's made in India. So Ed Wood directed a movie that's distributed internationally, including in the US, but the film itself is produced in India. So Ed is American, but the film is made in India.\n\nAs for Scott Derrickson, I think he's English, as many actors are. If he worked on an American production, especially in Hollywood, he would be from the USA. But if he worked on an Indian production, he could be English. But I'm not sure. Maybe he's of Indian descent? Wait, no, I think he's English.\n\nSo putting it together: Ed Wood is American, directed a film produced in India but distributed globally. Scott Derrickson could be either American or English, depending on his background. Since Ed directed an Indian film, and Scott might be part of a production there, that could mean he's Indian or possibly American if he's in a US-based production.\n\nBut I'm not certain about Scott's nationality. Maybe he's American. If both are in the same film production, maybe they're from the same country. But without more specific info, it's hard to say for sure.\n\nI should probably check that. Maybe I can think of other directors who worked on that movie. But I don't recall any other big names besides Ed Wood. So, if \"Tata Sinha's Family\" is produced in India and directed by an American, but Scott is involved, perhaps he's Indian or American.\n\nWait, but Scott Derrickson worked on \"Tata Sinha's Family,\" right? Or was that someone else? Maybe I should clarify that. If he was part of the production, and the production is in India, then he might be Indian. But if he's in a US-based production, he'd be American.\n\nI'm a bit confused now. Maybe I need to look up more about their backgrounds to get a clearer picture.\n",
        "Okay, so I'm trying to figure out whether Scott Derrickson and Ed Wood were the same nationality. I don't know much about either of them, but I think I can piece it together by looking at their full names and where they're from.\n\nStarting with Scott Derrickson. His last name, Derrickson, seems a bit unusual. I'm not immediately sure where that's from. Maybe I can break it down. The first name is Scott, which I know comes from various places, like the U.S., but that's not helpful. The middle name is Derrick, which I've heard might be a variant of Derrickson. Then the last name is Derrickson. Maybe that's a first name or last name? Wait, I think it's the last name. So maybe the full name is Scott Derrickson, but the middle name Derrick could be part of the last name or another name.\n\nNow, looking at Ed Wood. His last name is Wood, which is pretty common, so I don't think that gives me much. The first name is Ed, which is short for Edward, a common name in the U.S., though I'm not sure where it's common. Maybe the Wood surname is more telling. I know that \"Wood\" can refer to different places or be part of larger names. For example, \"Saul\u6728\" in Chinese is \"Wood,\" but that's probably not relevant here.\n\nWait, Scott Derrickson\u2014maybe it's like Scoll or something? I'm not sure. Alternatively, the middle name could be a typo or a different last name. But since I'm trying to figure out nationality, perhaps I can look at where they're associated.\n\nI recall that Ed Wood was a famous filmmaker. He directed \" gorram \" (but wait, that's a bad joke, maybe not relevant). I think he's American. So if he's American, then who else from the U.S. has a similar name structure?\n\nLooking at Scott Derrickson's nationality. I'm trying to think if I know any famous people with that name. I think I've heard of someone named Scott Derrickson in the context of sports or maybe entertainment. If I can't recall, maybe I should break down the surname. \"Derrickson\" sounds similar to \"Derckson\" or maybe \"Derckson\" isn't a common name elsewhere. It could be a variation of a name from, say, New Zealand or Australia, but I'm not certain.\n\nAlternatively, if I consider the possibility that the full name is Scott Derrickson, with the middle name Derrick, perhaps the last name is something else, but that doesn't make much sense. Maybe it's a typo, and it's actually Derrickson? Or perhaps the full name is Scott Derrickson, but the middle name is different. But I think I'm overcomplicating it.\n\nGiven that Ed Wood is definitely American, perhaps Scott Derrickson is as well. But I need to confirm. Alternatively, maybe one is British and the other American. If I think about the names, \"Derrickson\" might suggest a connection to places like Derrick, Texas, which is in the U.S., so that could hint at an American origin. Alternatively, \"Derrickson\" could be a last name from elsewhere, but without more info, it's hard to tell.\n\nI might be getting stuck because I don't have enough information. Maybe I can think of any other clues. If Scott Derrickson was a notable person, like in sports, perhaps he played basketball or something, but I can't think of a famous Scott Derrickson. Alternatively, maybe he's a public figure in another field, but without more context, it's tough.\n\nAlternatively, if I consider that sometimes surnames can be confusing, like in the case of \"Saul,\" which is a Chinese name, and people might have similar-sounding surnames. But \"Derrickson\" doesn't ring a bell for me in terms of being a common surname in other countries.\n\nWait, perhaps I should check online if I can. But since I can't, I'll try to reason it out. If Ed Wood is American, and if Scott Derrickson were, say, from Canada or the U.K., would that make sense? Or maybe they are both American, given the prominence of their names? Or maybe one is from a different country.\n\nAnother angle: the middle name \"Derrick\" might be a variation of the last name, making the full name something like Scott Derrickson, but again, I'm not certain.\n\nIn the end, without more information, it's hard to be sure. But if Ed Wood is American, it's possible that Scott Derrickson is too, but I can't confirm it definitively.\n",
        "Okay, so I'm trying to figure out whether Scott Derrickson and Ed Wood were of the same nationality. I remember reading about both of them in the context of their work in filmmaking, but I'm not entirely sure about their national origins. Let me think this through step by step.\n\nFirst, I should probably figure out where each of them is from. I know Ed Wood is an American filmmaker, so he's definitely American. Now, Scott Derrickson\u2014his last name, Derrickson, sounds a bit like someone from the United States because of the prevalence of that last name in the U.S. However, I'm not sure about the first name, Scott. Could it be that he's from another country as well?\n\nI think I've heard that Scott Derrickson has worked on several projects, maybe even directing some. Wait, isn't there a movie called \"The Englishman\"? I'm pretty sure that's directed by him. But I'm not certain about the director's background. Was he born in the U.S. or somewhere else?\n\nLet me try to recall or piece together any information I have. I remember that sometimes filmmakers have backgrounds that are quite global, especially if they have worked on international projects. But for Scott Derrickson, if I'm not mistaken, he has a substantial American upbringing and has worked on projects with an American focus, which might suggest he's American. However, I'm not entirely certain because sometimes people from other countries have strong American names, especially in professions where cultural background isn't always obvious from the name.\n\nWait, I think I've heard that Scott Derrickson is of mixed heritage but was primarily raised in the U.S. That would align him with American nationality. But I'm not 100% sure. Maybe I can think of other people with similar names\u2014some might be international. For example, there's a director named Scott Derrickson who worked on \"Life is Short.\" Was that one directed by an international director or an American?\n\nActually, I think \"Life is Short\" was directed by an American, but I might be confusing it with another film. I should be cautious not to rely on half-remembered details. Alternatively, perhaps looking for a list of directors born in the U.S. with the name Derrickson could help. But since I can't look it up now, I have to rely on my existing knowledge.\n\nAnother angle is to consider their films. Ed Wood is known for \"Ed Wood,\" which is a classic film about filmmaking, and he's definitely an American. Scott Derrickson's films, like \"The Englishman,\" have elements of mystery and international intrigue, but if his directorial work is centered in the U.S., that might indicate American nationality. However, I can't be certain just based on the films.\n\nI think I've heard that Scott Derrickson is American, possibly of mixed heritage. Therefore, if both are American, their nationality would be the same. If he's American and Ed Wood is American, then yes, they would share the same nationality.\n\nWait, but I'm not entirely sure about the specifics of his nationality. Maybe I should consider that some people have dual nationalities, but I don't recall reading that. Ed Wood, being a well-known filmmaker, is almost certainly American without any dual citizenship, unless he holds another, which isn't typically associated with him.\n\nSo, putting it all together, both Scott Derrickson and Ed Wood are American, so they would have the same nationality.\n",
        "Okay, so I'm trying to figure out if Scott Derrickson and Ed Wood are of the same nationality. I know both are directors, but I'm not sure about their nationalities. Let me think.\n\nFirst, Ed Wood. I remember he directed a famous movie called \"Fell into the Fire.\" He's been in the news sometimes, but I think he's American. Yeah, I believe he's an American director.\n\nNow, Scott Derrickson. I've heard his name in the context of video games. He worked on \"The Witcher 3,\" right? I think he's from Poland. Poland has a lot of game developers, so that makes sense. He's probably a Polish nationality.\n\nSo, putting that together, Ed Wood is American and Scott Derrickson is Polish. Therefore, they're not of the same nationality. I should check if there's any possibility they're from the same country, but I don't think so. They might not have worked together either, but that's not relevant to their nationality.\n",
        "Okay, so I need to figure out if Scott Derrickson and Ed Wood were the same nationality. I know that Ed Wood was an American filmmaker, so that's straightforward. But I'm not entirely sure about Scott Derrickson. Let me think about where I might have heard of him. I think he was a director or writer in the 80s and 90s, maybe working in the horror genre? Wait, was he American? I'm trying to recall if I heard him mentioned as an American before. Maybe in some film reviews or something. I don't remember any specific details about his nationality. \n\nI wonder if he was born in the U.S. or elsewhere. If he was born in the U.S., he'd be American, but if he was born elsewhere and moved, maybe he became American due to citizenship. But I don't have any information suggesting that. Alternatively, maybe he was from Canada or another country. If he directed or wrote for American films, that doesn't necessarily mean he's American unless he has citizenship.\n\nLet me think about his work. I think there was a film called \"The People Heat.\" Was that by him? I believe so. If that's the case, and if the film was produced in the U.S., then the director might have American nationality. But again, that's not definitive unless I confirm.\n\nWait, another thought: sometimes people who work on U.S. projects without being citizens might be of another nationality. Maybe Scott Derrickson was British? I'm not sure, but I can't think of any famous Scott Derrickson. Maybe I'm confusing with someone else. I should probably check his birthdate and place. If he was born in the U.S., then yes, he'd be American. But without specific information, it's hard to tell.\n\nI think I'll have to conclude that unless there's evidence suggesting he's not American, it's possible that he's American. But since the user is asking if they were the same nationality, and Ed Wood is definitely American, I might need more info about Scott Derrickson. Maybe I should look up Scott Derrickson's nationality to be certain.\n\nWait, actually, a quick search tells me that Scott Derrickson was an American filmmaker, born in the U.S. So he was American, just like Ed Wood. Therefore, they were of the same nationality.\n"
    ],
    "final_answer": [
        "\n\nScott Derrickson and Ed Wood are not of the same nationality. Ed Wood is an American director, best known for directing the 1994 Bollywood film \"Tata Sinha's Family,\" which was distributed internationally but produced in India. On the other hand, Scott Derrickson is an English actor, and while he worked on \"Tata Sinha's Family,\" which was produced in India, there is no explicit information confirming his nationality. Therefore, based on the available information, it is clear that they are from different countries.",
        "\n\nAfter considering the information and attempting to deduce the nationalities based on the names, here's the conclusion:\n\n**Final Answer:**\n\nEd Wood is known to be American. As for Scott Derrickson, there is some suggestion that his last name, Derrickson, might indicate a connection to the U.S., particularly given similar sounding places like Derrick, Texas. However, without additional information or context about Scott Derrickson, it is not definitively possible to confirm his nationality. Therefore, it is uncertain whether both individuals share the same nationality.",
        "\n\nYes, both Scott Derrickson and Ed Wood are of American nationality. Scott Derrickson, despite having a name common in the U.S., was primarily raised in the United States and has a substantial American background, including directing films like \"The Englishman.\" Ed Wood, being a renowned filmmaker, is definitively American. Therefore, they share the same nationality.",
        "\n\nNo, Scott Derrickson and Ed Wood are not of the same nationality. Ed Wood is an American director known for films like \"Fell into the Fire,\" while Scott Derrickson is a Polish director, active in the video game industry, notably on projects like \"The Witcher 3.\"",
        "\n\nScott Derrickson and Ed Wood were both of American nationality."
    ]
}
```


#### 📊 Output

The returned value `race_score` is a Python dictionary containing two keys:

* **`race_score`** – the final hallucination score (a higher value indicates a higher likelihood of hallucination)
* **`components`** – a detailed breakdown of the four diagnostic components used by RACE:

  * `consistency`: reasoning consistency ($S_{CC}$)
  * `sindex`: answer uncertainty ($S_{AA}$)
  * `cross_uncertainty`: reasoning–answer alignment ($S_{CA}$)
  * `entity_consistency`: reasoning internal coherence (for LRMs, $S_{Coh}$)
  * `uncertainty`: data for computing CoT weights

For example, when running any of the above examples, the result of `race_score` is as follows:

```python
{
    "race_score": 4.276308680521714,
    "components": {
        "uncertainty": {
            "final_answer_entropy": [...],
            "attention": [...]
        },
        "cross_uncertainty": 0.8314897654094036,
        "consistency": 0.9387737315985416,
        "entity_consistency": 0.7142857142857143,
        "sindex": 1.7917594692280547
    }
}
```


You can also use the function **`batch_calculate_race`** to evaluate multiple examples at once.
Both its **input** and **output** are lists, where each element corresponds exactly to the inputs and outputs of `calculate_race_score`.

```python
results = race_scorer.batch_calculate_race(main_data_list, sample_data_list)
```

Each element in `results` is a dictionary with the same structure as the single-sample output shown above.

> ⚠️ **Note:** The current implementation does **not** include batch inference or inference engine optimization.
> Therefore, the runtime efficiency of RACE has not yet reached its full potential.

> 💡 **Note:** We recommend running RACE on at least one 80 GB GPU for efficient CoT extraction and consistency evaluation. 

### 🔧 Released CoT Extractor

We have released the **CoT Extractor** used in the main experiments of our paper at
👉 [**huggingface: bebr2/RACE-CoT-Extractor-Llama-8B**](https://huggingface.co/bebr2/RACE-CoT-Extractor-Llama-8B).

This model is responsible for distilling concise and faithful reasoning chains from raw thinking traces before hallucination evaluation.
The **training code and dataset** can be found in [`./train`](./train).


### 🚀 Pipeline

We provide a complete pipeline from generation to evaluation. Run the setup script:

```bash
source run.sh
```

### 📁 Provided Files

Under `./modeloutput`, we include the entire outputs from **Qwen3-14B** on the **HotpotQA** dataset:

* `judge.json`: Hallucination labels (`1` indicates hallucination).
* `result.json`, `sample_result.json`: The model's main output and sampled output, respectively.
* `summary_result.json`, `summary_sample_result.json`: Outputs processed by the CoT Extractor.

You can pull them using `Git LFS` and quickly run `main.py` (see `run.sh` for parameter settings) to reproduce our results.

## Citations

```
@misc{wang2025jointevaluationanswerreasoning,
      title={Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models}, 
      author={Changyue Wang and Weihang Su and Qingyao Ai and Yiqun Liu},
      year={2025},
      eprint={2506.04832},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.04832}, 
}
```
