import json

with open("../datasum_prompt.txt") as f:
    sys_prompt = f.read()

def read_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = []
    for d in data:
        question = d["question"]
        think = d["think"]
        final_answer = d["final_answer"]
        cots = d["sum"]
        s = f"## Question\n{question}\n\n## Thought\n{think}\n\n## Final Answer\n{final_answer}"
        new_data.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": s},
            {"role": "assistant", "content": "[STEP] " + "\n[STEP] ".join(cots)}
        ])
    return new_data[:2000]

def llama_instruct_preprocess(msg, tokenizer, max_len, ignore_index=-100):
    ids = tokenizer.apply_chat_template(
        msg,
        tokenize=True,
        add_generation_prompt=False,
    )
    start = 0
    for i in range(len(ids) - 1, -1, -1):
        if ids[i-3:i+1] == [128006, 78191, 128007, 271]:
            start = i + 1
            break
    labels = [ignore_index] * start + ids[start:]
    
    return ids, labels