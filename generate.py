import os
import sys
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., ds7b). Please name the thinking model with a name that ends with the letter b")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default="./modeloutput", help="Directory to save model outputs")
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

model_name = args.model_name
dataset_name = args.dataset_name
model_path = args.model_path

output_path = f"./{args.output_dir}/{dataset_name}/{model_name}"
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2")

# Load dataset
if dataset_name == "HotpotQA":
    dataset = json.load(open("./dataset/HotpotQA/hotpot_dev_fullwiki_v1.json"))
elif dataset_name == "NQ":
    dataset = []
    with open("./dataset/NQ-open.efficientqa.dev.1.1.jsonl") as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
elif dataset_name == "squad":
    dataset = json.load(open("./dataset/squad.json"))
elif dataset_name == "triviaqa":
    dataset = json.load(open("./dataset/triviaqa.json"))

with open("./cot_prompt.txt") as f:
    cot_prompt = f.read()

def generate_response(text, greedy=True):
    # Prepare the messages with think tokens
    if "cot" in model_name:
        messages = [
            {"role": "user", "content": cot_prompt.format(text)}
        ]
    else:
        messages = [
            {"role": "user", "content": text}
        ]
    
    if greedy:
        generation_config = dict(
                            top_k=0,
                            top_p=1.0,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=2048)
    else:
        generation_config = dict(
                            top_p=0.95,
                            temperature=1,
                            do_sample=True,
                            num_beams=1,
                            num_return_sequences=5,
                            max_new_tokens=2048)
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    attention_mask = torch.ones(len(input_text)).to(model.device)
    
    # Generate
    outputs = model.generate(torch.tensor([input_text]).to(model.device), attention_mask=attention_mask.unsqueeze(0), pad_token_id=tokenizer.eos_token_id, **generation_config)
    
    # Decode
    if greedy:
        response = tokenizer.decode(outputs[0][len(input_text):], skip_special_tokens=True)
    else:
        response = [tokenizer.decode(output[len(input_text):], skip_special_tokens=True) for output in outputs]

    return response

result = []
sample_result = []

for i in tqdm(range(len(dataset))):
    question = dataset[i]['question']
    answer = dataset[i]['answer']
    output = generate_response(question)
    
    if model_name.endswith("b") or "thinking" in model_name:
        think = output.split('</think>')[0].split("<think>")[-1]
        if "</think>" in output:
            final_answer = output.split('</think>')[-1]
        else:
            continue
    elif "cot" in model_name:
        # cot output
        think = output.split('Answer:')[0]
        if "Answer:" in output:
            final_answer = output.split('Answer:')[-1]
        else:
            continue
    elif "chat" in model_name or "direct" in model_name:
        # direct chat output
        think = output
        final_answer = output

    result.append({
        "question": question,
        "answer": answer,
        "output": output,
        "think": think,
        "final_answer": final_answer
    })

    outputs = generate_response(question, greedy=False)
    answers = []
    thinks = []
    for output in outputs:
        if model_name.endswith("b") or "thinking" in model_name:
            think = output.split('</think>')[0].split("<think>")[-1]
            if "</think>" in output:
                final_answer = output.split('</think>')[-1]
            else:
                final_answer = ""
        elif "cot" in model_name:
            think = output.split('Answer:')[0]
            if "Answer:" in output:
                final_answer = output.split('Answer:')[-1]
            else:
                final_answer = ""
        elif "chat" in model_name:
            think = output
            final_answer = output
        answers.append(final_answer)
        thinks.append(think)

    sample_result.append({
        "index": i, 
        "question": question,
        "answer": answer,
        "output": outputs,
        "think": thinks,
        "final_answer": answers
    })

with open(f"{output_path}/result.json", "w+") as f:
    json.dump(result, f, indent=2)
    
with open(f"{output_path}/sample_result.json", "w+") as f:
    json.dump(sample_result, f, indent=2)
