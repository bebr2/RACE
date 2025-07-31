import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract CoT from model outputs")
    parser.add_argument("--model_name", required=True, help="Model name for input data path")
    parser.add_argument("--dataset_name", required=True, help="Dataset name")
    parser.add_argument("--cot_extractor_model_path", required=True, help="Path to the CoT extractor model")
    parser.add_argument("--data_path", required=True, help="Path to the input data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for extracted CoT")
    parser.add_argument("--gpu_ids", default="0", help="CUDA device to use")
    return parser.parse_args()

def generate_response(text, model, tokenizer, sys_prompt):
    # Prepare the messages
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text},
    ]
    generation_config = dict(
        top_k=0,
        top_p=1.0,
        do_sample=False,
        num_beams=1,
        max_new_tokens=2048
    )
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    attention_mask = torch.ones(len(input_text)).to(model.device)
    
    # Generate
    outputs = model.generate(
        torch.tensor([input_text]).to(model.device), 
        attention_mask=attention_mask.unsqueeze(0), 
        pad_token_id=tokenizer.eos_token_id, 
        **generation_config
    )
    
    # Decode
    response = tokenizer.decode(outputs[0][len(input_text):], skip_special_tokens=True)
    return response

def main():
    args = parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.cot_extractor_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.cot_extractor_model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    # Load system prompt
    with open("datasum_prompt.txt") as f:
        sys_prompt = f.read()
    
    # Set up paths
    data_path = os.path.join(args.data_path, args.dataset_name, args.model_name)
    output_path = args.output_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Load dataset
    dataset = json.load(open(os.path.join(data_path, "result.json"), "r"))
    
    # Load sample dataset
    sample_dataset += json.load(open(os.path.join(data_path, "sample_result.json"), "r"))
    
    assert len(sample_dataset) == len(dataset)
    
    result = []
    sample_result = []
    
    # Process all data
    for i in tqdm(range(len(dataset))):
        question = dataset[i]['question']
        final_answer = dataset[i]["final_answer"].strip()
        cots = []
        think = dataset[i]["think"]
        
        if final_answer:
            cot = generate_response(
                f"## Question\n{question}\n\n## Thought\n{think}\n\n## Final Answer\n{final_answer}",
                model, tokenizer, sys_prompt
            )
            cots = cot.split("[STEP]")
            cots = [c.strip() for c in cots if c.strip()]
        
        result.append({
            "question": question,
            "cots": cots,
        })
        
        # Process sample data
        cotss = []
        for j in range(len(sample_dataset[i]["final_answer"])):
            final_answer = sample_dataset[i]["final_answer"][j].strip()
            think = sample_dataset[i]["think"][j]
            if not final_answer:
                cotss.append([])
            else:
                cot = generate_response(
                    f"## Question\n{question}\n\n## Thought\n{think}\n\n## Final Answer\n{final_answer}",
                    model, tokenizer, sys_prompt
                )
                cots = cot.split("[STEP]")
                cots = [c.strip() for c in cots if c.strip()]
                cotss.append(cots)
        
        sample_result.append({
            "question": question,
            "cots": cotss,
        })
    
    # Save results
    with open(os.path.join(output_path, "summary_result.json"), "w+") as f:
        json.dump(result, f, indent=2)
    
    with open(os.path.join(output_path, "summary_sample_result.json"), "w+") as f:
        json.dump(sample_result, f, indent=2)

if __name__ == "__main__":
    main()
