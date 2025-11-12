import os
import json
import numpy as np
from RACE import RACEScorer


def load_json_files(directory, prefix):
    with open(os.path.join(directory, f"{prefix}.json"), 'r') as f:
        data = json.load(f)
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate RACE scores for LLM outputs')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--data_dir', type=str, default='/path/to/modeloutput', 
                        help='Base directory containing model outputs')
    parser.add_argument('--output', type=str, default='./race_scores.json', 
                        help='Output file path')
    parser.add_argument('--embedding_model', type=str, default='/path/to/all-MiniLM-L6-v2', 
                        help='Path to embedding model')
    parser.add_argument('--nli_model', type=str, default='/path/to/deberta-v3-large-mnli', 
                        help='Path to NLI model')
    parser.add_argument('--llm_model', type=str, default='/path/to/llama-model', 
                        help='Path to LLM model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for calculations')
    parser.add_argument('--sindex_threshold', type=float, default=0.9, help='Threshold for SIndex clustering')
    
    args = parser.parse_args()
    
    # Construct paths
    data_path = os.path.join(args.data_dir, args.dataset, args.model)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    
    # Sort files to ensure proper order
    
    # Load main dataset
    main_data = load_json_files(data_path, 'result')
    
    main_sum_data = load_json_files(data_path, 'summary_result')
    
    for i in range(len(main_data)):
        main_data[i]['cots'] = main_sum_data[i]['cots']
    
    # Load sample dataset
    sample_data = load_json_files(data_path, 'sample_result')
    
    
    sample_sum_data = load_json_files(data_path, 'summary_sample_result')
    
    for i in range(len(sample_data)):
        sample_data[i]['cots'] = sample_sum_data[i]['cots']
    
    # Initialize RACE scorer
    race_scorer = RACEScorer(
        embedding_model_path=args.embedding_model,
        nli_model_path=args.nli_model,
        llm_model_path=args.llm_model,
        use_gpu=args.gpu,
        sindex_threshold=args.sindex_threshold
    )
    
    # Calculate RACE scores
    race_scores = race_scorer.batch_calculate_race(main_data, sample_data)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(race_scores, f, indent=2)
    
    print(f"RACE scores saved to {args.output}")
    
    # Display aggregate statistics
    scores_only = [score['race_score'] for score in race_scores]
    print(f"Average RACE score: {np.mean(scores_only):.4f}")
    print(f"Median RACE score: {np.median(scores_only):.4f}")
    print(f"Min RACE score: {np.min(scores_only):.4f}")
    print(f"Max RACE score: {np.max(scores_only):.4f}")


if __name__ == "__main__":
    main()
