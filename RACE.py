
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import numpy as np
import json
import torch
import torch.nn.functional as F
import spacy
from nltk import sent_tokenize
from itertools import combinations
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from modelscope import AutoTokenizer as MSAutoTokenizer
from modelscope import AutoModel as MSAutoModel

try:
    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
except ImportError:
    print("SelfCheckGPT not installed. Some features will be disabled.")
    print("Install with: pip install selfcheckgpt")


class RACEScorer:
    """
    RACE (Reliability Assessment with Consistency Evaluation) scorer.
    
    This class calculates reliability scores for LLM outputs based on
    multiple components: uncertainty, self-consistency, and entity consistency.
    """
    
    def __init__(self, 
                 embedding_model_path="/path/to/all-MiniLM-L6-v2", 
                 nli_model_path="/path/to/deberta-v3-large-mnli",
                 llm_model_path="/path/to/llama-model",
                 use_gpu=True,
                 sindex_threshold=0.9):
        """
        Initialize RACE scorer.
        
        Args:
            embedding_model_path: Path to sentence embedding model
            nli_model_path: Path to NLI model for contradiction detection
            llm_model_path: Path to LLM for uncertainty estimation
            use_gpu: Whether to use GPU for computation
            sindex_threshold: Threshold for SIndex clustering
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.sindex_threshold = sindex_threshold
        
        print("Loading models...")
        # Load embedding model
        try:
            self.embed_tokenizer = MSAutoTokenizer.from_pretrained(embedding_model_path)
            self.embed_model = MSAutoModel.from_pretrained(embedding_model_path).to(self.device)
            print(f"Embedding model loaded from {embedding_model_path}")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            print("Will use fallback for embedding calculations")
            self.embed_tokenizer = None
            self.embed_model = None
        
        # Load NLI model for SCG
        try:
            self.selfcheck_nli = SelfCheckNLI(nli_model=nli_model_path, device=self.device)
            print(f"NLI model loaded from {nli_model_path}")
        except Exception as e:
            print(f"Failed to load NLI model: {e}")
            print("Will skip contradiction detection")
            self.selfcheck_nli = None
        
        # Load LLM for uncertainty
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path, 
                device_map="auto" if use_gpu else None,
                torch_dtype=torch.float32
            )
            print(f"LLM model loaded from {llm_model_path}")
        except Exception as e:
            print(f"Failed to load LLM model: {e}")
            print("Will skip uncertainty calculation")
            self.llm_tokenizer = None
            self.llm_model = None
            
        # Load spaCy for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_trf")
            print("SpaCy model loaded for entity extraction")
        except Exception as e:
            print(f"Failed to load spaCy model: {e}")
            print("Will skip entity extraction")
            self.nlp = None

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings"""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sentences, q):
        """Get embeddings for sentences with question context"""
        if self.embed_model is None or self.embed_tokenizer is None:
            # Fallback to simpler approach if model not available
            return [[1.0] * 384 for _ in sentences]  # Dummy embeddings
            
        consentences = [f"{q} [SEP] {s}" for s in sentences]
        encoded_input = self.embed_tokenizer(consentences, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.tolist()

    def compute_cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def clustering_algorithm(self, sequences, question, thresh=None):
        """
        Clusters sequences using average distance and cosine similarity.
        
        Args:
            sequences: List of sequences to be clustered
            question: Question for context
            thresh: Distance threshold for merging clusters (default: self.sindex_threshold)
            
        Returns:
            clusters, embeddings dict, number of total sequences
        """
        if thresh is None:
            thresh = self.sindex_threshold
            
        sequences = sorted(list(set(sequences)))
        ebs = self.get_embedding(sequences, question)
        embeddings = {si: eb for si, eb in zip(sequences, ebs)}
        
        clusters = []
        
        for si in sequences:
            new_cluster = [si]
            need_new = True
            
            for i in range(len(clusters)):
                cluster = clusters[i]
                cluster_dis = 0
                
                for c in cluster:
                    emb_si = embeddings[si]
                    emb_c = embeddings[c]
                    
                    cos_sim = self.compute_cosine_similarity(emb_si, emb_c)
                    dis = 1 - cos_sim
                    cluster_dis += dis
                
                avg_dis = cluster_dis / len(cluster)
                
                if avg_dis < 1 - thresh:
                    clusters[i] = clusters[i] + new_cluster
                    need_new = False
                    break

            if need_new:
                clusters.append(new_cluster)
        
        assert len(sequences) == sum(len(c) for c in clusters)
        # print(len(clusters), "clusters")
        return clusters, embeddings, len(sequences)

    def compute_sindex(self, total_points, clusters, embeddings):
        """
        Calculate the SINdex for a given set of clusters and sequences.
        
        Args:
            total_points: Total number of sequences
            clusters: List of clusters
            embeddings: Dictionary mapping sequences to embeddings
            
        Returns:
            SINdex value
        """
        k = len(clusters)
        
        p_i = {}
        cos_sim_ci = {}
        
        for i, cluster in enumerate(clusters):
            p_i[i] = len(cluster) / total_points
            
            total_cos_sim = 0
            num_pairs = 0
            for seq1, seq2 in combinations(cluster, 2):
                emb1 = embeddings[seq1]
                emb2 = embeddings[seq2]
                total_cos_sim += self.compute_cosine_similarity(emb1, emb2)
                num_pairs += 1
            
            if num_pairs > 0:
                cos_sim_ci[i] = total_cos_sim / num_pairs
            else:
                cos_sim_ci[i] = 1  # If the cluster has only one element
        
        p_prime_i = {i: p_i[i] * cos_sim_ci[i] for i in range(k)}
        
        s_index = 0
        for i in range(k):
            s_index += p_prime_i[i] * np.log(p_prime_i[i])
        
        return -s_index

    def calculate_sindex(self, main_answer, sample_answers, question):
        """
        Calculate SIndex score for a main answer compared to sample answers.
        
        Args:
            main_answer: Primary answer to evaluate
            sample_answers: List of alternate answers for comparison
            question: The question that was answered
            
        Returns:
            SIndex score
        """
        all_answers = [main_answer] + sample_answers
        all_answers = [a.strip() if a.strip() else " " for a in all_answers]
        
        clusters, embeddings, total_points = self.clustering_algorithm(all_answers, question)
        # print(f"Clusters: {clusters}")
        # print(f"total_points: {total_points}")
        
        score = self.compute_sindex(total_points, clusters, embeddings)
        return score

    def get_entities(self, text):
        """Extract named entities from text"""
        if self.nlp is None:
            return []
            
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

    def calculate_entity_consistency(self, main_text, cot_text):
        """
        Calculate entity consistency between reasoning and answer.
        
        Args:
            main_text: Main reasoning text
            cot_texts: Chain of thought texts
            
        Returns:
            Entity consistency score
        """
        if self.nlp is None:
            return 0.0
            
        try:
            think_entities = set(self.get_entities(main_text))
            cot_entities = set(self.get_entities(cot_text))
            
            # Clean entities (lowercase, remove leading "the")
            def clean(entities):
                cleaned = set()
                for ent in entities:
                    e = ent.lower()
                    if e.startswith("the "):
                        e = e[4:].strip()
                    cleaned.add(e)
                return cleaned
                
            think_entities = clean(think_entities)
            cot_entities = clean(cot_entities)
            
            # Calculate consistency as proportion of entities in think not in cot
            if think_entities:
                return len(think_entities - cot_entities) / len(think_entities)
            return 0
        except Exception as e:
            print(f"Error in entity consistency calculation: {e}")
            return 0

    def calculate_uncertainty(self, question, cots, final_answer, cross=True):
        """
        Calculate uncertainty of the model in its answer.
        
        Args:
            question: The question being answered
            cots: List of chain-of-thought reasoning steps
            final_answer: The final answer provided
            
        Returns:
            Dictionary with uncertainty metrics
        """
        if self.llm_model is None or self.llm_tokenizer is None or not final_answer:
            return {
                "final_answer_entropy": None,
                "attention": None
            }
            
        if not cots:
            return {
                "final_answer_entropy": None,
                "attention": [1.0] if len(cots) > 0 else None
            }

        try:
            if not cross:
                messages_only_q = [
                    {"role": "user", "content": question},
                ]
                q_id_list = self.llm_tokenizer.apply_chat_template(messages_only_q, add_generation_prompt=True)
                q_id_list += self.llm_tokenizer.encode(f"Thought: ", add_special_tokens=False)
                think_id_list = []

                for cot in cots:
                    think_id_list.append(self.llm_tokenizer.encode(f"{cot}\n", add_special_tokens=False))
                think_id_list[-1] = think_id_list[-1] + self.llm_tokenizer.encode("\n\nAnswer: ", add_special_tokens=False)
                final_answer_id_list = self.llm_tokenizer.encode(f"{final_answer}", add_special_tokens=False)
                
                new_think_id_list = []
                for tl in think_id_list:
                    new_think_id_list += tl
                input_text = q_id_list + new_think_id_list + final_answer_id_list
                    
                think_start = len(q_id_list)-1
                think_end = len(q_id_list)+len(new_think_id_list)-1
                final_answer_start = think_end

                input_ids = torch.tensor([input_text]).to(self.llm_model.device)
                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    outputs = self.llm_model(input_ids, attention_mask=attention_mask, output_attentions=True)
                    attentions = outputs.attentions
                    logits = outputs.logits.cpu()

                    final_answer_logits = logits[0, final_answer_start:-1, :]
                    final_answer_probs = F.softmax(final_answer_logits, dim=-1)
                    final_answer_entropy = torch.sum(torch.special.entr(final_answer_probs), dim=-1).tolist()
                    
                    # Calculate attention from answer to each reasoning step
                    all_layers_attention = torch.stack(attentions, dim=0)
                    avg_attention = torch.mean(torch.mean(all_layers_attention, dim=0), dim=1)
                    
                    cot_attentions = []
                    cot_start_positions = [0]
                    cum_length = 0
                    for i in range(len(think_id_list)):
                        cum_length += len(think_id_list[i])
                        cot_start_positions.append(cum_length)
                    
                    for i in range(len(think_id_list)):
                        start_idx = think_start + cot_start_positions[i]
                        if i < len(think_id_list) - 1:
                            end_idx = think_start + cot_start_positions[i+1]
                        else:
                            end_idx = think_end
                            
                        cot_attention = avg_attention[0, final_answer_start:-1, start_idx:end_idx]
                        avg_cot_attention = torch.mean(torch.mean(cot_attention, dim=1))
                        cot_attentions.append(avg_cot_attention.item())
                    
                    if sum(cot_attentions) > 0:
                        cot_attentions = [att / sum(cot_attentions) for att in cot_attentions]
                return {
                    "final_answer_entropy": final_answer_entropy,
                    "attention": cot_attentions
                }
            else:
                messages_only_q = [
                        {"role": "user", "content": question},
                    ]
                q_id_list = self.llm_tokenizer.apply_chat_template(messages_only_q, add_generation_prompt=True)
                think = "\n".join(cots)
                think_id_list = self.llm_tokenizer.encode(f"Thought: {think}\n\nAnswer: ", add_special_tokens=False)
                final_answer_id_list = self.llm_tokenizer.encode(f"{final_answer}", add_special_tokens=False)
                input_text = q_id_list + think_id_list + final_answer_id_list
                    
                think_start = len(q_id_list)-1
                think_end = len(q_id_list)+len(think_id_list)-1
                final_answer_start = think_end

                input_ids = torch.tensor([input_text]).to(self.llm_model.device)

                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    outputs = self.llm_model(input_ids, attention_mask=attention_mask, output_attentions=True)
                    logits = outputs.logits.cpu()

                    final_answer_logits = logits[0, final_answer_start:-1, :]
                    final_answer_probs = F.softmax(final_answer_logits, dim=-1)
                    final_answer_prob = []
                    for i in range(len(final_answer_probs)):
                        tokenid = final_answer_id_list[i]
                        prob = final_answer_probs[i][tokenid].item()
                        final_answer_prob.append(prob)

                    final_answer_entropy = torch.sum(torch.special.entr(final_answer_probs), dim=-1).tolist()
                    
                return {
                    "final_answer_entropy": final_answer_entropy,
                    "attention": None
                }
        except Exception as e:
            print(f"Error in uncertainty calculation: {e}")
            return {
                "final_answer_entropy": None,
                "attention": None
            }

    def calculate_cross_uncertainty(self, question, cots, sample_answers):
        """
        Calculate uncertainty across different answers using the same reasoning.
        
        Args:
            question: The question being answered
            cots: List of chain-of-thought reasoning steps
            main_answer: The main answer
            sample_answers: List of alternative answers
            
        Returns:
            Average entropy across answers
        """
        if self.llm_model is None or not sample_answers:
            return 10.0  # Default high uncertainty
            
        entropies = []
        
        try:
            for answer in sample_answers:
                if not answer:
                    continue
                    
                result = self.calculate_uncertainty(question, cots, answer)

                if result and result["final_answer_entropy"]:
                    entropies.append(np.mean(result["final_answer_entropy"]))
            # print(f"Entropies: {entropies}")
            if not entropies:
                return 10.0
                    
            return np.mean(entropies)
        except Exception as e:
            print(f"Error in cross uncertainty calculation: {e}")
            return 10.0

    def calculate_consistency_score(self, main_cots, sample_cots_list, uncertainties):
        """
        Calculate consistency scores for chain of thought steps
        
        Args:
            cots: Chain of thought reasoning steps
            uncertainties: Uncertainty values from LLM
            
        Returns:
            Consistency score
        """
        if not main_cots or not sample_cots_list:
            return 10.0  # Default high uncertainty
            
        try:
            if uncertainties and uncertainties["attention"]:
                weights = uncertainties["attention"]
            else:
                weights = [1/len(main_cots) for _ in range(len(main_cots))]
                
            if self.selfcheck_nli:
                # Get contradiction scores between CoT steps
                cot_consistency = self.selfcheck_nli.predict(
                    sentences=main_cots,
                    sampled_passages=["\n".join(sample_cots) for sample_cots in sample_cots_list]
                ).tolist()
                
                # Weight by attention
                score = 0
                for w, consistency in zip(weights, cot_consistency):
                    score += w * consistency
                    
                return score
            else:
                # Fallback if NLI model not available
                return 5.0
        except Exception as e:
            print(f"Error in consistency calculation: {e}")
            return 5.0

    def calculate_race_score(self, main_data, sample_data):
        """
        Calculate the overall RACE score for a given main output and sample outputs.
        
        Args:
            main_data: Dictionary with main output data (question, final_answer, think, cots)
            sample_data: Dictionary with sample outputs data (final_answer, cots)
            
        Returns:
            Dictionary with components and final RACE score
        """
        question = main_data.get("question", "")
        main_answer = main_data.get("final_answer", "")
        main_reasoning = main_data.get("think", "")
        main_cots = main_data.get("cots", [])
        
        sample_answers = sample_data.get("final_answer", [])
        sample_cots = sample_data.get("cots", [])
        
        # Calculate uncertainty component
        uncertainty = self.calculate_uncertainty(question, main_cots, main_answer, False)
        
        # Calculate cross-uncertainty (with other answers)
        cross_uncertainty = self.calculate_cross_uncertainty(
            question, main_cots, sample_answers
        )
        
        # Calculate consistency component
        consistency = self.calculate_consistency_score(main_cots, sample_cots, uncertainty)
        
        # Calculate entity consistency
        entity_consistency = self.calculate_entity_consistency(main_reasoning, "\n".join(main_cots))
        
        # Calculate SIndex score
        sindex = self.calculate_sindex(main_answer, sample_answers, question)
        
        # Combine components into overall RACE score
        race_score = consistency + cross_uncertainty + sindex
        if entity_consistency > 0:
            race_score += entity_consistency
            
        return {
            "race_score": race_score,
            "components": {
                "uncertainty": uncertainty,
                "cross_uncertainty": cross_uncertainty,
                "consistency": consistency,
                "entity_consistency": entity_consistency,
                "sindex": sindex
            }
        }

    def batch_calculate_race(self, main_dataset, sample_dataset):
        """
        Calculate RACE scores for a batch of data
        
        Args:
            main_dataset: List of main output data
            sample_dataset: List of sample output data
            
        Returns:
            List of RACE scores and components
        """
        results = []
        
        for main_item, sample_item in tqdm(zip(main_dataset, sample_dataset), 
                                           total=min(len(main_dataset), len(sample_dataset)),
                                           desc="Calculating RACE scores"):
            score = self.calculate_race_score(main_item, sample_item)
            results.append(score)
            
        return results


def load_json_files(directory, prefix):
    with open(os.path.join(directory, f"{prefix}.json"), 'r') as f:
        data = json.load(f)
    return data


def main():
    """
    Example of how to use the RACE scorer.
    
    Usage:
        python RACE.py --dataset HotpotQA --model qwen7b --output ./race_results/
    """
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
