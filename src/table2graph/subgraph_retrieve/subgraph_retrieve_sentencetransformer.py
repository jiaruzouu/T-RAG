import pdb
import json
import argparse
import re
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
import math
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_column_type(column_values):
    """Efficiently determine whether a column is 'real' (numeric) or 'text'."""
    num_count = sum(
        1 for val in column_values
        if isinstance(val, (int, float)) or re.match(r"^\d+[\d,]*\.?\d*$", str(val))
    )
    return "real" if num_count / len(column_values) > 0.5 else "text"


def linearize_table(table):
    caption = table.get("caption", "")
    table_data = table.get("table", {})
    header = table_data.get("header", [])
    rows = table_data.get("rows", [])
    header_str = " | ".join(header)
    rows_str = " ".join(" | ".join(row) for row in rows if row)
    table_str = f"Caption: {caption} | Header: {header_str} | Content: {rows_str}"
    return table_str

# With SentenceTransformer, each table is encoded as a single embedding.
def aggregate_table_representation(encodings):
    return encodings["table_embedding"]

def build_similarity_matrix(processed_encodings, similarity_threshold=0.3):
    table_indices = []
    representations = []
    for table_idx, encodings in processed_encodings.items():
        rep = aggregate_table_representation(encodings)  # Already on GPU.
        table_indices.append(table_idx)
        representations.append(rep)
    R = torch.stack(representations, dim=0)
    R_norm = F.normalize(R, p=2, dim=1)
    S = R_norm @ R_norm.T
    S = torch.where(S < similarity_threshold, torch.zeros_like(S), S)
    return S, R_norm, table_indices

def build_transition_matrix(S):
    row_sum = S.sum(dim=1, keepdim=True)
    P = S / (row_sum + 1e-10)
    return P

def compute_personalization_vector(query, sentence_model, R_norm):
    query_repr = sentence_model.encode(query, convert_to_tensor=True, device=device)
    query_norm = F.normalize(query_repr, p=2, dim=0)
    sims = (R_norm @ query_norm.unsqueeze(1)).squeeze(1)
    sims = sims.clamp(min=0)
    total = sims.sum()
    if total > 0:
        personalization = sims / total
    else:
        personalization = torch.ones_like(sims) / sims.numel()
    return personalization

def run_pagerank_gpu(P, personalization, alpha=0.85, max_iter=50, tol=1e-6):
    x = personalization.clone()
    for i in range(max_iter):
        x_new = (1 - alpha) * personalization + alpha * (P.T @ x)
        if torch.norm(x_new - x, p=1) < tol:
            x = x_new
            break
        x = x_new
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to process dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="Number of retrieval iterations to perform")
    parser.add_argument("--filter_percentages", type=str, default="10,30,50",
                        help="Comma-separated percentages (e.g., 10,30,50) to keep in each iteration. "
                             "Used if --filter_topks is not provided.")
    parser.add_argument("--filter_topks", type=str, default=None,
                        help="Comma-separated top-k values (e.g., 50,30,10) to keep in each iteration. "
                             "Length should equal num_iterations. If provided, these values override percentages.")
    parser.add_argument("--schema_only", action="store_true",
                        help="Whether to use only the table schema for encoding.")
    parser.add_argument("--headers_only", action="store_true",
                        help="Whether to use only the table headers for encoding.")
    parser.add_argument("--testing_num", type=int, default=100,
                        help="Number of examples to process")
    parser.add_argument("--cluster_embedding_method", type=str, default="contriever",
                        help="Embedding method to use for clustering previously.")

    
    args = parser.parse_args()


    use_topk = False
    if args.filter_topks is not None:
        try:
            filter_topks = [int(x) for x in args.filter_topks.split(",")]
            if len(filter_topks) != args.num_iterations:
                raise ValueError("The number of filter_topks must equal num_iterations.")
            use_topk = True
        except Exception as e:
            raise ValueError("Error parsing filter_topks: " + str(e))
    else:
        try:
            filter_percentages = [float(x) for x in args.filter_percentages.split(",")]
            if len(filter_percentages) != args.num_iterations:
                raise ValueError("The number of filter percentages must equal num_iterations.")
        except Exception as e:
            raise ValueError("Error parsing filter_percentages: " + str(e))

    # Load SentenceTransformer model.
    ## Option 1: all-mpnet-base-v2
    # sentence_model = SentenceTransformer('all-mpnet-base-v2')
    # sentence_model.to(device)
    
    ## Option 2: E5-large
    sentence_model = SentenceTransformer('intfloat/e5-large-v2')
    sentence_model.to(device)

    dataset = args.dataset
    testing_num = args.testing_num
    clustered_table_file = f"./data/{dataset}/{dataset}_clustered_tables_{args.cluster_embedding_method}.jsonl"
    table_file = f"./data/{dataset}/{dataset}_table.jsonl"
    table_match_file = f"./data/{dataset}/{dataset}_table_match.json"
    output_dir = f"./data/{dataset}/"
    output_file = f"{output_dir}{dataset}_retrieved_tables_schema_{testing_num}_{filter_topks[0]}_contriever.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(table_match_file, "r", encoding="utf-8") as f:
        source_sub_table_mapping = json.load(f)

    table_dict = {}
    with open(table_file, "r", encoding="utf-8") as f:
        for line in f:
            table_data = json.loads(line)
            table_dict[table_data["table_idx"]] = table_data

    processed_data = []
    half_retrieve = 0
    total = 0
    idx = 0
    print(f"Schema Only: {args.schema_only}")
    print(f"Headers Only: {args.headers_only}")
    with open(output_file, "a", encoding="utf-8") as output_f:
        with open(clustered_table_file, "r", encoding="utf-8") as f:
            for line in f:
                idx += 1
                if idx > testing_num:
                    break

                processed_encodings = {}
                processed_table_idxs = set()

                clustered_data = json.loads(line)
                table_size = clustered_data["clustered_tables"]["size"]
                source_table_idx = clustered_data["source_table_idx"]
                ground_truth_table_idx = source_sub_table_mapping[str(source_table_idx)]
                query = clustered_data["query"]
                query_label = clustered_data["label"]
                clustered_indices = clustered_data["clustered_tables"]["clustered_tables"]

                matched_tables = [table_dict[idx] for idx in clustered_indices if idx in table_dict]
                processed_data.append({
                    "source_table_idx": source_table_idx,
                    "query": query,
                    "matched_tables": matched_tables
                })

                with torch.no_grad():
                    for table in tqdm(matched_tables, desc="Processing tables"):
                        table_idx = table["table_idx"]
                        caption = table.get("caption", "")
                        if args.schema_only:
                            table_str = f"Table Caption: {caption}. Table Headers: {table['table']['header']}"
                            # table_str = f"Table Headers: {table['table']['header']}"
                        elif args.headers_only:
                            table_str = f"Table Headers: {table['table']['header']}"
                        else:
                            table_str = linearize_table(table)

                        table_embedding = sentence_model.encode(table_str, convert_to_tensor=True, device=device)
                        if table_idx not in processed_table_idxs:
                            processed_table_idxs.add(table_idx)
                            processed_encodings[table_idx] = {
                                "table_idx": table_idx,
                                "source_table_idx": source_table_idx,
                                "test_query": query,
                                "table_id": caption,
                                "table_embedding": table_embedding
                            }

                # --- Iterative Graph-based Ranking via Personalized PageRank ---
                initial_total = len(processed_encodings)
                current_encodings = processed_encodings
                for iteration in trange(args.num_iterations):
                    S, R_norm, table_indices = build_similarity_matrix(current_encodings, similarity_threshold=0.3)
                    P = build_transition_matrix(S)
                    personalization = compute_personalization_vector(query, sentence_model, R_norm)
                    pagerank_scores = run_pagerank_gpu(P, personalization, alpha=0.85, max_iter=50, tol=1e-6)
                    ranked_tables = sorted(zip(table_indices, pagerank_scores.cpu().tolist()),
                                        key=lambda x: x[1], reverse=True)
                    
                    current_count = len(ranked_tables)
                    if use_topk:
                        keep_count = min(current_count, filter_topks[iteration])
                        print(f"Iteration {iteration+1}: Keeping top {keep_count} out of {current_count} tables (top-k metric).")
                    else:
                        keep_percentage = filter_percentages[iteration]
                        keep_count = max(1, math.ceil(current_count * (keep_percentage / 100.0)))
                        print(f"Iteration {iteration+1}: Keeping {keep_count} out of {current_count} tables ({keep_percentage}%).")
                    
                    filtered_table_indices = [table_idx for table_idx, _ in ranked_tables[:keep_count]]
                    current_encodings = {table_idx: current_encodings[table_idx] for table_idx in filtered_table_indices}
                    
                    if len(current_encodings) == 1:
                        break
                
                final_total = len(current_encodings)
                overall_ratio = final_total / initial_total
                print(f"Total tables at start: {initial_total}, Final filtered tables after {iteration+1} iterations: {final_total}, Overall ratio: {overall_ratio:.2f}")
                            
                # Re-run ranking on the final filtered table set for evaluation.
                final_S, final_R_norm, final_table_indices = build_similarity_matrix(current_encodings, similarity_threshold=0.3)
                final_P = build_transition_matrix(final_S)
                final_personalization = compute_personalization_vector(query, sentence_model, final_R_norm)
                final_pagerank_scores = run_pagerank_gpu(final_P, final_personalization, alpha=0.85, max_iter=50, tol=1e-6)
                final_ranked_tables = sorted(zip(final_table_indices, final_pagerank_scores.cpu().tolist()),
                                            key=lambda x: x[1], reverse=True)
                
                final_table_ids = [table_idx for table_idx, score in final_ranked_tables]
                if set(ground_truth_table_idx).issubset(set(final_table_ids)):
                    for rank, (table_idx, score) in enumerate(final_ranked_tables, 1):
                        if table_idx in ground_truth_table_idx:
                            print(f"Final - Rank: {rank}, Table ID: {table_idx}, Score: {score}")
                    half_retrieve += 1
                else:
                    missing = set(ground_truth_table_idx) - set(final_table_ids)
                    print(f"Not all ground truth tables are present. Missing: {missing}")
                total += 1
                print(f"Total Queries Processed: {total}, Half Retrieve Count: {half_retrieve}")

                final_result = {
                    "query": query,
                    "query_label": query_label,
                    "source_table_idx": source_table_idx,
                    "retrieve_sub_table_idx": final_table_ids,
                    "ground_truth_sub_table_idx": ground_truth_table_idx,
                    "retrieved_tables": []
                }

                for rank, (table_idx, score) in enumerate(final_ranked_tables, 1):
                    table_details = table_dict.get(table_idx, {}) 
                    
                    selected_details = {
                        "split": table_details.get("split"),
                        "source_table_idx": table_details.get("source_table_idx"),
                        "table_idx": table_details.get("table_idx"),
                        "caption": table_details.get("caption"),
                        "table": table_details.get("table"),
                    }

                    final_result["retrieved_tables"].append(selected_details)

                output_f.write(json.dumps(final_result) + "\n")