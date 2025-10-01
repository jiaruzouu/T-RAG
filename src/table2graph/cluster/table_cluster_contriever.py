import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import spacy
from collections import defaultdict
import json
import pdb
from tqdm import tqdm
import argparse
import os 

# -----------------------
# Contriever Wrapper
# -----------------------
def mean_pooling(token_embeddings, attention_mask):
    """
    Perform mean pooling on token embeddings.
    """
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return sentence_embeddings

class ContrieverWrapper:
    """
    A wrapper to use Facebook's Contriever for sentence embedding.
    This class mimics the interface of SentenceTransformer.
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever')
        self.model.eval()  # Set to evaluation mode

    def encode(self, sentences):
        """
        Encode a list of sentences into embeddings using Contriever.
        """
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        return embeddings.cpu().numpy()

# -----------------------
# Global objects
# -----------------------
# Load spaCy model for structure-based feature extraction
nlp = spacy.load("en_core_web_sm")
# Load Contriever model for semantic similarity
model = ContrieverWrapper()

# -----------------------
# Utility functions
# -----------------------

def extract_structure_features(sentence):
    """
    Extract structural features from a sentence.
    Returns a numpy array of features.
    """
    doc = nlp(sentence)
    token_count = len(doc)
    token_lengths = [len(token.text) for token in doc if not token.is_punct]
    avg_token_length = np.mean(token_lengths) if token_lengths else 0.0
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    punctuation_count = sum(1 for token in doc if token.is_punct)

    return np.array([
        token_count, avg_token_length,
        pos_counts["NOUN"], pos_counts["VERB"], pos_counts["ADJ"],
        punctuation_count
    ])

def cluster_sentences(sentences, n_clusters=5):
    """
    Clusters sentences based on three different feature representations:
      - structure (using spaCy‐extracted features)
      - TFIDF (bag‐of-words)
      - semantic (Contriever embeddings)
    Returns:
      kmeans_models: a dict mapping metric names to their KMeans model,
      features_dict: a dict mapping metric names to the feature arrays,
      sentence_indices: a dict mapping metric names to a dict {cluster_id: [sentence_indices]}.
    """
    print("Extracting structural features...")
    struct_features = np.array([extract_structure_features(sent) for sent in tqdm(sentences)])
    
    print("Computing TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    print("Computing semantic embeddings using Contriever...")
    semantic_embeddings = model.encode(sentences)
    
    features_dict = {
        "structure": struct_features,
        "TFIDF": tfidf_matrix.toarray(),
        "semantic": semantic_embeddings
    }
    
    kmeans_models = {}
    sentence_indices = {}
    print(f"Clustering sentences into {n_clusters} clusters for each metric...")
    for metric, features in features_dict.items():
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features)
        kmeans_models[metric] = kmeans
        
        # Build dictionary mapping cluster id to the list of sentence indices in that cluster.
        cluster_dict = defaultdict(list)
        for idx, cluster_id in enumerate(kmeans.labels_):
            cluster_dict[cluster_id].append(idx)
        sentence_indices[metric] = cluster_dict

    return kmeans_models, features_dict, sentence_indices

def select_typical_sentences(sentences, features_dict, kmeans_models, k=3):
    """
    For each metric and for each cluster, select k sentences that are most
    similar to the cluster centroid.
    Returns a dict mapping metric names to another dict {cluster_id: [sentence, ...]}.
    """
    typical_sentences = {metric: {} for metric in features_dict.keys()}
    print(f"Selecting {k} typical sentences per cluster...")
    for metric, features in tqdm(features_dict.items()):
        kmeans = kmeans_models[metric]
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
    
        # For TFIDF, normalize for cosine similarity
        if metric == "TFIDF":
            features = normalize(features)
            cluster_centers = normalize(cluster_centers)
    
        for cluster_id in range(kmeans.n_clusters):
            # Get indices of sentences in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_features = features[cluster_indices]
            centroid = cluster_centers[cluster_id].reshape(1, -1)
    
            # Compute cosine similarities between each sentence in the cluster and the centroid
            similarities = cosine_similarity(cluster_features, centroid).flatten()
            # Select the top-k indices (using argsort)
            top_k_indices = cluster_indices[np.argsort(similarities)[-k:]]
            typical_sentences[metric][cluster_id] = [sentences[i] for i in top_k_indices]
    
    return typical_sentences

def compute_similarity(new_query, typical_embeddings):
    """
    Compute similarity between a new query and a set of typical sentence embeddings
    (one per cluster) using cosine similarity.
    Returns a dict mapping metric -> dict of {cluster_id: average similarity score}.
    """
    scores = {}
    sem_new = model.encode([new_query])
    for metric, cluster_data in typical_embeddings.items():
        scores[metric] = {}
        for cluster_id, emb in cluster_data.items():
            scores[metric][cluster_id] = np.mean(cosine_similarity(sem_new, emb)[0])
    return scores

def find_best_cluster(new_query, typical_embeddings):
    """
    For a given new query, determine the best matching cluster (i.e. with highest similarity)
    for each metric.
    Returns a dict mapping metric -> best cluster id.
    """
    scores = compute_similarity(new_query, typical_embeddings)
    best_cluster = {}
    for metric, cluster_scores in scores.items():
        best_cluster_id = max(cluster_scores, key=cluster_scores.get)
        best_cluster[metric] = best_cluster_id
    return best_cluster

# -----------------------
# Pipeline functions
# -----------------------

def process_dataset(sentences, n_clusters=10, k=100):
    """
    Given a list of sentences, run clustering and select typical sentences.
    Returns:
      kmeans_models, features_dict, sentence_indices, typical_embeddings.
    """
    kmeans_models, features_dict, sentence_indices = cluster_sentences(sentences, n_clusters=n_clusters)
    typical_sents = select_typical_sentences(sentences, features_dict, kmeans_models, k=k)
    typical_embeddings = {metric: {} for metric in typical_sents.keys()}
    print("Precomputing typical sentence embeddings using Contriever...")
    for metric, clusters in typical_sents.items():
        for cluster_id, sent_list in clusters.items():
            typical_embeddings[metric][cluster_id] = model.encode(sent_list)
    return kmeans_models, features_dict, sentence_indices, typical_embeddings

def evaluate_queries(query_data, source_ids, kmeans_models, features_dict, sentence_indices, typical_embeddings):
    """
    Evaluate the given clustering setup on a list of testing queries.
    For each query, we look for table entries (via source_ids) that match the query’s source_table_idx.
    For each metric, if all matching entries fall in a unique cluster and the prediction (from typical embeddings)
    matches that cluster, we count it as correct.
    
    Returns a dictionary of counters (per metric and total) and a dictionary (total_tables_shared)
    mapping each query (by a unique key) to the set of unique table indices retrieved.
    """
    counters = {
        "structure_correct": 0,
        "TFIDF_correct": 0,
        "semantic_correct": 0,
        "total_correct": 0,
        "structure_cluster_mismatch": 0,
        "TFIDF_cluster_mismatch": 0,
        "semantic_cluster_mismatch": 0,
        "total": 0
    }
    total_tables_shared = {}  # key: query key, value: dict with clustered_tables (set) and size
    for query in tqdm(query_data, desc="Evaluating queries"):
        counters["total"] += 1
        new_query = query["query"]
        ground_truth_id = query["source_table_idx"]
        # Find all indices in the dataset that match the current table (source_table_idx)
        matching_indices = [i for i, tid in enumerate(source_ids) if tid == ground_truth_id]
    
        if len(matching_indices) == 0:
            # If no matching table is found, count a mismatch for all metrics.
            counters["structure_cluster_mismatch"] += 1
            counters["TFIDF_cluster_mismatch"] += 1
            counters["semantic_cluster_mismatch"] += 1
            continue
    
        actual_clusters = {}
        for metric in features_dict.keys():
            labels = [kmeans_models[metric].labels_[i] for i in matching_indices]
            if len(set(labels)) != 1:
                # The table entries for this query are split among different clusters.
                if metric == "structure":
                    counters["structure_cluster_mismatch"] += 1
                elif metric == "TFIDF":
                    counters["TFIDF_cluster_mismatch"] += 1
                elif metric == "semantic":
                    counters["semantic_cluster_mismatch"] += 1
            else:
                actual_clusters[metric] = labels[0]
    
        # Predict the best cluster for the query based on typical embeddings.
        prediction = find_best_cluster(new_query, typical_embeddings)
    
        correct_flag = False
        clustered_tables = set()
        if "structure" in actual_clusters and prediction["structure"] == actual_clusters["structure"]:
            counters["structure_correct"] += 1
            correct_flag = True
            clustered_tables.update(sentence_indices["structure"][actual_clusters["structure"]])
        if "TFIDF" in actual_clusters and prediction["TFIDF"] == actual_clusters["TFIDF"]:
            counters["TFIDF_correct"] += 1
            correct_flag = True
            clustered_tables.update(sentence_indices["TFIDF"][actual_clusters["TFIDF"]])
        if "semantic" in actual_clusters and prediction["semantic"] == actual_clusters["semantic"]:
            counters["semantic_correct"] += 1
            correct_flag = True
            clustered_tables.update(sentence_indices["semantic"][actual_clusters["semantic"]])
    
        if correct_flag:
            counters["total_correct"] += 1
            # Use a compound key for the query (assume query text combined with source_table_idx is unique)
            query_key = f"{ground_truth_id}_{new_query}"
            total_tables_shared[query_key] = {
                "clustered_tables": clustered_tables,
                "size": len(clustered_tables)
            }
    
    return counters, total_tables_shared

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--n_clusters", type=int, help="number of clusters")
    parser.add_argument("--k", type=int, help="number of typical sentences per cluster")
    args = parser.parse_args()
    
    # --- File paths (adjust as needed) ---
    dataset = args.dataset
    n_clusters = args.n_clusters
    k = args.k
    # For table schema data (with key "table_schema")
    table_schema_file = f"./data/{dataset}/{dataset}_schema.jsonl"
    # For example queries data (with key "example_query")
    example_query_file = f"./data/{dataset}/{dataset}_example_query.jsonl"
    # Testing queries (assumed to have keys "query" and "source_table_idx")
    testing_query_file = f"./data/{dataset}/{dataset}_query.jsonl"
    output_dir = f"./data/{dataset}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/{dataset}_clustered_tables_contriever.jsonl"
    
    print(f"Dataset: {dataset}")
    
    # --- Load table schema data ---
    table_schema_data = []
    with open(table_schema_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                table_schema_data.append(json.loads(line))
    table_schema_sentences = [item["table_schema"] for item in table_schema_data]
    table_schema_source_ids = [item["source_table_idx"] for item in table_schema_data]
    print(f"Loaded {len(table_schema_sentences)} table schema sentences.")
    
    # --- Load example query data ---
    example_query_data = []
    with open(example_query_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                example_query_data.append(json.loads(line))
    example_query_sentences = [item["example_query"] for item in example_query_data]
    example_query_source_ids = [item["source_table_idx"] for item in example_query_data]
    print(f"Loaded {len(example_query_sentences)} example query sentences.")
    
    # --- Load testing queries ---
    query_data = []
    with open(testing_query_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                query_data.append(json.loads(line))
    print(f"Loaded {len(query_data)} testing queries.")
        
    # --- Process and evaluate table schema data ---
    print("\n=== Processing Table Schema Data ===")
    ts_kmeans, ts_features, ts_sentence_indices, ts_typical_embeddings = process_dataset(table_schema_sentences, n_clusters, k)
    print("Evaluating Table Schema Data...")
    ts_counters, ts_total_tables_shared = evaluate_queries(query_data, table_schema_source_ids, 
                                                           ts_kmeans, ts_features, ts_sentence_indices, ts_typical_embeddings)
    
    # --- Process and evaluate example query data ---
    print("\n=== Processing Example Query Data ===")
    eq_kmeans, eq_features, eq_sentence_indices, eq_typical_embeddings = process_dataset(example_query_sentences, n_clusters, k)
    print("Evaluating Example Query Data...")
    eq_counters, eq_total_tables_shared = evaluate_queries(query_data, example_query_source_ids, 
                                                           eq_kmeans, eq_features, eq_sentence_indices, eq_typical_embeddings)
    
    
    overall_total_correct = 0
    overall_tables_shared = {}
    for query in query_data:
        query_key = f"{query['source_table_idx']}_{query['query']}"
        union_tables = set()
        
        # If the query was evaluated as correct in either pipeline, use the evaluation results.
        if query_key in ts_total_tables_shared or query_key in eq_total_tables_shared:
            if query_key in ts_total_tables_shared:
                union_tables.update(ts_total_tables_shared[query_key]["clustered_tables"])
            if query_key in eq_total_tables_shared:
                union_tables.update(eq_total_tables_shared[query_key]["clustered_tables"])
            overall_total_correct += 1
        else:
            # For queries not evaluated as correct, compute predictions from both pipelines.
            pred_ts = find_best_cluster(query["query"], ts_typical_embeddings)
            pred_eq = find_best_cluster(query["query"], eq_typical_embeddings)
            
            for metric in ts_sentence_indices.keys():
                if metric in pred_ts:
                    union_tables.update(ts_sentence_indices[metric][pred_ts[metric]])
            for metric in eq_sentence_indices.keys():
                if metric in pred_eq:
                    union_tables.update(eq_sentence_indices[metric][pred_eq[metric]])
        
        overall_tables_shared[query_key] = {"clustered_tables": union_tables, "size": len(union_tables)}
    
    # --- Print results ---
    print("\n==================== Evaluation Results ====================")
    print("\n--- Table Schema Data ---")
    print("Structure Correct:", ts_counters["structure_correct"])
    print("TFIDF Correct:", ts_counters["TFIDF_correct"])
    print("Semantic Correct:", ts_counters["semantic_correct"])
    print("Total Correct:", ts_counters["total_correct"])
    print("Structure Cluster Mismatch:", ts_counters["structure_cluster_mismatch"])
    print("TFIDF Cluster Mismatch:", ts_counters["TFIDF_cluster_mismatch"])
    print("Semantic Cluster Mismatch:", ts_counters["semantic_cluster_mismatch"])
    print("Total Queries Processed:", ts_counters["total"])
    total_tables_ts = sum(v["size"] for v in ts_total_tables_shared.values())
    avg_tables_per_query_ts = total_tables_ts / ts_counters["total_correct"] if ts_counters["total_correct"] > 0 else 0
    print("Average Clustered Tables per Query:", f"{avg_tables_per_query_ts:.2f}")
    
    print("\n--- Example Query Data ---")
    print("Structure Correct:", eq_counters["structure_correct"])
    print("TFIDF Correct:", eq_counters["TFIDF_correct"])
    print("Semantic Correct:", eq_counters["semantic_correct"])
    print("Total Correct:", eq_counters["total_correct"])
    print("Structure Cluster Mismatch:", eq_counters["structure_cluster_mismatch"])
    print("TFIDF Cluster Mismatch:", eq_counters["TFIDF_cluster_mismatch"])
    print("Semantic Cluster Mismatch:", eq_counters["semantic_cluster_mismatch"])
    print("Total Queries Processed:", eq_counters["total"])
    total_tables_eq = sum(v["size"] for v in eq_total_tables_shared.values())
    avg_tables_per_query_eq = total_tables_eq / eq_counters["total_correct"] if eq_counters["total_correct"] > 0 else 0
    print("Average Clustered Tables per Query:", f"{avg_tables_per_query_eq:.2f}")
    
    print("\n--- Overall Evaluation ---")
    print("Overall Total Correct:", overall_total_correct)
    total_tables_overall = sum(v["size"] for v in overall_tables_shared.values())
    avg_tables_per_query_overall = total_tables_overall / overall_total_correct if overall_total_correct > 0 else 0
    print("Average Clustered Tables per Query (Overall):", f"{avg_tables_per_query_overall:.2f}")
    
    with open(output_file, "w") as out_f:
        for query in query_data:  
            query_key = f"{query['source_table_idx']}_{query['query']}"
            if query_key in overall_tables_shared:
                query_result = overall_tables_shared[query_key]
                query_result["clustered_tables"] = list(query_result["clustered_tables"])
                store_structure = {
                    "source_table_idx": query["source_table_idx"],
                    "query": query["query"],
                    "label": query["label"],
                    "clustered_tables": query_result
                }
                out_f.write(json.dumps(store_structure) + "\n")
