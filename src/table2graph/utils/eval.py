import json

def acc_at_k(ground_truth, retrieved, k):
    """
    Returns acc@k.
    If any of the ground truth items is found within the top k retrieved results, return 1; otherwise, return 0.
    """
    top_k = set(retrieved[:k])
    return 1 if all(item in top_k for item in ground_truth) else 0

def recall_at_k(ground_truth, retrieved, k):
    """
    Computes recall@k, which is the fraction of ground truth items that appear in the top k retrieved results.
    """
    top_k = set(retrieved[:k])
    if len(ground_truth) == 0:
        return 0
    relevant_found = len(set(ground_truth) & top_k)
    return relevant_found / len(ground_truth)

datasets = ['tabfact']

for dataset in datasets:
    with open(f'./data/contriever/{dataset}/{dataset}_retrieved_tables_schema_100_50_contriever.jsonl', 'r') as f2:
        data = [json.loads(line) for line in f2]
    
    for k in [10, 20, 50]:
        acc = 0
        recall = 0
        n = 0
        for d in data:
            ground_truth = d['ground_truth_sub_table_idx']
            retrieved = d['retrieve_sub_table_idx']
            if len(ground_truth) == 0:
                continue
            acc += acc_at_k(ground_truth, retrieved, k)
            recall += recall_at_k(ground_truth, retrieved, k)
            n += 1
        print(f"Dataset: {dataset}, k: {k}, acc@{k}: {acc/n:.4f}, recall@{k}: {recall/n:.4f}")
