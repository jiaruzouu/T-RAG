import os
import subprocess

# Define parameters
DATASET = "sqa"
CUDA_DEVICE = "7"
top_k_list = [50]  # List of top_k values
testing_num = 100
# Define log directory
LOG_DIR = f"logs/{DATASET}"
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the directory exists
cluster_embedding_method = "contriever"  # contriever e5 sentencetransformer
table_to_graph_embedding_method = "sentencetransformer"  # contriever sentencetransformer Note: sentencetransformer contains (i) all-mpnet-base-v2 and (ii) E5-large,  choose inside the script

for top_k in top_k_list:
    # Define log file for each top_k
    LOG_FILE = f"{DATASET}_subgraph_testingnum{testing_num}_topK{top_k}_{table_to_graph_embedding_method}.log"
    LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

    # Construct the command
    command = [
        "python", f"subgraph_retrieve/subgraph_retrieve_{table_to_graph_embedding_method}.py",
        "--dataset", DATASET,
        "--num_iterations", "1",
        "--filter_topks", str(top_k),
        "--testing_num", str(testing_num),
        "--cluster_embedding_method", cluster_embedding_method,
        "--schema_only",
    ]

    # Run the command and log output
    with open(LOG_PATH, "w") as log_file:
        process = subprocess.run(
            command,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_DEVICE},
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

    print(f"Execution complete for top_k={top_k}. Logs saved to {LOG_PATH}")
