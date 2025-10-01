import os
import subprocess

# Define your parameter lists
datasets = ["sqa"]  # options: "hybridqa", "wtq", "tabfact"
n_clusters = [3]    # options: 5, 10, 20
ks = [50]           # options: 100, 150, 200
embedding_method = "contriever"  # options: contriever, e5, sentencetransformer

for k in ks:
    for n_cluster in n_clusters:
        for dataset in datasets:
            # Create logs directory
            log_dir = f"logs/{dataset}/"
            os.makedirs(log_dir, exist_ok=True)

            # Log file path
            log_file = f"{log_dir}{dataset}_cluster_k{k}_n{n_cluster}_{embedding_method}.log"
            print(f"Logging to {log_file}")

            # Write header in log file
            with open(log_file, "w") as f:
                f.write(f"Clustering table in Dataset: {dataset} with k={k} and n_clusters={n_cluster}\n")

            # Run the clustering script and append output to log file
            with open(log_file, "a") as f:
                subprocess.run(
                    [
                        "python",
                        f"cluster/table_cluster_{embedding_method}.py",
                        "--dataset", dataset,
                        "--n_clusters", str(n_cluster),
                        "--k", str(k)
                    ],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
