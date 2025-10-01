from huggingface_hub import hf_hub_download
import shutil
import os
import json
import random 
from collections import defaultdict

datasets = ["TATQA", "TabFact", "HybridQA", "WTQ", "SQA"]

for dataset in datasets:
    
    ## Step1: DOWNLOAD DATASET
    print(f"Downloading dataset from HuggingFace: {dataset}")
    
    save_dir = f"./data/{dataset.lower()}/"
    os.makedirs(save_dir, exist_ok=True)

    table_file = f"{dataset.lower()}_table.jsonl"
    query_file = f"{dataset.lower()}_query.jsonl"

    files = [query_file, table_file]

    for f in files:
        downloaded_path = hf_hub_download(
            repo_id=f"jiaruz2/MultiTableQA_{dataset}",
            filename=f,
            repo_type="dataset"
        )
        
        dest_path = os.path.join(save_dir, f)  
        shutil.copy(downloaded_path, dest_path)
        print(f"Downloaded {f} → {dest_path}")
        
    
    ## Step2: PROCESS TABLE SCHEMA and EXAMPLE QUERY
    schema_output_file = f'./data/{dataset.lower()}/{dataset.lower()}_schema.jsonl'
    example_query_output_file = f'./data/{dataset.lower()}/{dataset.lower()}_example_query.jsonl'
    
    with open(save_dir + table_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
        
    num_lines = len(lines)

    with open(schema_output_file, 'a', encoding='utf-8') as f:
        for i in range(num_lines):
            line_data = lines[i]
            table_schema = f"Caption: {line_data['caption']}; Headers: {line_data['table']['header']};"
            source_table_idx = line_data['source_table_idx']
            table_idx = line_data['table_idx']
            output = {
                "table_schema": table_schema,
                "source_table_idx": source_table_idx,
                "table_idx": table_idx
            }
            f.write(json.dumps(output) + '\n')
        
        print(f"Processed table schema and saved to {schema_output_file}")
            
    with open(example_query_output_file, 'a', encoding='utf-8') as f:
        for i in range(num_lines):
            line_data = lines[i]
            Example_queries = line_data['example_query']
            
            # Randomly select one example query from the list of example queries
            if len(Example_queries) > 0:
                example_query = Example_queries[random.randint(0, len(Example_queries)-1)]
            else:
                example_query = ""

            source_table_idx = line_data['source_table_idx']
            table_idx = line_data['table_idx']
            
            output = {
                "example_query": example_query,
                "source_table_idx": source_table_idx,
                "table_idx": table_idx
            }
            f.write(json.dumps(output) + '\n')
    
        print(f"Processed example queries and saved to {example_query_output_file}")
    
    ## Step3: PROCESS SOURCE-SUB TABLE MATCH
    match_output_file = f'./data/{dataset.lower()}/{dataset.lower()}_table_match.json'
    table_mapping = defaultdict(set)
    with open(save_dir + table_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            source_table_idx = int(data["source_table_idx"])
            table_idx = data["table_idx"]
        
            # Add table_idx to the corresponding source_table_idx
            table_mapping[source_table_idx].add(table_idx)
            
    output_data = {int(key): list(value) for key, value in table_mapping.items()}

    with open(match_output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4)
    
    print(f"Processed source-sub table match and saved to {match_output_file}")