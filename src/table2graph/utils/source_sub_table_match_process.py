import json
from collections import defaultdict

# Input and output file paths
dataset = "sqa"  
input_file = f"./data/{dataset}/{dataset}_table.jsonl"  # Change this to your actual input file path
output_file = f"./data/{dataset}/{dataset}_table_match.json"  # Change this to your desired output file path

# Dictionary to store source_table_idx as keys and list of table_idx as values
table_mapping = defaultdict(set)

# Read the input JSONL file and process each line
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line.strip())
        source_table_idx = int(data["source_table_idx"])
        table_idx = data["table_idx"]
        
        # Add table_idx to the corresponding source_table_idx
        table_mapping[source_table_idx].add(table_idx)

# Convert sets to lists for JSON serialization
output_data = {int(key): list(value) for key, value in table_mapping.items()}

# Write the output JSONL file as a single dictionary
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(output_data, outfile, indent=4)

print("Processing complete. Output saved to", output_file)
