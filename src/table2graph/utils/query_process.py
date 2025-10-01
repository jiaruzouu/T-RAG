    import json
import random 

dataset='sqa'
input_file = f'./data/{dataset}/{dataset}_table.jsonl'
output_file = f'./data/{dataset}/{dataset}_example_query.jsonl'
with open(input_file, 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f]
    
num_lines = len(lines)

with open(output_file, 'a', encoding='utf-8') as f:
    for i in range(num_lines):
        line_data = lines[i]
        Example_queries = line_data['example_query']
        # Randomly select one example query from the list of example queries
        example_query = Example_queries[random.randint(0, len(Example_queries)-1)]
        source_table_idx = line_data['source_table_idx']
        table_idx = line_data['table_idx']
        output = {
            "example_query": example_query,
            "source_table_idx": source_table_idx,
            "table_idx": table_idx
        }
        f.write(json.dumps(output) + '\n')