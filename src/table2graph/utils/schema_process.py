import json

dataset='sqa'
input_file = f'./data/{dataset}/{dataset}_table.jsonl'
output_file = f'./data/{dataset}/{dataset}_schema.jsonl'
with open(input_file, 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f]
    
num_lines = len(lines)

with open(output_file, 'a', encoding='utf-8') as f:
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