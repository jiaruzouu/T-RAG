import json 
import openai
import pandas as pd
from tqdm import tqdm, trange
import pdb
from vllm import LLM, SamplingParams
import os 
import argparse
import anthropic
# from evaluation import Evaluator

def get_few_shot_prompt(task_name: str):
    few_shot_prompt_list = {
        "tabfact":
'''
- Example 1:
# Final Answer: <answer>1</answer>
- Example 2:
# Final Answer: <answer>0</answer>
- Example 3:
# Final Answer: <answer>1</answer>
''',

        "hybridqa":

'''
- Example 1:
Final Answer: <answer>Jerry</answer>
- Example 2:
Final Answer: <answer>Starke Rudolf</answer>
- Example 3:
Final Answer: <answer>British</answer>
''',
        "wtq":
'''
- Example 1:
# Final Answer: <answer>["Aberdeen vs Hamilton Academical"]</answer>
- Example 2:
# Final Answer: <answer>["19 min", "20 min", "33 min"]</answer>
- Example 3:
# Final Answer: <answer>["RC Narbonne", "Montpellier RC", "Aviron Bayonnais", "Section Paloise", "RC Toulonnais"]</answer>
''',
        "sqa":
'''
- Example 1:
# Final Answer: <answer>["Roberto Feliberti Cintron"]</answer>
- Example 2:
# Final Answer: <answer>["53"]</answer>
- Example 3:
# Final Answer: <answer>["13", "55", "S01", "S30", "S800c", "S1200pj"]</answer>
'''

### NOTE: Add few-shot examples for other datasets
}
    
    return few_shot_prompt_list[task_name]


def get_instruction(task_name: str):
    instruction_list = {
        "tabfact": "Use the retrieved most relevant tables to verify whether the provided claim/query are true or false. Work through the problem step by step, and then return 0 if it's false, or 1 if it's true. Only return 0 or 1 without any other information. \n",
        "hybridqa": "Use the retrieved most relevant tables to answer the question. Only return the string instead of other format information. Do not repeat the question. \n",
        "sqa": "Utilize the most relevant retrieved tables to answer the question. Work through the problem step by step, and then return a list of strings to include ALL POSSIBLE final answers to the query. Note: Do not add extra content in the final answer lists. \n",
        "wtq": "Utilize the most relevant retrieved tables to answer the question. Work through the problem step by step, and then return a list of strings to include ALL POSSIBLE final answers to the query. Note: Do not add extra content in the final answer lists\n",
        # NOTE: Add instructions for other datasets
    }
    return instruction_list[task_name]

def table_to_html(table_data: dict) -> str:
    """
    Convert a table dictionary to an HTML representation.
    
    :param table_data: Dictionary containing the table data.
    :return: HTML string representation of the table with a caption.
    """
    caption = table_data.get("caption", "")
    header = table_data["table"]["header"]
    rows = table_data["table"]["rows"]
    df = pd.DataFrame(rows, columns=header)
    
    html_table = df.to_html(index=False, escape=False) ##NOTE: You can also use other formats like markdown, json, latex, etc.
    
    return f"<h3>{caption}</h3>\n{html_table}"



def construct_prompt_gpt(retrieve_instance: dict, dataset) -> str:
    prompt = ""
    testing_query = retrieve_instance["query"]
    retrieve_tables = retrieve_instance["retrieved_tables"]
    system_prompt = '''
As a expert in tabular data analysis and RAG, you are given a query and a set of tables. 
The query is the question you need to answer and the set of tables are the source of information you can retrieve to help you answer the given query.
You are asked to provide a response to the query based on the information in the tables. Follow the instructions below:

# Step one: Find most relevant tables to answer the query
1. Read the query and the tables carefully.
2. Given the query information, figure out and find the most relevant tables (normally 1-3 tables) from the set of tables to answer the query.
3. Once you have identified the relevant tables, follow the step two to answer the query.
4. Note that sometimes the answer of the query may not be directly obtained from the given tables or the tables might totally irrelevant to the query. In this case, You need to think step by step and try your best to answer the question based on your pre-trained knowledge.

# Step two: Answer the query based on the retrieved tables
For step two, follow the detailed instructions here:
'''
    task_instruction = get_instruction(dataset)
    
    prompt += f"{system_prompt}{task_instruction}\n"
    
    prompt += f'''
# Query: {testing_query}
# The Table Set:
'''
    
    for i,table in enumerate(retrieve_tables):
        table_html = table_to_html(table)
        prompt += f"Table {i+1}:\n {table_html}\n"
        
    prompt += f'''
# Output Instructions: Here we provide output instructions that you MUST strictly follow.
1. You MUST think step by step via the chain-of-thought for the given task and then give a final answer.
2. Your output MUST conclude two compenents: the chain-of-thought (CoT) steps to reach the final answer and the final answer.
3. Output your thinking steps and the final answer using <answer>NA</answer> if the query can not be answered. Note that you MUST try your based based on the given tables and your pre-trained knowledge to give an answer.
4. For the CoT component, you MUST enclose your reasoning between <reasoning> and </reasoning> tags.
5. For the final answer component, you MUST enclose your reasoning between <answer> and </answer> tags.
Here are few-shot examples to demonstrate the final answer component format: 
{get_few_shot_prompt(dataset)}
'''
    prompt += "\n# Now Output Your response below:"
    
    return prompt

# For claude 
def construct_prompt_claude(retrieve_instance: dict, dataset) -> str:
    prompt = ""
    testing_query = retrieve_instance["query"]
    retrieve_tables = retrieve_instance["retrieved_tables"]
    system_prompt = '''
As a expert in tabular data analysis and RAG, you are given a query and a set of tables. 
The query is the question you need to answer and the set of tables are the source of information you can retrieve to help you answer the given query.
You are asked to provide a response to the query based on the information in the tables. Follow the instructions below:

# Step one: Find most relevant tables to answer the query
1. Read the query and the tables carefully.
2. Given the query information, figure out and find the most relevant tables (normally 1-3 tables) from the set of tables to answer the query.
3. Once you have identified the relevant tables, follow the step two to answer the query.
4. Note that sometimes the answer of the query may not be directly obtained from the given tables or the tables might totally irrelevant to the query. In this case, You need to think step by step and try your best to answer the question based on your pre-trained knowledge.

# Step two: Answer the query based on the retrieved tables
For step two, follow the detailed instructions here:
'''
    task_instruction = get_instruction(dataset)
    
    prompt += f"{system_prompt}{task_instruction}\n"
    
    prompt += f'''
# Query: {testing_query}
# The Table Set:
'''
    
    for i,table in enumerate(retrieve_tables):
        table_html = table_to_html(table)
        prompt += f"Table {i+1}:\n {table_html}\n"
    
    prompt += f'''
# Output Instructions: Here we provde several examples including the query and the corresponding answer to demonstrate the outut format.
{get_few_shot_prompt(dataset)}
- As incicated by the few shot demonstrations, your output MUST ONLY contain the answer following ```answer\n\n``` 
- DO NOT output ANY other contents such as your thinking steps.
- If you cannot find the answer, ONLY output ```answer\nNA\n``` to indicate that the answer is not in the table.
'''
    prompt += "\n# Output: "
    
    return prompt

# For Open-source models like LLaMA, Qwen, Gemma, Phi-3, etc.
def construct_prompt_open_source(retrieve_instance: dict, dataset) -> str:
    prompt = ""
    testing_query = retrieve_instance["query"]
    retrieve_tables = retrieve_instance["retrieved_tables"]
    system_prompt = '''
As a expert in tabular data analysis and RAG, you are given a query and a set of tables. 
The query is the question you need to answer and the set of tables are the source of information you can retrieve to help you answer the given query.
You are asked to provide a response to the query based on the information in the tables. Follow the instructions below:

# Step one: Find most relevant tables to answer the query
1. Read the query and the tables carefully.
2. Given the query information, figure out and find the most relevant tables (normally 1-3 tables) from the set of tables to answer the query.
3. Once you have identified the relevant tables, follow the step two to answer the query.
4. Note that sometimes the answer of the query may not be directly obtained from the given tables or the tables might totally irrelevant to the query. In this case, You need to think step by step and try your best to answer the question based on your pre-trained knowledge.

# Step two: Answer the query based on the retrieved tables
For step two, follow the detailed instructions here:
'''
    task_instruction = get_instruction(dataset)
    
    prompt += f"{system_prompt}{task_instruction}\n"
    
    prompt += f'''
# Query: {testing_query}
# The Table Set:
'''
    
    for i,table in enumerate(retrieve_tables):
        table_html = table_to_html(table)
        prompt += f"Table {i+1}:\n {table_html}\n"

    prompt += f'''
# Step three: Output Instructions: Here we provide output instructions that you MUST strictly follow.
1. You MUST think step by step via the chain-of-thought for the given task and then give a final answer.
2. Your output MUST conclude two compenents: the chain-of-thought (CoT) steps to reach the final answer and the final answer.
3. For the CoT component, you MUST enclose your reasoning between <reasoning> and </reasoning> tags.
4. For the final answer component, you MUST enclose your reasoning between <answer> and </answer> tags.
Here are few-shot examples to demonstrate the final answer component format: 
{get_few_shot_prompt(dataset)}
5. If you try your best but still cannot find the answer from both the given table sources and your pretrained knowledge, then output your thinking steps and the final answer using <answer>NA</answer> to indicate that the answer can not be answered. 
'''
    prompt += "\n# Now Output Your response below:"
    
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process dataset.")
    parser.add_argument("--topk", type=int, required=True, help="Name of the retrieved tables")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--mode", type=str, required=True, help="API or offline")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--starting_idx", type=int, default=0, help="Starting index")
    parser.add_argument("--testing_num", type=int, required=True, help="testing_num of the queries")
    parser.add_argument("--embedding_method", type=str, default="contriever", help="Embedding method used during retrieval process")

    args = parser.parse_args()
    topk = args.topk
    dataset = args.dataset
    mode = args.mode
    model = args.model
    starting_idx = args.starting_idx
    testing_num = args.testing_num
    
    print(f"args: {args}")
    
    retrieve_table_file = f"../table2graph/data/{dataset}/{dataset}_retrieved_tables_schema_{testing_num}_{topk}_{embedding_method}.jsonl"
    
    output_dir = f"./output/{dataset}/{model}/"
    output_file = f"{output_dir}/output_{testing_num}_{topk}.jsonl"
    key_file = "./key.json"
    
    with open(retrieve_table_file, "r") as f:
        retrieve_instances = f.readlines()
    
    if "gpt" in model:
        key_type = "openai"
    elif "claude" in model:
        key_type = "claude"
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    prompts = []
    print("Constructing prompts...")
    groundtruths = []
    querys = []
    for line in tqdm(retrieve_instances):
        retrieve_instance = json.loads(line)
        groundtruths.append(retrieve_instance["query_label"])
        querys.append(retrieve_instance["query"])
        if "gpt" in model:
            prompt = construct_prompt_gpt(retrieve_instance, dataset)
        elif "claude" in model:
            prompt = construct_prompt_claude(retrieve_instance, dataset)
        
        elif "llama" in model:
            prompt = construct_prompt_open_source(retrieve_instance, dataset)
            
        elif "gemma" in model:
            prompt = construct_prompt_open_source(retrieve_instance, dataset)
            
        elif "Qwen" in model:
            prompt = construct_prompt_open_source(retrieve_instance, dataset)
        
        elif "Phi" in model:
            prompt = construct_prompt_open_source(retrieve_instance, dataset)
        
        else:
            raise ValueError("Model corresponding to the prompt is not found.")
        
        prompts.append(prompt)

    print("Generating responses...")
    if mode == "offline":
        sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=2048)
        llm = LLM(model=model, gpu_memory_utilization=0.6, tensor_parallel_size=4)
        outputs = llm.generate(prompts, sampling_params)
        
        
        with open(output_file, "a") as f:            
            for i, output in tqdm(enumerate(outputs)):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                output = {
                    "query": querys[i],
                    "groundtruth": groundtruths[i],
                    "output": generated_text,
                }

                f.write(json.dumps(output) + "\n")
                
    elif mode == "API":
        with open(key_file, "r") as f:
            key = json.load(f)
            api_key = key[key_type]
        
        if "gpt" in model:
            with open(retrieve_table_file, "r") as f:
                retrieve_instances = f.readlines()

            client = openai.OpenAI(api_key=api_key)

            with open(output_file, "a") as f:  # Open file for writing API responses
                for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Processing Prompts"):
                    if i < starting_idx:
                        continue
                    
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt.strip()}],
                            temperature=0.1,
                            max_tokens=2048,
                            top_p=0.95,
                        )

                        generated_text = response.choices[0].message.content.strip()
                        
                        output = {
                            "query": querys[i],
                            "groundtruth": groundtruths[i],
                            "output": generated_text,
                        }
                        
                        if i < 5:
                            print(generated_text)
                        
                        f.write(json.dumps(output) + "\n")

                    except Exception as e:
                        print(f"Error calling OpenAI API: {e}")
            
        elif "claude" in model:
            with open(retrieve_table_file, "r") as f:
                retrieve_instances = f.readlines()

            client = anthropic.Anthropic(api_key=api_key)

            with open(output_file, "a") as f:  # Open file for writing API responses
                for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Processing Prompts"):
                    if i < starting_idx:
                        continue
                    try:
                        response = client.messages.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt.strip()}],
                            temperature=0.1,
                            max_tokens=2048,
                            top_p=0.95,
                        )
                        
                        generated_text = response.content[0].text.strip()
                        print(generated_text)
                        output = {
                            "query": querys[i],
                            "groundtruth": groundtruths[i],
                            "output": generated_text,
                        }
                        
                        if i < 5:
                            print(generated_text)
                        
                        f.write(json.dumps(output) + "\n")

                    except Exception as e:
                        print(f"Error calling Anthropic API: {e}")
            
