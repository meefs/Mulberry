import os
import json
import pdb
import random
import argparse

PROMPT = """Generate an image description based on the question.
Then, provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx.

Format your response with the following sections, separated by ###:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{question}"""

REFLECTION_PROMPT = "I think I have made a mistake. Let me rethink it.\n\n"



def reformat_reasoning(reasoning):
    reasoning_list = reasoning.split('###')
    output = ''
    len_steps = len(reasoning_list)
    for i, step in enumerate(reasoning_list):
        if i == 0:
            continue
        if i == 1:
            step = '### Image Description:' + ('###' + step).replace('### Image Description:', '').strip()
            output = output + step.replace('### Image Description:', '### Image Description:\n') + '\n\n'
        elif i == 2:
            step = '### Rationales:' + ('###' + step).replace('### Rationales:', '').strip()
            output = output + step.replace('### Rationales:', '### Rationales:\n') + '\n\n'
        elif i == 3:
            output = output + '### ' + step.strip() + '\n\n'
        elif i > 3 and i < len_steps - 1:
            if not step.strip().startswith(f'Step {i-3}'):
                print(step)
                print(reasoning)
                raise ValueError()
            step = f'### Step {i-3}:' + ('###' + step).replace(f'### Step {i-3}:', '').strip()
            output = output + step.replace(f'### Step {i-3}:', f'### Step {i-3}:\n') + '\n\n'
        elif i == len_steps-1:
            step = '### The final answer is:' + ('###' + step).replace('### The final answer is:', '').strip()
            output = output + step.replace('### The final answer is:', '### The final answer is:\n')
    return output

def reformat_reasoning_prefix(reasoning):
    if '### The final answer is:' in reasoning:
        raise ValueError()
    reasoning_list = reasoning.split('###')
    output = ''
    len_steps = len(reasoning_list)
    for i, step in enumerate(reasoning_list):
        if i == 0:
            continue
        if i == 1:
            step = '### Image Description:' + ('###' + step).replace('### Image Description:', '').strip()
            output = output + step.replace('### Image Description:', '### Image Description:\n') + '\n\n'
        elif i == 2:
            step = '### Rationales:' + ('###' + step).replace('### Rationales:', '').strip()
            output = output + step.replace('### Rationales:', '### Rationales:\n') + '\n\n'
        elif i == 3:
            output = output + '### ' + step.strip() + '\n\n'
        elif i > 3:
            if not step.strip().startswith(f'Step {i-3}'):
                print(step)
                print(reasoning)
                raise ValueError()
            step = f'### Step {i-3}:' + ('###' + step).replace(f'### Step {i-3}:', '').strip()
            output = output + step.replace(f'### Step {i-3}:', f'### Step {i-3}:\n') + '\n\n'
    return output

def read_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def get_depth(response):
    steps = response.split('###')
    return len(steps) - 1

def get_next_node_response(response, depth):
    res = response.split('###')
    return '###' + res[depth+1]

def get_remain_response(response, depth):
    res = response.split('###')
    output = ''
    for r in res[depth:]:
        output = output + '###' + r
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--models", nargs='*')
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--reflection_data_percentage", type=float, default=0.1)
    args = parser.parse_args()

    data_path = args.data_path

    if data_path.endswith('.jsonl'):
        total_data = read_jsonl(data_path)
    elif data_path.endswith('json'):
        with open(data_path, 'r') as f:
            total_data = json.load(f)
    else:
        raise NotImplementedError()

    # First, randomize all the data, then take X% of the data as reflection data.

    reflection_data_precentage = args.reflection_data_percentage
    reflection_data_limit = int(len(total_data) * reflection_data_precentage)

    random.shuffle(total_data)

    models = args.models
    print(models)
    
    reflect_data = []
    sft_data = []
    
    for data in total_data:
        tmp_dict = {}

        if any('error' in data[model]['response'].split('### The final answer is:')[-1].lower() and data[model]['valid']==1 and data[model]['is_correct']==1 for model in models):
            continue

        valid_num = 0
        correct_num = 0
        for model in models:
            if data[model]['valid'] == 1:
                valid_num += 1
                if data[model]['is_correct'] == 1:
                    correct_num += 1

        # no valid data
        if valid_num == 0 or correct_num == 0:
            continue

        tmp_value = -1
        max_value_model = ''
        for model in models:
            if data[model]['valid'] == 1 and data[model]['is_correct'] == 1 and data[model]['value'] > tmp_value:
                tmp_value = data[model]['value']
                max_value_model = model

        if_reflect_used = False
        # construct reflection data 
        if correct_num >= 2 and len(reflect_data) < reflection_data_limit:
            correct_data = data[max_value_model]
            
            prefix = reformat_reasoning_prefix(data['prefix_prompt'])
            prefix_depth = get_depth(prefix)

            max_diff = -2
            for model in models:
                if model == max_value_model:
                    continue
                if data[model]['valid'] == 1 and data[model]['step_value'][prefix_depth] < 0 and data[max_value_model]['step_value'][prefix_depth] - data[model]['step_value'][prefix_depth] > max_diff:
                    max_diff = data[max_value_model]['value'] - data[model]['value']
                    incorrect_data = data[model]
                    if_reflect_used = True
            
            if if_reflect_used:
                incorrect_data['response'] = reformat_reasoning(incorrect_data['response'])
                correct_data['response'] = reformat_reasoning(correct_data['response'])

                if prefix_depth < get_depth(incorrect_data['response'])-1:
                    reflection_response = prefix + get_next_node_response(incorrect_data['response'], prefix_depth) + REFLECTION_PROMPT + get_remain_response(correct_data['response'], prefix_depth+1)

                    response = reflection_response
                        
                    tmp_dict['messages'] = [{'role': 'user', 'content': '<image>' + data['conversations'][0]['value'].replace("<image>\n", "")}, {'role': 'assistant', 'content': response}]
                    tmp_dict['images'] = data['image']
                    reflect_data.append(tmp_dict)
                    continue
            else:
                if_reflect_used = False
                
        if if_reflect_used:
            continue

        response = data[max_value_model]['response']
        tmp_dict['messages'] = [{'role': 'user', 'content': '<image>' + data['conversations'][0]['value'].replace("<image>\n", "")}, {'role': 'assistant', 'content': reformat_reasoning(response)}]
        tmp_dict['images'] = data['image']
        sft_data.append(tmp_dict)

    print('reflection', len(reflect_data))
    print('sft', len(sft_data))

    reflect_data.extend(sft_data)
    random.shuffle(reflect_data)
    with open(args.output_path, 'w') as f:
        json.dump(reflect_data, f, indent=4)
