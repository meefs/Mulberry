from openai import OpenAI
import json
import base64
from tqdm import tqdm
import os
import math
import argparse
from PIL import Image
from collections import Counter
import heapq
import math

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def read_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def question_process(d):
    if 'question' in d.keys():
        question = d['question']
    elif '\nAnswer the question with a short answer.' in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace('\nAnswer the question with a short answer.', '')
    elif "\nAnswer with the option's letter from the given choices directly." in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace("\nAnswer with the option's letter from the given choices directly.", '')
    elif "\nAnswer the question using a single word or phrase." in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace("\nAnswer the question using a single word or phrase.", '')
    elif "<image>\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\n" in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace('<image>\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\n', '')
    elif "<image>\nBased on the image, directly select the correct answer for the following question:\n" in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace('<image>\nBased on the image, directly select the correct answer for the following question:\n', '')
    else:
        question = d["conversations"][0]['value']

    if not question.startswith('Question:'):
        question = 'Question: ' + question

    return question

def find_img_path(d,args):
    if os.path.exists(os.path.join(args.image_dir_path, d['image'])):
        img_path = os.path.join(args.image_dir_path, d['image'])
    elif os.path.exists(d['image']):
        img_path = d['image']
    else:
        raise ValueError(f"Image path not found: {d['image']}")

    return img_path


def gpt_forward(client, prompt, base64_image=None, temperature=0.9):
    content = [{
                    "type": "text",
                    "text": prompt
                }]
    if base64_image is not None:
        content.append({
            "type": "image_url",
            "image_url":{
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        temperature = temperature
    )

    return completion.choices[0].message.content


def get_correctness(judge_output):
    if 'yes' in judge_output.lower() and 'no' not in judge_output.lower():
        return 1
    else:
        return -1

def qwen2_vl_forward(model, processor, question, prefix_prompt, img_path, temperature=0.9):
    messages = [
        {
            'role': "system",
            "content": 'You are a helpful assistant.'
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": question},
            ],
        },
    ]

    image = Image.open(img_path)
    h, w = image.size
    if h < 28 or w < 28:
        factor = 28 / h if h < w else 28 / w
        if h < w:
            image = image.resize((28, int(w * factor)))
        else:
            image = image.resize((int(h * factor), 28))

    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix_prompt]
                
    inputs = processor(
        text=texts,
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1, temperature=temperature)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return prefix_prompt + output_texts

def llama_forward(model, processor, question, prefix_prompt, img_path, temperature=0.9):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": question},
            ],
        },
    ]
    image = Image.open(img_path)
    texts = processor.apply_chat_template(messages, add_generation_prompt=True) + prefix_prompt

    inputs = processor(image, texts, return_tensors="pt").to('cuda')

    generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=temperature)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return prefix_prompt + output_texts

def check_data(steps):
    steps = steps.split('###')
    len_steps = len(steps)
    for i, step in enumerate(steps):
        if i == 0:
            if step == '':
                continue
            else:
                return False

        elif i == 1:
            if step.strip().startswith('Image Description:'):
                continue
            else:
                return False

        elif i == 2:
            if step.strip().startswith('Rationales:'):
                continue
            else:
                return False
        
        elif i == 3:
            if step.strip().startswith("Let's think step by step"):
                continue
            else:
                return False
        
        elif i > 3 and i < len_steps - 1:
            if step.strip().startswith(f"Step {i-3}"):
                continue
            else:
                return False

        elif i == len_steps-1:
            if step.strip().startswith("The final answer is:") and i > 4:
                continue
            else:
                return False
    return True

def check_validity(response):
    if not check_data(response):
        return False
    if len(response.split('### The final answer is:')) == 2:
        return True
    return False

def get_depth(response):
    # image description is depth 1
    steps = response.split('###')
    return len(steps) - 1


def get_step(response, depth):
    res = response.split('###')
    return '###' + res[depth]


def step_correctness_to_list(response, depth):
    step_correctness_list = []
    output_scores = response.split('Final Decision:')[-1].strip()
    output_scores_list = output_scores.split(';')
    for score in output_scores_list:
        if 'incorrect' in score.lower():
            step_correctness_list.append(-1)
        elif 'neutral' in score.lower():
            step_correctness_list.append(0)
        elif 'correct' in score.lower():
            step_correctness_list.append(1)
    if len(step_correctness_list) != depth-1:
        return [-2]
    return step_correctness_list
    

def prune_response(response, idx): 
    steps = response.split('###')
    len_steps = len(steps) - 1
    if idx == 0:
        return ''
    elif idx == 1:
        index = response.find('### Rationales:')
        if index != -1:
            return response[:index]
        else:
            return response
    elif idx == 2:
        index = response.find("### Let's think step by step") if response.find("### Let's think step by step") != -1 else response.find("###Let's think step by step")
        if index != -1:
            return response[:index]
    elif idx > 2 and idx < len_steps-1:
        index = response.find(f'### Step {idx-2}')
        if index != -1:
            return response[:index]
    elif idx == len_steps-1:
        index = response.find(f'### The final answer is:')
        if index != -1:
            return response[:index]
    else:
        return ''


def prune_tree(comcts_dict, start_index, threshold=0):
    pruned_comcts_dict = dict()
    step_correctness_list = comcts_dict['step_correctness']
    first_less_than_zero_idx = -1
    for i, value in enumerate(step_correctness_list):
        if i < start_index:
            continue
        if value < threshold:
            first_less_than_zero_idx = i
            break

    if first_less_than_zero_idx == -1 or first_less_than_zero_idx == 0:
        comcts_dict['valid'] = -1
        return comcts_dict
    
    pruned_response = prune_response(comcts_dict['response'], first_less_than_zero_idx)
    pruned_step_correctness = step_correctness_list[:first_less_than_zero_idx]

    pruned_comcts_dict['response'] = pruned_response
    pruned_comcts_dict['step_correctness'] = pruned_step_correctness
    pruned_comcts_dict['valid'] = comcts_dict['valid']

    return pruned_comcts_dict


def modified_qwen_response(response):
    for i in range(1, 15):
        step_idx = f'Step {i}:'
        if f'### Step {i}:' not in response and step_idx in response:
            if response.count(step_idx) == 1:
                response = response.replace(step_idx, f'### Step {i}:')

    if "### Final Answer:\nThe final answer is:" in response:
        response = response.replace('### Final Answer:\nThe final answer is:', '### The final answer is:')
    elif "### Final Answer:" in response:
        response = response.replace('### Final Answer:', '### The final answer is:')
    
    if "### Rationale:" in response and "### Rationales" not in response:
        response = response.replace('### Rationale:', '### Rationales:')

    return response

def modified_llama_response(response):
    for i in range(1, 15):
        step_idx = f'Step {i}:'
        if f'### Step {i}:' not in response and step_idx in response:
            if response.count(step_idx) == 1:
                response = response.replace(step_idx, f'### Step {i}:')

    if '### The final answer' not in response and "The final answer is" in response:
        response = response.replace("The final answer is", '### The final answer is:')
    
    if "### Rationale:" in response and "### Rationales" not in response:
        response = response.replace('### Rationale:', '### Rationales:')

    return response


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
                step = f'### Step {i-3}:' + ('###' + step).replace(f'### Step {i-3}:', '').strip()
                output = output + step.replace(f'### Step {i-3}:', f'### Step {i-3}:\n') + '\n\n'
        return output
