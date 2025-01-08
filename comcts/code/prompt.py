JUDGE_PROMPT = """Evaluate whether the model's answer matches the correct result. 

- If it does not align, respond with 'No'.
- If there is a logical error in the reasoning steps, respond with 'No'.
- If the model's answer aligns with the correct result, respond with 'Yes'. 

Provide only 'Yes' or 'No' as the output, with no explanation.

The question is: {question}

The model's answer is: {model_answer}

The correct result is: {gt_answer}"""



PROMPT = """Generate an image description based on the question.
Then, provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx'. If the question is multiple-choice, provide the options along with their content. If it is free-form, directly present the final result. Do not provide any explanation.

Format your response with the following sections, separated by ###:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{question}"""



LOCATE_ERROR_PROMPT = ''''### Question:
{question}

### Ground truth answer:
{gt}

### Reasoning steps:
{reasoning}

Given the question and reasoning steps listed above, along with the corresponding ground truth answer, please evaluate the correctness of the image description, rationales, and each step of the reasoning process.

Requirements:
1. Output the decision ("correct", "neutral", "incorrect") for each step following the format of "Final Decision:\nImage Description: [your decision]; Rationales: [your decision]; Let's think step by step: [your decision]; Step 1: [your decision]; Step 2: [your decision]; ...";
2. Do not provide any explanation.'''



GPT_PREFIX_PROMPT = """Generate an image description based on the question.
Then, provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx'. If the question is multiple-choice, provide the options along with their content. If it is free-form, directly present the final result. Do not provide any explanation.

Format your response with the following sections, separated by ###:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{question}

Please complete the response based on the reasoning prefix without altering its content.

Reasoning prefix: {reasoning_prefix}"""