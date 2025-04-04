# Import packages
import json
from huggingface_hub import login
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import random
import torch
import time
import re
from tqdm import tqdm
import pandas as pd
import os

# Set seeds
random.seed(0)
torch.manual_seed(0)

def prompt_outcome(note, outcome):
    return note + "\nPATIENT OUTCOME: " + outcome

def get_message(note):
    system = 'You are a cardiologist. Your task is to generate a prognosis for the next few years with reasoning. The three possible prognoses are survivor, sudden cardiac death, and pump failure death. Order the three prognoses from most likely to least likely and explain the reasoning behind the order. Please structure the results as follows: RANKING: [Your ranking here] \n REASONING: [Your explanation here]'
    prompt = f"Here is the patient data: \n{note}"
    prompt = f"Here is the patient data: \n{note}"

    messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]

    return messages

def extract_assistant_response(response):
    parts = response.split("assistant\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else response

if __name__ == "__main__":

    # Set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    # Log into Huggingface
    with open("../../huggingface_token.txt", "r") as file:
        access_token = file.read().strip()
    login(access_token)
    
    # Load Huggingface Model
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, low_cpu_mem_usage=True,
                        torch_dtype=torch.float16, device_map='auto')

    # Load in csv file with prompts
    df = pd.read_csv("../Data/subject-info-cleaned-with-prompts.csv") # Plan D does not require repeats

    # Create empty column to store results
    df['Prognosis'] = None

    # Prompt LLM
    for i in tqdm(range(len(df)), desc = "Generating responses"):
        # Get message
        message = get_message(prompt_outcome(df['Prompts'][i], df['Outcome'][i]))
    
        # Put message into LLM
        input_text = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
        inputs = tokenizer(input_text, return_tensors = "pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens = 1000)
    
        # Get result
        result = tokenizer.decode(output[0], skip_special_tokens = True)
        result = result.replace("**", "")
        result = extract_assistant_response(result)
    
        # Store result
        df.loc[i, 'Prognosis'] = result

    # Store dataframe as csv file
    df.to_csv("../Data/subject-info-cleaned-with-prognosis-D-Llama3B.csv") # Plan D