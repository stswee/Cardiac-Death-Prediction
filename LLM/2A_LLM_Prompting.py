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

def get_message(note):
    system = 'You are a cardiologist. Your task is to generate a prognosis for the next few years with reasoning. The three possible prognoses are survivor, sudden cardiac death, and pump failure death. Please structure the results as follows: PREDICTION: [SURVIVOR | SUDDEN CARDIAC DEATH | PUMP FAILURE DEATH] \n REASONING: [Your explanation here]'
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
    