# Import packages
import json
from huggingface_hub import login
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments
import transformers
import random
import torch
import time
import re
from tqdm import tqdm
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import cuda
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import label_binarize
from scipy.special import softmax


# Set seeds
random.seed(0)
torch.manual_seed(0)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["Prognosis"], padding="max_length", truncation=True, max_length = 512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to probabilities
    probs = softmax(logits, axis=1)

    # Convert labels to one-hot encoding
    labels_one_hot = label_binarize(labels, classes=[0, 1, 2])  # Ensure class order matches label mapping

    # Compute accuracy
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)

    # Compute precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    # Compute multi-class AUC
    auc = roc_auc_score(labels_one_hot, probs, multi_class="ovr")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

if __name__ == "__main__":

    # Set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    # Import data
    df = pd.read_csv("../Data/subject-info-cleaned-with-prognosis-D-Llama3B.csv")
    
    # Get prognosis and outcome
    df = df[['Prognosis', 'Outcome']]
    
    # Map labels to integers
    # Survivor = 0, SCD = 1, PFD = 2
    label_map = {"survivor": 0, "sudden cardiac death": 1, "pump failure death": 2}
    df['Outcome'] = df['Outcome'].map(label_map)
    df.rename(columns = {"Outcome": "labels"}, inplace = True)

    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_pandas(df)

    # Load BioBERT
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    num_labels = 3  # Three possible outcomes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Apply tokenization
    dataset = dataset.map(tokenize_function, batched=True)

    # Split data
    train_test = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test["train"]
    val_dataset = train_test["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="../Results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="../logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics
    )
    trainer.train()

    # Evaluate
    print(trainer.evaluate())