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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
import argparse

# Ensure reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["Prompts"], padding="max_length", truncation=True, max_length = 512)

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
    auc_multi = roc_auc_score(labels_one_hot, probs, multi_class="ovr")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_multi
    }

def plot_metrics(labels, predictions, probs, dataset_name, model_name):
    # Define mappings for the dataset and model to human-readable names
    dataset_map = {
        "../Data/subject-info-cleaned-with-prognosis-D-Llama3B.csv": "Llama3B",
        "../Data/subject-info-cleaned-with-prognosis-D-Llama8B.csv": "Llama8B",
    }
    model_map = {
        "dmis-lab/biobert-base-cased-v1.1": "BioBERT",
        "emilyalsentzer/Bio_ClinicalBERT": "ClinicalBERT",
    }

    # Extract readable LLM and LM names
    dataset_filename = os.path.basename(dataset_name)
    llm_name = dataset_map.get(dataset_filename, "UnknownLLM")
    lm_name = model_map.get(model_name, "UnknownLM")
    
    # Title and filename prefix
    title_prefix = f"{llm_name} + {lm_name}"
    model_filename = model_name.replace("/", "_")
    plot_prefix = f"{dataset_filename.replace('.csv', '')}_{model_filename}"
    
    # Plot AUROC
    labels_one_hot = label_binarize(labels, classes=[0, 1, 2])
    class_names = ["Survivor", "Sudden Cardiac Death", "Pump Failure Death"]
    
    plt.figure(figsize=(8, 6))
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(labels_one_hot[:, i], probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_score:.2f})")
    
    # Random guess line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"AUROC Curve: {title_prefix}")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_auroc_plot.png")
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {title_prefix}")
    plt.savefig(f"{plot_prefix}_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":

    # User arguments
    parser = argparse.ArgumentParser(description="Process dataset and model name.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file", default="../Data/subject-info-cleaned-with-prompts.csv") # Datasets: ../Data/subject-info-cleaned-with-prompts.csv
    parser.add_argument("--model_name", type=str, help="Name of the pre-trained model", default="emilyalsentzer/Bio_ClinicalBERT") # Models: dmis-lab/biobert-base-cased-v1.1, emilyalsentzer/Bio_ClinicalBERT
    args = parser.parse_args()

    # Set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    # Import data
    df = pd.read_csv(args.dataset_path)
    
    # Get prognosis and outcome
    df = df[['Prompts', 'Outcome']]
    
    # Map labels to integers
    label_map = {"survivor": 0, "sudden cardiac death": 1, "pump failure death": 2}
    df['Outcome'] = df['Outcome'].map(label_map)
    df.rename(columns={"Outcome": "labels"}, inplace=True)

    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_pandas(df)

    # Load BioBERT (or any specified model)
    num_labels = 3  # Three possible outcomes
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    
    # Apply tokenization
    dataset = dataset.map(tokenize_function, batched=True)

    # Split data
    train_test = dataset.train_test_split(test_size=0.2, seed = 0)
    train_dataset = train_test["train"]
    val_dataset = train_test["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="../Results",
        eval_strategy="epoch",
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
    eval_results = trainer.evaluate()
    print(eval_results)

    # Save evaluation metrics
    metrics_filename = f"{os.path.basename(args.dataset_path).replace('.csv', '')}_{args.model_name.replace('/', '_')}_metrics.json"
    with open(metrics_filename, "w") as f:
        json.dump(eval_results, f, indent=4)

    # Generate predictions for validation set
    predictions_output = trainer.predict(val_dataset)
    logits = predictions_output.predictions
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=1)
    labels = np.array(val_dataset["labels"])

    # Plot AUROC and Confusion Matrix
    plot_metrics(labels, predictions, probs, args.dataset_path, args.model_name)