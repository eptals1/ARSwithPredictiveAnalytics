import torch
from transformers import (
    RobertaTokenizerFast, 
    RobertaForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import json
import numpy as np
from typing import List, Dict
import evaluate
import os

def normalize_label(label: str) -> str:
    """
    Normalize label format by converting 'B-SKILLS' to 'B-SKILL' and handling other variations
    """
    if label == 'O':
        return label
    
    # Map of variations to standard format
    label_map = {
        'B-SKILLS': 'B-SKILL',
        'B-EDUCATIONS': 'B-EDUCATION',
        'B-EXPERIENCES': 'B-EXPERIENCE',
        'B-CERTIFICATIONS': 'B-CERTIFICATION',
        'B-ADDRESSES': 'B-ADDRESS'
    }
    
    return label_map.get(label, label)

def load_labeled_data(json_path: str):
    """
    Load and prepare labeled data for training
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to format suitable for training
    processed_data = {
        'tokens': [],
        'labels': [],
    }
    
    # Define label to id mapping with standardized labels
    label_list = ['O', 'B-SKILL', 'B-SKILLS', 'B-EDUCATION', 'B-EXPERIENCE', 'B-CERTIFICATION', 'B-ADDRESS', 'B-GENDER']
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    for item in data:
        processed_data['tokens'].append(item['tokens'])
        # Convert string labels to ids
        label_ids = [label2id[label] for label in item['labels']]
        processed_data['labels'].append(label_ids)
    
    return processed_data, label2id, id2label

def get_unique_labels(data):
    """Get all unique labels from the dataset"""
    unique_labels = set()
    for item in data:
        unique_labels.update(item['labels'])
    return sorted(list(unique_labels))

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize the text and align the labels with the tokenized words
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    """
    Compute metrics for model evaluation
    """
    metric = evaluate.load("seqeval")
    
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output/models/roberta-resume-ner", exist_ok=True)
    
    # Load labeled data
    print("Loading labeled data...")
    json_path = "output/dataset/labeled_data.json"
    
    # First, let's see what labels we have
    with open(json_path, 'r') as f:
        data = json.load(f)
    unique_labels = get_unique_labels(data)
    print("\nUnique labels found in dataset:")
    for label in unique_labels:
        print(f"- {label}")
    
    processed_data, label2id, id2label = load_labeled_data(json_path)
    
    # Create dataset
    dataset = Dataset.from_dict(processed_data)
    
    # Split dataset
    train_test = dataset.train_test_split(test_size=0.2)
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    model_name = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenize datasets
    print("\nPreparing datasets...")
    train_dataset = train_test["train"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=train_test["train"].column_names
    )
    
    eval_dataset = train_test["test"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=train_test["test"].column_names
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="output/models/roberta-resume-ner",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the model
    print("\nSaving model...")
    trainer.save_model("output/models/roberta-resume-ner/final")
    
    # Evaluate the model
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")

