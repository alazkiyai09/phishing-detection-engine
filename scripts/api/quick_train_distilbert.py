#!/usr/bin/env python3
"""
Quick DistilBERT training script for testing.
Generates synthetic email data and trains a model in ~30 minutes (GPU) or ~2 hours (CPU).
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
import numpy as np

# Configuration
N_SAMPLES = 1000
OUTPUT_DIR = "/home/ubuntu/21Days_Project/models/day3_distilbert"
MAX_SEQ_LENGTH = 256  # Reduced for faster training
RANDOM_STATE = 42

print("=" * 60)
print("Quick DistilBERT Training")
print("=" * 60)
print(f"Samples: {N_SAMPLES}")
print(f"Output: {OUTPUT_DIR}")
print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
print()

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Estimated time: ~30 minutes")
else:
    print(f"WARNING: Using CPU. Estimated time: ~2 hours")
print()

# Step 1: Generate synthetic email data
print("[1/5] Generating synthetic email data...")

np.random.seed(RANDOM_STATE)

# Email templates
phishing_templates = [
    {
        "subject": "URGENT: Verify Your Account Now",
        "body": "Dear Customer, Your account will be suspended within 24 hours unless you verify your information. Please click here immediately: {url} Provide your account number and SSN to prevent suspension.",
        "sender": "security@{domain}.com",
        "url": "http://{domain}.xyz/login"
    },
    {
        "subject": "Wire Transfer Required Immediately",
        "body": "URGENT: We need you to wire funds immediately. This is time sensitive. Click here to process: {url} Confirm your routing number and account number.",
        "sender": "payments@{domain}.com",
        "url": "http://{domain}-secure.net/verify"
    },
    {
        "subject": "Unusual Sign-In Detected",
        "body": "We detected unusual sign-in activity to your account. Verify your identity now: {url} Update your password and security questions immediately.",
        "sender": "alert@{domain}.com",
        "url": "http://{domain}-portal.xyz/secure"
    },
    {
        "subject": "Confirm Your Payment Information",
        "body": "Your payment could not be processed. Update your billing information immediately: {url} Provide your credit card number and SSN.",
        "sender": "billing@{domain}.com",
        "url": "http://{domain}-verify.com/update"
    },
    {
        "subject": "Act Now: Account Compromised",
        "body": "Your account has been compromised! Act now to secure it: {url} Verify your identity with SSN and account number.",
        "sender": "support@{domain}.com",
        "url": "http://secure-{domain}.xyz/restore"
    }
]

legitimate_templates = [
    {
        "subject": "Your Monthly Statement is Available",
        "body": "Dear Customer, Your monthly statement is now available. Log in to your account to view it: https://{domain}.com This is an automated message. Please do not reply.",
        "sender": "notifications@{domain}.com",
        "url": "https://{domain}.com/login"
    },
    {
        "subject": "Welcome to Our Service",
        "body": "Thank you for signing up! Your account has been created successfully. You can log in at: https://{domain}.com If you have any questions, contact our support team.",
        "sender": "welcome@{domain}.com",
        "url": "https://{domain}.com"
    },
    {
        "subject": "Password Reset Request",
        "body": "You requested a password reset. If this was you, click here: https://{domain}.com/reset If you did not request this, please ignore this email.",
        "sender": "noreply@{domain}.com",
        "url": "https://{domain}.com/reset"
    },
    {
        "subject": "Your Order Has Been Shipped",
        "body": "Good news! Your order has been shipped and is on its way. Track your package at: https://{domain}.com/track Expected delivery: 3-5 business days.",
        "sender": "shipping@{domain}.com",
        "url": "https://{domain}.com/track"
    },
    {
        "subject": "Account Security Update",
        "body": "We've updated our security policies. Please review the new terms at: https://{domain}.com/terms Your account security is important to us.",
        "sender": "info@{domain}.com",
        "url": "https://{domain}.com/terms"
    }
]

banks = ["chase", "wellsfargo", "bankofamerica", "citi", "usbank"]

# Generate phishing emails
phishing_emails = []
for i in range(N_SAMPLES // 2):
    template = phishing_templates[i % len(phishing_templates)]
    bank = banks[i % len(banks)]
    phishing_emails.append({
        "text": f"[SUBJECT] {template['subject']} [BODY] {template['body']} [SENDER] {template['sender'].format(domain=bank)} [URL] {template['url'].format(domain=bank)}",
        "label": 1
    })

# Generate legitimate emails
legitimate_emails = []
for i in range(N_SAMPLES // 2):
    template = legitimate_templates[i % len(legitimate_templates)]
    bank = banks[i % len(banks)]
    legitimate_emails.append({
        "text": f"[SUBJECT] {template['subject']} [BODY] {template['body']} [SENDER] {template['sender'].format(domain=bank)} [URL] {template['url'].format(domain=bank)}",
        "label": 0
    })

# Combine and shuffle
all_emails = phishing_emails + legitimate_emails
np.random.shuffle(all_emails)

print(f"  Generated {len(all_emails)} emails")
print(f"  Phishing: {len(phishing_emails)}")
print(f"  Legitimate: {len(legitimate_emails)}")

# Step 2: Initialize tokenizer
print("\n[2/5] Initializing tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
print("  Tokenizer: distilbert-base-uncased")

# Step 3: Create dataset
print("\n[3/5] Creating dataset...")

class EmailDataset(Dataset):
    def __init__(self, emails, tokenizer, max_length):
        self.emails = emails
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        email = self.emails[idx]
        encoding = self.tokenizer(
            email['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(email['label'], dtype=torch.long)
        }

# Split data
split = int(0.8 * len(all_emails))
train_emails = all_emails[:split]
test_emails = all_emails[split:]

train_dataset = EmailDataset(train_emails, tokenizer, MAX_SEQ_LENGTH)
test_dataset = EmailDataset(test_emails, tokenizer, MAX_SEQ_LENGTH)

print(f"  Training: {len(train_dataset)} samples")
print(f"  Testing: {len(test_dataset)} samples")

# Step 4: Initialize model
print("\n[4/5] Initializing model...")

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

print(f"  Model: distilbert-base-uncased")
print(f"  Parameters: {model.num_parameters():,}")

# Step 5: Train model
print("\n[5/5] Training model...")

training_args = TrainingArguments(
    output_dir=f'{OUTPUT_DIR}/tmp',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=f'{OUTPUT_DIR}/logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="auprc",
    greater_is_better=True,
    report_to="none",  # Disable wandb/tensorboard
    learning_rate=2e-5,
    seed=RANDOM_STATE
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    probs = predictions[:, 1]  # Probability of class 1 (phishing)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    auprc = average_precision_score(labels, probs)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auprc': auprc
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

start_time = time.time()
trainer.train()
training_time = time.time() - start_time

print(f"  Training completed in {training_time/60:.2f} minutes")

# Evaluate
print("\nEvaluating model...")
metrics = trainer.evaluate()
print(f"  Final AUPRC: {metrics['eval_auprc']:.4f}")
print(f"  Final Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"  Final F1: {metrics['eval_f1']:.4f}")

# Step 6: Save model
print(f"\nSaving model to {OUTPUT_DIR}...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save model and tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save metadata
metadata = {
    "model_type": "DistilBERT",
    "version": "1.0.0-quick",
    "base_model": "distilbert-base-uncased",
    "training_date": datetime.now().isoformat(),
    "training_samples": N_SAMPLES,
    "max_seq_length": MAX_SEQ_LENGTH,
    "num_labels": 2,
    "performance": {
        "auprc": float(metrics['eval_auprc']),
        "accuracy": float(metrics['eval_accuracy']),
        "f1": float(metrics['eval_f1']),
        "precision": float(metrics['eval_precision']),
        "recall": float(metrics['eval_recall'])
    },
    "hyperparameters": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "learning_rate": 2e-5,
        "max_seq_length": MAX_SEQ_LENGTH
    },
    "training_time_minutes": training_time / 60,
    "device_used": device
}

with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Save as pytorch_model.bin for compatibility with Day 3 project
torch.save(model.state_dict(), f"{OUTPUT_DIR}/pytorch_model.bin")

print(f"  Model saved: {OUTPUT_DIR}/")
print(f"  Metadata saved: {OUTPUT_DIR}/metadata.json")

print("\n" + "=" * 60)
print("DistilBERT training completed successfully!")
print("=" * 60)
print(f"\nModel location: {OUTPUT_DIR}")
print(f"Ready to use in API!")
