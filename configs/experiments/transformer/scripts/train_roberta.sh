#!/bin/bash
# Train BERT classifier
python3 train.py \
    --model roberta \
    --config experiments/configs/roberta.yaml \
    --data data/raw/phishing_emails_hf.csv \
    --output checkpoints/roberta \
    --epochs 3
