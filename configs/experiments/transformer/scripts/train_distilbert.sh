#!/bin/bash
# Train BERT classifier
python3 train.py \
    --model distilbert \
    --config experiments/configs/distilbert.yaml \
    --data data/raw/phishing_emails_hf.csv \
    --output checkpoints/distilbert \
    --epochs 3
