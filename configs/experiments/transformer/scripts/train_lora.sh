#!/bin/bash
# Train BERT classifier
python3 train.py \
    --model lora-bert \
    --config experiments/configs/lora_bert.yaml \
    --data data/raw/phishing_emails_hf.csv \
    --output checkpoints/lora_bert \
    --epochs 3
