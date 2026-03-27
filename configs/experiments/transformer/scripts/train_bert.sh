#!/bin/bash
# Train BERT classifier
python3 train.py \
    --model bert \
    --config experiments/configs/bert.yaml \
    --data data/raw/phishing_emails_hf.csv \
    --output checkpoints/bert \
    --epochs 3
