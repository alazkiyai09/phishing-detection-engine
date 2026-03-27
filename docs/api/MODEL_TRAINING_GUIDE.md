# Model Training Guide

This guide explains how to train the models required for the Unified Phishing Detection API.

## Overview

The API requires three models:
1. **Day 2: XGBoost Model** - Classical ML with 60+ features
2. **Day 3: Transformer Model** - DistilBERT for text classification
3. **Day 4: Multi-Agent System** - GLM-powered agents (no training needed, uses API)

## Model Storage

All trained models should be stored in:
```
/home/ubuntu/21Days_Project/models/
```

---

## Day 2: XGBoost Model Training

### Prerequisites

```bash
cd /home/ubuntu/21Days_Project/day2_classical_ml_benchmark
pip install -r requirements.txt
```

### Training Steps

1. **Prepare Data** (using Day 1 features)

```python
# Generate synthetic features for testing
python generate_sample_features.py
```

Or use your own phishing dataset:
- Format: CSV with columns matching the 60+ features from Day 1
- Label column: `label` (0 = legitimate, 1 = phishing)
- Required columns: All feature columns from `phishing_email_analysis`

2. **Run Benchmark** (this will train all models including XGBoost)

```bash
python -m src.benchmark
```

3. **Save XGBoost Model**

The benchmark should save models, but if not, use this script:

```python
import json
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score

# Load your data
df = pd.read_csv("your_data.csv")

# Separate features and labels
X = df.drop(columns=['label', 'email_id', 'date'], errors='ignore')
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auprc = average_precision_score(y_test, y_prob)
print(f"AUPRC: {auprc:.4f}")
print(classification_report(y_test, y_pred))

# Save model
output_dir = "/home/ubuntu/21Days_Project/models/day2_xgboost"
import os
os.makedirs(output_dir, exist_ok=True)

# Save as JSON
model.save_model(f"{output_dir}/xgboost_phishing_classifier.json")

# Save metadata
metadata = {
    "model_type": "XGBoost",
    "version": "1.0.0",
    "training_date": pd.Timestamp.now().isoformat(),
    "features": list(X.columns),
    "n_features": len(X.columns),
    "performance": {
        "auprc": float(auprc),
        "accuracy": float((y_pred == y_test).mean())
    },
    "hyperparameters": model.get_params()
}

with open(f"{output_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved to {output_dir}")
```

### Expected Output

```
/home/ubuntu/21Days_Project/models/day2_xgboost/
├── xgboost_phishing_classifier.json
└── metadata.json
```

### Minimum Performance Requirements

- **AUPRC**: > 0.90 (financial phishing)
- **Recall**: > 0.95 (minimize false negatives)
- **FPR**: < 0.01 (false positive rate)

---

## Day 3: Transformer Model Training

### Prerequisites

```bash
cd /home/ubuntu/21Days_Project/day3_transformer_phishing
pip install -r requirements.txt
```

### Training Steps

1. **Download/Prepare Dataset**

```bash
# Option 1: Download HuggingFace dataset
python data/download_hf_dataset.py

# Option 2: Generate synthetic data
python data/generate_synthetic_data.py --n_samples 1000
```

2. **Configure Training**

Edit `experiments/configs/distilbert.yaml`:

```yaml
model:
  name: distilbert
  num_labels: 2

training:
  learning_rate: 2e-5
  batch_size: 16
  num_epochs: 3
  warmup_ratio: 0.1
  gradient_accumulation_steps: 1

data:
  max_seq_length: 512
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

3. **Run Training**

```bash
python train.py --config experiments/configs/distilbert.yaml
```

4. **Verify Model Files**

Check that checkpoints are created:

```bash
ls -lh checkpoints/
```

Expected output:
```
checkpoints/distilbert/
├── best_model.pt           # Trained model weights
├── config.yaml            # Training configuration
├── training_history.json  # Metrics over time
└── tokenizer_config.json  # Tokenizer settings
```

5. **Move to Models Directory**

```bash
# Create output directory
mkdir -p /home/ubuntu/21Days_Project/models/day3_distilbert

# Copy model files
cp checkpoints/distilbert/best_model.pt /home/ubuntu/21Days_Project/models/day3_distilbert/pytorch_model.bin
cp checkpoints/distilbert/tokenizer_config.json /home/ubuntu/21Days_Project/models/day3_distilbert/
cp checkpoints/distilbert/config.yaml /home/ubuntu/21Days_Project/models/day3_distilbert/
```

### Manual Training Script

If automated training doesn't work:

```python
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load/preprocess your data
# ... (see day3_transformer_phishing/src/data/dataset.py)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints/distilbert",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="auprc",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save
model.save_pretrained("/home/ubuntu/21Days_Project/models/day3_distilbert")
tokenizer.save_pretrained("/home/ubuntu/21Days_Project/models/day3_distilbert")
```

### Expected Output

```
/home/ubuntu/21Days_Project/models/day3_distilbert/
├── pytorch_model.bin         # Model weights
├── config.yaml              # Model config
├── tokenizer_config.json    # Tokenizer config
├── vocab.txt                # Vocabulary (if not using base)
└── metadata.json            # Training metadata
```

### Minimum Performance Requirements

- **AUPRC**: > 0.92 (better than classical ML)
- **Inference Speed**: < 1s per email (p95)
- **Model Size**: < 300MB (DistilBERT)

---

## Day 4: Multi-Agent System

No training required! The multi-agent system uses GLM API and requires no model files.

### Setup GLM API

1. **Get API Key**
   - Visit: https://open.bigmodel.cn/
   - Sign up and get API key
   - Set environment variable:

```bash
export GLM_API_KEY="your_api_key_here"
```

2. **Test GLM Backend**

```bash
cd /home/ubuntu/21Days_Project/multi_agent_phishing_detector

# Run demo
python -m src.main demo
```

---

## Verification

After training all models, verify the setup:

```bash
# Check model files exist
ls -lh /home/ubuntu/21Days_Project/models/

# Expected output:
# day2_xgboost/
#   ├── xgboost_phishing_classifier.json
#   └── metadata.json
# day3_distilbert/
#   ├── pytorch_model.bin
#   ├── config.yaml
#   └── tokenizer_config.json
```

---

## Troubleshooting

### XGBoost Issues

**Problem**: `Feature names mismatch`
```bash
# Solution: Ensure Day 1 features match exactly
python -c "from phishing_email_analysis.src.transformers import PhishingFeaturePipeline; print(PhishingFeaturePipeline().get_feature_names())"
```

### Transformer Issues

**Problem**: CUDA out of memory
```bash
# Solution: Use smaller batch size or CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

**Problem**: Slow training
```bash
# Solution: Use DistilBERT instead of BERT (faster, smaller)
# Change config to use distilbert instead of bert
```

### GLM API Issues

**Problem**: Authentication error
```bash
# Solution: Verify API key
echo $GLM_API_KEY  # Should show your key

# Test connection
curl -X POST "https://open.bigmodel.cn/api/paas/v4/chat/completions" \
  -H "Authorization: Bearer $GLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4-flash","messages":[{"role":"user","content":"test"}]}'
```

---

## Next Steps

After training models:

1. Start the API:
```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api
uvicorn app.main:app --reload
```

2. Check health endpoint:
```bash
curl http://localhost:8000/health
```

3. Verify all models are loaded:
```bash
curl http://localhost:8000/api/v1/models
```

---

## Performance Benchmarking

After training, benchmark each model:

```python
import time

# XGBoost should be < 200ms
start = time.time()
prediction = xgb_model.predict(features)
print(f"XGBoost: {(time.time() - start) * 1000:.2f}ms")

# Transformer should be < 1s
start = time.time()
prediction = transformer_model.predict(email)
print(f"Transformer: {(time.time() - start) * 1000:.2f}ms")

# Multi-agent will be 3-5s (expected)
```

---

For questions or issues, refer to individual project READMEs:
- Day 1: `/home/ubuntu/21Days_Project/phishing_email_analysis/README.md`
- Day 2: `/home/ubuntu/21Days_Project/day2_classical_ml_benchmark/README.md`
- Day 3: `/home/ubuntu/21Days_Project/day3_transformer_phishing/README.md`
- Day 4: `/home/ubuntu/21Days_Project/multi_agent_phishing_detector/README.md`
