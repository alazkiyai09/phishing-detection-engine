# Quick Training Scripts

Fast model training for testing and development.

## Overview

These scripts generate synthetic email data and train models quickly:
- **XGBoost**: ~5 minutes, CPU only
- **DistilBERT**: ~30 minutes (GPU) or ~2 hours (CPU)

## Quick Start

### Train All Models (Recommended)

```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api/scripts
python quick_train_all.py
```

This will:
1. Train XGBoost model (~5 min)
2. Train DistilBERT model (~30 min - 2 hrs)
3. Save models to `/home/ubuntu/21Days_Project/models/`

### Train Individual Models

**XGBoost only** (~5 minutes):
```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api/scripts
python quick_train_xgboost.py
```

**DistilBERT only** (~30 min - 2 hrs):
```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api/scripts
python quick_train_distilbert.py
```

## Requirements

### For XGBoost
```bash
pip install xgboost scikit-learn pandas numpy
```

### For DistilBERT
```bash
pip install torch transformers scikit-learn
```

**For GPU support (much faster)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## What Gets Trained

### XGBoost Model
- **Samples**: 1,000 synthetic emails (500 phishing, 500 legitimate)
- **Features**: 60 engineered features (URL, header, sender, content, structural, linguistic, financial)
- **Training time**: ~5 minutes
- **File size**: ~500 KB
- **Expected AUPRC**: ~0.85-0.90

### DistilBERT Model
- **Samples**: 1,000 synthetic emails
- **Architecture**: DistilBERT-base-uncased (66M parameters)
- **Max sequence length**: 256 tokens
- **Training epochs**: 3
- **Training time**: ~30 min (GPU) or ~2 hrs (CPU)
- **File size**: ~250 MB
- **Expected AUPRC**: ~0.90-0.95

## Output Location

```
/home/ubuntu/21Days_Project/models/
├── day2_xgboost/
│   ├── xgboost_phishing_classifier.json
│   └── metadata.json
└── day3_distilbert/
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer_config.json
    └── metadata.json
```

## Performance Expectations

### XGBoost (quick_train_xgboost.py)

```
Metrics on test set:
- AUPRC: 0.85-0.90
- Accuracy: 0.85-0.90
- Training time: 3-5 minutes
```

**Top features** (typically):
1. `bank_impersonation_score`
2. `urgency_keyword_count`
3. `has_suspicious_tld`
4. `spf_fail`
5. `credential_harvesting_score`

### DistilBERT (quick_train_distilbert.py)

```
Metrics on test set:
- AUPRC: 0.90-0.95
- Accuracy: 0.90-0.95
- Training time: 30 min (GPU) or 2 hrs (CPU)
```

**Training progression** (typical):
```
Epoch 1: AUPRC ~0.85, Loss ~0.4
Epoch 2: AUPRC ~0.90, Loss ~0.2
Epoch 3: AUPRC ~0.93, Loss ~0.15
```

## Usage After Training

Once models are trained, you can use them in the API:

### Start API
```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api
docker-compose up -d
```

### Test Models
```bash
# Check model availability
curl http://localhost:8000/health | jq

# List models with performance
curl http://localhost:8000/api/v1/models | jq

# Analyze email with XGBoost
curl -X POST "http://localhost:8000/api/v1/analyze/email" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_email": "From: security@chase-secure-portal.xyz\nSubject: URGENT\n...",
    "model_type": "xgboost"
  }' | jq

# Analyze email with Transformer
curl -X POST "http://localhost:8000/api/v1/analyze/email" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_email": "From: security@chase-secure-portal.xyz\nSubject: URGENT\n...",
    "model_type": "transformer"
  }' | jq

# Use ensemble (both models)
curl -X POST "http://localhost:8000/api/v1/analyze/email" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_email": "From: security@chase-secure-portal.xyz\nSubject: URGENT\n...",
    "model_type": "ensemble"
  }' | jq
```

## Troubleshooting

### XGBoost Training Issues

**Problem**: Import error for sklearn
```bash
# Solution
pip install scikit-learn==1.3.2
```

**Problem**: Training takes too long
```bash
# Already optimized - should take < 5 minutes
# If slower, check CPU usage with htop
```

### DistilBERT Training Issues

**Problem**: CUDA out of memory
```bash
# Solution 1: Reduce batch size
# Edit script: per_device_train_batch_size=8

# Solution 2: Use CPU (slower but works)
export CUDA_VISIBLE_DEVICES=""
python quick_train_distilbert.py
```

**Problem**: Training is very slow on CPU
```bash
# This is expected - 2 hours on CPU vs 30 min on GPU
# Consider using Google Colab (free GPU) for faster training
```

**Problem**: Transformers not installed
```bash
# Solution
pip install transformers==4.35.2 torch==2.1.1
```

### Model Not Found by API

**Problem**: API says model not available
```bash
# Check file exists
ls -lh /home/ubuntu/21Days_Project/models/day2_xgboost/xgboost_phishing_classifier.json
ls -lh /home/ubuntu/21Days_Project/models/day3_distilbert/pytorch_model.bin

# Check permissions
chmod 644 /home/ubuntu/21Days_Project/models/day*/*.*

# Restart API
docker-compose restart api
```

## Customization

### Change Number of Samples

Edit the script:
```python
N_SAMPLES = 1000  # Change this
```

**Recommended values**:
- Quick test: 500
- Good balance: 1000 (default)
- Better accuracy: 5000
- Production: 10000+

### Change Training Epochs (DistilBERT)

Edit the script:
```python
num_train_epochs=3  # Change this
```

**More epochs = better accuracy but longer training**

### Use Different Base Model

Edit the script:
```python
# Change from DistilBERT to BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', ...)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
```

**Note**: BERT is larger (110M params) and slower than DistilBERT (66M params).

## Training on Google Colab (Free GPU)

If you don't have a GPU:

1. Upload `quick_train_distilbert.py` to Google Colab
2. Install dependencies:
   ```python
   !pip install transformers torch scikit-learn
   ```
3. Run the script
4. Download the trained model files
5. Copy to `/home/ubuntu/21Days_Project/models/day3_distilbert/`

## Next Steps

After training:
1. ✅ Models are ready to use
2. 🚀 Start the API: `docker-compose up -d`
3. 🧪 Test with sample emails
4. 📊 View metrics in Grafana
5. 🎯 If needed, fine-tune with real data

## For Production

These quick-trained models are **good for testing/demonstration** but **not for production**.

For production models:
- Use **real phishing datasets** (not synthetic)
- Train on **10,000+ samples**
- Use **proper validation** (temporal split)
- Conduct **thorough testing** (false positive analysis)

See: `docs/MODEL_TRAINING_GUIDE.md` for production training.
