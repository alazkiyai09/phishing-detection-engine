#!/usr/bin/env python3
"""
Master script to run all quick training.
Trains both XGBoost and DistilBERT models using synthetic data.
Total time: ~5 minutes (XGBoost) + ~30 minutes (DistilBERT with GPU)
"""
import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Configuration
MODELS_DIR = "/home/ubuntu/21Days_Project/models"
SCRIPT_DIR = Path(__file__).parent

print("=" * 70)
print(" " * 15 + "QUICK MODEL TRAINING - ALL MODELS")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check models directory
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(f"{MODELS_DIR}/day2_xgboost", exist_ok=True)
os.makedirs(f"{MODELS_DIR}/day3_distilbert", exist_ok=True)

total_start = time.time()
results = {}

# Train XGBoost
print("\n" + "=" * 70)
print("PART 1: XGBoost Training (Classical ML)")
print("=" * 70)
print("Estimated time: ~5 minutes")
print()

start = time.time()
try:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "quick_train_xgboost.py")],
        check=True,
        capture_output=False
    )
    results['xgboost'] = 'SUCCESS'
    print("\n✅ XGBoost training completed successfully")
except subprocess.CalledProcessError as e:
    results['xgboost'] = 'FAILED'
    print(f"\n❌ XGBoost training failed: {e}")

xgboost_time = time.time() - start
print(f"Time taken: {xgboost_time/60:.2f} minutes")

# Train DistilBERT
print("\n" + "=" * 70)
print("PART 2: DistilBERT Training (Transformer)")
print("=" * 70)
print("Estimated time: ~30 minutes (GPU) or ~2 hours (CPU)")
print()

start = time.time()
try:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "quick_train_distilbert.py")],
        check=True,
        capture_output=False
    )
    results['distilbert'] = 'SUCCESS'
    print("\n✅ DistilBERT training completed successfully")
except subprocess.CalledProcessError as e:
    results['distilbert'] = 'FAILED'
    print(f"\n❌ DistilBERT training failed: {e}")

distilbert_time = time.time() - start
print(f"Time taken: {distilbert_time/60:.2f} minutes")

# Summary
total_time = time.time() - total_start

print("\n" + "=" * 70)
print(" " * 25 + "TRAINING SUMMARY")
print("=" * 70)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nResults:")
print(f"  XGBoost:     {'✅ SUCCESS' if results['xgboost'] == 'SUCCESS' else '❌ FAILED'} ({xgboost_time/60:.1f} min)")
print(f"  DistilBERT:  {'✅ SUCCESS' if results['distilbert'] == 'SUCCESS' else '❌ FAILED'} ({distilbert_time/60:.1f} min)")
print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print()

# Check model files
print("Model files:")
print(f"  XGBoost: {MODELS_DIR}/day2_xgboost/")
print(f"    ├── xgboost_phishing_classifier.json")
xgboost_exists = os.path.exists(f"{MODELS_DIR}/day2_xgboost/xgboost_phishing_classifier.json")
print(f"    └── {'✅' if xgboost_exists else '❌'} File exists")

print(f"\n  DistilBERT: {MODELS_DIR}/day3_distilbert/")
print(f"    ├── pytorch_model.bin")
distilbert_exists = os.path.exists(f"{MODELS_DIR}/day3_distilbert/pytorch_model.bin")
print(f"    ├── config.json")
config_exists = os.path.exists(f"{MODELS_DIR}/day3_distilbert/config.json")
print(f"    └── {'✅' if distilbert_exists else '❌'} Files exist")

# Final status
all_success = results.get('xgboost') == 'SUCCESS' and results.get('distilbert') == 'SUCCESS'

print("\n" + "=" * 70)
if all_success:
    print(" " * 20 + "🎉 ALL MODELS TRAINED SUCCESSFULLY! 🎉")
else:
    print(" " * 20 + "⚠️  SOME MODELS FAILED TO TRAIN")
print("=" * 70)

if all_success:
    print("\n✨ Next steps:")
    print("1. Start the API: docker-compose up -d")
    print("2. Check health: curl http://localhost:8000/health")
    print("3. Test email analysis: curl -X POST http://localhost:8000/api/v1/analyze/email")
    print("4. View models: curl http://localhost:8000/api/v1/models")
    print()
    print("Your models are ready to use!")

sys.exit(0 if all_success else 1)
