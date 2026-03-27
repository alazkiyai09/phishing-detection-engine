#!/usr/bin/env python3
"""
Quick XGBoost training script for testing.
Generates synthetic data and trains a model in ~5 minutes.
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
import xgboost as xgb

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/ubuntu/21Days_Project/phishing_email_analysis")

# Configuration
N_SAMPLES = 1000
OUTPUT_DIR = "/home/ubuntu/21Days_Project/models/day2_xgboost"
RANDOM_STATE = 42

print("=" * 60)
print("Quick XGBoost Training")
print("=" * 60)
print(f"Samples: {N_SAMPLES}")
print(f"Output: {OUTPUT_DIR}")
print()

# Step 1: Generate synthetic features
print("[1/4] Generating synthetic features...")

# Set random seed
np.random.seed(RANDOM_STATE)

# Generate 60 features (simplified version of Day 1 features)
feature_names = [
    # URL features (10)
    'url_count', 'has_ip_url', 'avg_url_length', 'max_url_length',
    'has_suspicious_tld', 'has_https', 'avg_subdomain_count',
    'has_url_shortener', 'special_char_ratio', 'has_port_specified',
    # Header features (10)
    'spf_pass', 'spf_fail', 'dkim_present', 'dkim_valid',
    'dmarc_pass', 'dmarc_fail', 'hop_count', 'reply_to_mismatch',
    'has_priority_flag', 'has_authentication_results',
    # Sender features (10)
    'is_freemail', 'display_name_mismatch', 'display_name_has_bank',
    'domain_age_days', 'has_numbers_in_domain', 'email_address_length',
    'domain_length', 'sender_name_length', 'has_reply_to_path', 'suspicious_pattern',
    # Content features (10)
    'urgency_keyword_count', 'cta_button_count', 'threat_language_count',
    'financial_term_count', 'immediate_action_count', 'verification_request_count',
    'click_here_count', 'password_request_count', 'account_suspended_count', 'url_in_body_count',
    # Structural features (10)
    'html_text_ratio', 'has_attachments', 'attachment_count',
    'has_executable_attachment', 'has_office_attachment', 'embedded_image_count',
    'external_image_count', 'has_forms', 'has_javascript', 'email_size_kb',
    # Linguistic features (10)
    'spelling_error_rate', 'grammar_score_proxy', 'formality_score',
    'reading_ease_score', 'sentence_count', 'avg_sentence_length',
    'exclamation_mark_count', 'question_mark_count', 'all_caps_ratio', 'punctuation_ratio',
    # Financial features (10)
    'bank_impersonation_score', 'wire_urgency_score', 'credential_harvesting_score',
    'invoice_terminology_density', 'account_number_request', 'routing_number_request',
    'ssn_request', 'payment_urgency_score', 'financial_institution_mentions', 'wire_transfer_keywords'
]

# Generate synthetic data
n_phishing = N_SAMPLES // 2
n_legitimate = N_SAMPLES - n_phishing

# Generate data for each feature separately to avoid dictionary key conflicts
def generate_phishing_features(n_samples):
    """Generate features for phishing emails."""
    data = {}

    # URL features (10) - higher suspicious values
    for feat in feature_names[0:10]:
        if feat == 'has_suspicious_tld':
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
        elif feat == 'has_https':
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        elif feat == 'has_ip_url':
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        else:
            data[feat] = np.random.beta(2, 5, size=n_samples)

    # Header features (10) - more failures
    for feat in feature_names[10:20]:
        if 'pass' in feat:
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        elif 'fail' in feat:
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
        else:
            data[feat] = np.random.beta(1, 3, size=n_samples)

    # Sender features (10)
    for feat in feature_names[20:30]:
        data[feat] = np.random.beta(2, 5, size=n_samples)

    # Content features (10) - higher urgency
    for feat in feature_names[30:40]:
        data[feat] = np.random.beta(3, 2, size=n_samples)

    # Structural features (10)
    for feat in feature_names[40:50]:
        data[feat] = np.random.beta(2, 3, size=n_samples)

    # Linguistic features (10)
    for feat in feature_names[50:60]:
        data[feat] = np.random.beta(3, 2, size=n_samples)

    # Financial features (10) - high risk
    for feat in feature_names[60:70]:
        data[feat] = np.random.beta(4, 1, size=n_samples)

    # Override specific features for phishing
    data['bank_impersonation_score'] = np.maximum(data['bank_impersonation_score'], 0.7)
    data['urgency_keyword_count'] = np.maximum(data['urgency_keyword_count'], 0.5)
    data['credential_harvesting_score'] = np.maximum(data['credential_harvesting_score'], 0.6)
    data['wire_urgency_score'] = np.maximum(data['wire_urgency_score'], 0.6)

    return pd.DataFrame(data)

def generate_legitimate_features(n_samples):
    """Generate features for legitimate emails."""
    data = {}

    # URL features (10) - cleaner
    for feat in feature_names[0:10]:
        if feat == 'has_suspicious_tld':
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        elif feat == 'has_https':
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9])
        elif feat == 'has_ip_url':
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])
        else:
            data[feat] = np.random.beta(1, 5, size=n_samples)

    # Header features (10) - more passes
    for feat in feature_names[10:20]:
        if 'pass' in feat:
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8])
        elif 'fail' in feat:
            data[feat] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        else:
            data[feat] = np.random.beta(4, 1, size=n_samples)

    # Sender features (10)
    for feat in feature_names[20:30]:
        data[feat] = np.random.beta(1, 4, size=n_samples)

    # Content features (10) - less urgency
    for feat in feature_names[30:40]:
        data[feat] = np.random.beta(1, 3, size=n_samples)

    # Structural features (10)
    for feat in feature_names[40:50]:
        data[feat] = np.random.beta(1, 4, size=n_samples)

    # Linguistic features (10)
    for feat in feature_names[50:60]:
        data[feat] = np.random.beta(2, 3, size=n_samples)

    # Financial features (10) - low risk
    for feat in feature_names[60:70]:
        data[feat] = np.random.beta(1, 4, size=n_samples)

    # Override specific features for legitimate
    data['bank_impersonation_score'] = np.minimum(data['bank_impersonation_score'], 0.3)
    data['urgency_keyword_count'] = np.minimum(data['urgency_keyword_count'], 0.3)
    data['credential_harvesting_score'] = np.minimum(data['credential_harvesting_score'], 0.2)
    data['wire_urgency_score'] = np.minimum(data['wire_urgency_score'], 0.2)

    return pd.DataFrame(data)

# Combine into DataFrame
df_phishing = generate_phishing_features(n_phishing)
df_phishing['label'] = 1

df_legitimate = generate_legitimate_features(n_legitimate)
df_legitimate['label'] = 0

df = pd.concat([df_phishing, df_legitimate], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"  Generated {len(df)} samples ({n_phishing} phishing, {n_legitimate} legitimate)")
print(f"  Features: {len(feature_names)}")

# Step 2: Prepare data for training
print("\n[2/4] Preparing data for training...")

# Separate features and labels
X = df[feature_names]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  Training: {len(X_train)} samples")
print(f"  Testing: {len(X_test)} samples")

# Step 3: Train XGBoost model
print("\n[3/4] Training XGBoost model...")

start_time = time.time()

model = xgb.XGBClassifier(
    n_estimators=100,  # Reduced for faster training
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1  # Use all CPU cores
)

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"  Training completed in {training_time:.2f} seconds")

# Step 4: Evaluate model
print("\n[4/4] Evaluating model...")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
auprc = average_precision_score(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
accuracy = (y_pred == y_test).mean()

print(f"  AUPRC: {auprc:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# Feature importance
importance = model.feature_importances_
feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

print("\nTop 10 Important Features:")
for feat, imp in feature_importance[:10]:
    print(f"  {feat}: {imp:.4f}")

# Step 5: Save model
print(f"\n[5/5] Saving model to {OUTPUT_DIR}...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save model
model.save_model(f"{OUTPUT_DIR}/xgboost_phishing_classifier.json")

# Save metadata
metadata = {
    "model_type": "XGBoost",
    "version": "1.0.0-quick",
    "training_date": datetime.now().isoformat(),
    "training_samples": N_SAMPLES,
    "features": feature_names,
    "n_features": len(feature_names),
    "performance": {
        "auprc": float(auprc),
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy)
    },
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    "training_time_seconds": training_time
}

with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Model saved: {OUTPUT_DIR}/xgboost_phishing_classifier.json")
print(f"  Metadata saved: {OUTPUT_DIR}/metadata.json")

print("\n" + "=" * 60)
print("XGBoost training completed successfully!")
print("=" * 60)
print(f"\nModel location: {OUTPUT_DIR}")
print(f"Ready to use in API!")
