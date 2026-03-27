"""
Generate sample features for benchmark testing.

This script creates a synthetic feature dataset for testing the Day 2 benchmark
without requiring the full Day 1 pipeline to run.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_features(n_samples=1000, phishing_ratio=0.5, random_state=42):
    """
    Generate synthetic feature dataset for testing.

    Creates 60+ features similar to Day 1 pipeline output.

    Args:
        n_samples: Total number of samples
        phishing_ratio: Ratio of phishing emails
        random_state: Random seed

    Returns:
        DataFrame with features and labels
    """
    np.random.seed(random_state)

    n_phishing = int(n_samples * phishing_ratio)
    n_legitimate = n_samples - n_phishing

    # URL Features (10)
    url_features = {
        "url_count": np.random.uniform(0, 1, n_samples),
        "has_ip_url": np.random.randint(0, 2, n_samples).astype(float),
        "avg_url_length": np.random.uniform(0, 1, n_samples),
        "has_suspicious_tld": np.random.randint(0, 2, n_samples).astype(float),
        "has_https": np.random.randint(0, 2, n_samples).astype(float),
        "avg_subdomain_count": np.random.uniform(0, 1, n_samples),
        "has_url_shortener": np.random.randint(0, 2, n_samples).astype(float),
        "special_char_ratio": np.random.uniform(0, 1, n_samples),
        "has_port_specified": np.random.randint(0, 2, n_samples).astype(float),
        "max_url_length": np.random.uniform(0, 1, n_samples),
    }

    # Header Features (10)
    header_features = {
        "spf_pass": np.random.randint(0, 2, n_samples).astype(float),
        "spf_fail": np.random.randint(0, 2, n_samples).astype(float),
        "dkim_present": np.random.randint(0, 2, n_samples).astype(float),
        "dkim_valid": np.random.randint(0, 2, n_samples).astype(float),
        "dmarc_pass": np.random.randint(0, 2, n_samples).astype(float),
        "dmarc_fail": np.random.randint(0, 2, n_samples).astype(float),
        "hop_count": np.random.uniform(0, 1, n_samples),
        "reply_to_mismatch": np.random.randint(0, 2, n_samples).astype(float),
        "has_priority_flag": np.random.randint(0, 2, n_samples).astype(float),
        "has_authentication_results": np.random.randint(0, 2, n_samples).astype(float),
    }

    # Sender Features (10)
    sender_features = {
        "is_freemail": np.random.randint(0, 2, n_samples).astype(float),
        "display_name_mismatch": np.random.randint(0, 2, n_samples).astype(float),
        "display_name_has_bank": np.random.randint(0, 2, n_samples).astype(float),
        "domain_age_days": np.random.uniform(0, 1, n_samples),
        "has_numbers_in_domain": np.random.randint(0, 2, n_samples).astype(float),
        "email_address_length": np.random.uniform(0, 1, n_samples),
        "domain_length": np.random.uniform(0, 1, n_samples),
        "sender_name_length": np.random.uniform(0, 1, n_samples),
        "has_reply_to_path": np.random.randint(0, 2, n_samples).astype(float),
        "suspicious_pattern": np.random.randint(0, 2, n_samples).astype(float),
    }

    # Content Features (10)
    content_features = {
        "urgency_keyword_count": np.random.uniform(0, 1, n_samples),
        "cta_button_count": np.random.uniform(0, 1, n_samples),
        "threat_language_count": np.random.uniform(0, 1, n_samples),
        "financial_term_count": np.random.uniform(0, 1, n_samples),
        "immediate_action_count": np.random.uniform(0, 1, n_samples),
        "verification_request_count": np.random.uniform(0, 1, n_samples),
        "click_here_count": np.random.uniform(0, 1, n_samples),
        "password_request_count": np.random.uniform(0, 1, n_samples),
        "account_suspended_count": np.random.uniform(0, 1, n_samples),
        "url_in_body_count": np.random.uniform(0, 1, n_samples),
    }

    # Structural Features (10)
    structural_features = {
        "html_text_ratio": np.random.uniform(0, 1, n_samples),
        "has_attachments": np.random.randint(0, 2, n_samples).astype(float),
        "attachment_count": np.random.uniform(0, 1, n_samples),
        "has_executable_attachment": np.random.randint(0, 2, n_samples).astype(float),
        "has_office_attachment": np.random.randint(0, 2, n_samples).astype(float),
        "embedded_image_count": np.random.uniform(0, 1, n_samples),
        "external_image_count": np.random.uniform(0, 1, n_samples),
        "has_forms": np.random.randint(0, 2, n_samples).astype(float),
        "has_javascript": np.random.randint(0, 2, n_samples).astype(float),
        "email_size_kb": np.random.uniform(0, 1, n_samples),
    }

    # Linguistic Features (10)
    linguistic_features = {
        "spelling_error_rate": np.random.uniform(0, 1, n_samples),
        "grammar_score_proxy": np.random.uniform(0, 1, n_samples),
        "formality_score": np.random.uniform(0, 1, n_samples),
        "reading_ease_score": np.random.uniform(0, 1, n_samples),
        "sentence_count": np.random.uniform(0, 1, n_samples),
        "avg_sentence_length": np.random.uniform(0, 1, n_samples),
        "exclamation_mark_count": np.random.uniform(0, 1, n_samples),
        "question_mark_count": np.random.uniform(0, 1, n_samples),
        "all_caps_ratio": np.random.uniform(0, 1, n_samples),
        "punctuation_ratio": np.random.uniform(0, 1, n_samples),
    }

    # Financial Features (10)
    financial_features = {
        "bank_impersonation_score": np.random.uniform(0, 1, n_samples),
        "wire_urgency_score": np.random.uniform(0, 1, n_samples),
        "credential_harvesting_score": np.random.uniform(0, 1, n_samples),
        "invoice_terminology_density": np.random.uniform(0, 1, n_samples),
        "account_number_request": np.random.randint(0, 2, n_samples).astype(float),
        "routing_number_request": np.random.randint(0, 2, n_samples).astype(float),
        "ssn_request": np.random.randint(0, 2, n_samples).astype(float),
        "payment_urgency_score": np.random.uniform(0, 1, n_samples),
        "financial_institution_mentions": np.random.uniform(0, 1, n_samples),
        "wire_transfer_keywords": np.random.uniform(0, 1, n_samples),
    }

    # Combine all features
    all_features = {}
    all_features.update(url_features)
    all_features.update(header_features)
    all_features.update(sender_features)
    all_features.update(content_features)
    all_features.update(structural_features)
    all_features.update(linguistic_features)
    all_features.update(financial_features)

    # Create DataFrame
    features_df = pd.DataFrame(all_features)

    # Make phishing emails more distinct (bias features for phishing class)
    for col in features_df.columns:
        # Phishing emails (first n_phishing rows) get higher feature values
        phishing_bias = np.random.uniform(0.3, 0.7, n_samples)
        features_df[col] = np.where(
            np.arange(n_samples) < n_phishing,
            np.clip(features_df[col] + phishing_bias * 0.3, 0, 1),
            features_df[col]
        )

    # Add labels and metadata
    features_df["label"] = [1] * n_phishing + [0] * n_legitimate
    features_df["phishing_type"] = (
        ["financial"] * (n_phishing // 3) +
        ["generic"] * (n_phishing // 3) +
        ["spear"] * (n_phishing - 2 * (n_phishing // 3)) +
        [None] * n_legitimate
    )

    # Add date column (for temporal split)
    base_date = pd.Timestamp("2024-01-01")
    features_df["date"] = [
        base_date + pd.Timedelta(days=np.random.randint(0, 180))
        for _ in range(n_samples)
    ]

    # Shuffle
    features_df = features_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return features_df


def main():
    """Generate and save sample features."""
    print("="*80)
    print("GENERATING SAMPLE FEATURES FOR DAY 2 BENCHMARK")
    print("="*80)

    n_samples = 1000
    print(f"\n📊 Generating {n_samples} samples with 60+ features...")

    features_df = generate_sample_features(n_samples=n_samples, random_state=42)

    # Save
    output_path = Path("/home/ubuntu/21Days_Project/day2_classical_ml_benchmark/data/sample_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"\n✅ Features saved to {output_path}")
    print(f"   Shape: {features_df.shape}")
    print(f"   Features: {len([c for c in features_df.columns if c not in ['label', 'phishing_type', 'date']])}")
    print(f"\n   Class distribution:")
    print(f"   Phishing: {(features_df['label'] == 1).sum()} ({features_df['label'].mean()*100:.1f}%)")
    print(f"   Legitimate: {(features_df['label'] == 0).sum()} ({(1-features_df['label'].mean())*100:.1f}%)")

    if "phishing_type" in features_df.columns:
        print(f"\n   Phishing types:")
        for ptype in ["financial", "generic", "spear"]:
            count = (features_df["phishing_type"] == ptype).sum()
            print(f"   {ptype.capitalize()}: {count}")

    print("\n" + "="*80)
    print("✅ SUCCESS: Sample features ready for benchmark")
    print("="*80)
    print(f"\nTo run the benchmark:")
    print(f"  cd /home/ubuntu/21Days_Project/day2_classical_ml_benchmark")
    print(f"  python3 src/benchmark.py")


if __name__ == "__main__":
    main()
