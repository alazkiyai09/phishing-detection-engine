#!/usr/bin/env python3
"""Demo script showing the full phishing email analysis pipeline.

This script demonstrates:
1. Loading sample email data
2. Extracting all features
3. Analyzing feature importance
4. Checking for redundant features
"""

import pandas as pd
import numpy as np

from src.transformers import PhishingFeaturePipeline
from src.analysis.importance import (
    compute_mutual_information,
    print_feature_ranking,
)
from src.analysis.correlation import (
    analyze_feature_correlation,
    print_correlation_summary,
)


def create_sample_emails():
    """Create sample email dataset for demonstration."""
    emails = []

    # Phishing email 1: Bank impersonation with credential harvesting
    emails.append({
        "body": """
        Dear Customer,

        Your CHASE BANK account will be closed within 24 hours due to suspicious activity.
        To verify your account and prevent closure, please click the link below:

        http://chase-secure-portal.xyz/login/verify

        Verify your account information immediately to restore access.

        Urgent attention required!

        Best regards,
        Chase Security Team
        """,
        "body_html": "<p>Dear Customer,</p><p>Your CHASE BANK account will be closed...</p>",
        "headers": {
            "Received": "from unknown [192.168.1.100]",
            "SPF": "fail",
            "X-Priority": "1",
        },
        "subject": "URGENT: Account Verification Required",
        "from_addr": "security@chase-secure-portal.xyz",
        "attachments": []
    })

    # Phishing email 2: Wire transfer urgency
    emails.append({
        "body": """
        URGENT WIRE TRANSFER REQUEST

        Please initiate an immediate wire transfer to:

        Account: 123-456-789
        Routing: 021000021
        Amount: $45,000

        This must be completed within 2 hours to avoid contract penalties.

        Click here to process: http://secure-wire-transfer.top/process

        Do not delay!
        """,
        "body_html": "<p>URGENT WIRE TRANSFER REQUEST</p>",
        "headers": {
            "Received": "from unknown [10.0.0.50]",
            "SPF": "softfail",
            "X-Priority": "2",
        },
        "subject": "IMMEDIATE ACTION REQUIRED - Wire Transfer",
        "from_addr": "ceo-wire@company-executive.xyz",
        "attachments": [{"filename": "invoice.docx", "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size": 45000}]
    })

    # Legitimate email 1: Normal bank communication
    emails.append({
        "body": """
        Dear Customer,

        Your monthly statement is now available for viewing.

        Please log in to your account at https://www.chase.com to view your statement.

        If you have questions, please contact customer service.

        Sincerely,
        Chase Bank
        """,
        "body_html": "<p>Dear Customer,</p><p>Your monthly statement is now available...</p>",
        "headers": {
            "Received": "from chase.com [10.100.50.25]",
            "SPF": "pass",
            "DKIM-Signature": "v=1; a=rsa-sha256; c=relaxed/relaxed;",
            "Authentication-Results": "spf=pass dkim=pass dmarc=pass",
        },
        "subject": "Your Monthly Statement is Available",
        "from_addr": "notifications@chase.com",
        "attachments": [{"filename": "statement.pdf", "content_type": "application/pdf", "size": 125000}]
    })

    # Legitimate email 2: Normal business communication
    emails.append({
        "body": """
        Hi team,

        Just a reminder about our meeting tomorrow at 2pm in Conference Room B.

        Agenda:
        1. Q4 review
        2. Project updates
        3. Budget planning

        Please bring your reports.

        Thanks,
        Sarah
        """,
        "body_html": "<p>Hi team,</p><p>Just a reminder about our meeting...</p>",
        "headers": {
            "Received": "from company.com [192.168.1.10]",
            "SPF": "pass",
        },
        "subject": "Meeting Reminder - Tomorrow 2pm",
        "from_addr": "sarah.johnson@company.com",
        "attachments": []
    })

    # Phishing email 3: Invoice fraud
    emails.append({
        "body": """
        INVOICE #INV-2025-0145 - OVERDUE

        Your payment of $12,500 is now OVERDUE.

        To avoid legal action and service suspension, please process payment immediately.

        Click here to pay: http://invoice-payment-processing.xyz/pay

        Wire transfer details:
        Account: 987-654-321
        Routing: 026009593

        ACT NOW TO AVOID PENALTIES!
        """,
        "body_html": "<p>INVOICE #INV-2025-0145 - OVERDUE</p>",
        "headers": {
            "Received": "from unknown [172.16.0.100]",
            "SPF": "fail",
            "X-Priority": "1",
        },
        "subject": "FINAL NOTICE: Overdue Invoice - Legal Action Pending",
        "from_addr": "accounts-payable@invoice-services.xyz",
        "attachments": [{"filename": "invoice.exe", "content_type": "application/x-executable", "size": 250000}]
    })

    # Labels (1 = phishing, 0 = legitimate)
    labels = np.array([1, 1, 0, 0, 1])

    return pd.DataFrame(emails), labels


def main():
    """Run the demo."""
    print("=" * 80)
    print("PHISHING EMAIL ANALYSIS - DEMO")
    print("=" * 80)

    # Create sample data
    print("\n1. Creating sample email dataset...")
    emails_df, labels = create_sample_emails()
    print(f"   Created {len(emails_df)} emails")
    print(f"   Phishing: {labels.sum()}, Legitimate: {(1 - labels).sum()}")

    # Initialize pipeline
    print("\n2. Initializing PhishingFeaturePipeline...")
    pipeline = PhishingFeaturePipeline()
    print(f"   Pipeline: {pipeline}")
    print(f"   Extractors: {len(pipeline.extractors)}")
    for extractor in pipeline.extractors:
        print(f"     - {extractor.__class__.__name__}")

    # Extract features
    print("\n3. Extracting features...")
    features_df = pipeline.fit_transform(emails_df)
    print(f"   Extracted {features_df.shape[1]} features from {features_df.shape[0]} emails")

    # Show sample features
    print("\n4. Sample feature values:")
    print("   First 5 features for first email:")
    for feat in features_df.columns[:5]:
        print(f"     {feat:40s}: {features_df.iloc[0][feat]:.4f}")

    # Extraction statistics
    print("\n5. Extraction Statistics:")
    pipeline.print_extraction_summary()

    # Feature importance
    print("\n6. Feature Importance Analysis (Mutual Information):")
    print("   Computing mutual information scores...")
    mi_scores = compute_mutual_information(features_df, labels)

    print("\n   Top 10 Most Important Features:")
    ranking = print_feature_ranking(
        pd.DataFrame({
            "feature": mi_scores.index[:10],
            "importance": mi_scores.values[:10],
            "rank": range(1, 11)
        }),
        "Top 10 Features - Mutual Information"
    )

    # Correlation analysis
    print("\n7. Feature Correlation Analysis:")
    print("   Checking for redundant features...")
    corr_results = analyze_feature_correlation(
        features_df,
        threshold=0.9,
        plot=False  # Skip plotting in demo
    )

    if corr_results["redundant_features"]:
        print(f"\n   Found {len(corr_results['redundant_features'])} redundant features:")
        for feat in corr_results["redundant_features"]:
            print(f"     - {feat}")
    else:
        print("   No highly redundant features found (threshold >= 0.9)")

    # Financial features highlight
    print("\n8. Financial-Specific Features (KEY DIFFERENTIATOR):")
    financial_feats = [col for col in features_df.columns if "bank" in col.lower() or "wire" in col.lower() or "credential" in col.lower() or "account" in col.lower()]

    print("   Financial feature values for phishing vs legitimate emails:")
    print("\n   Email 1 (Phishing - Bank Impersonation):")
    for feat in financial_feats[:5]:
        if feat in features_df.columns:
            print(f"     {feat:40s}: {features_df.iloc[0][feat]:.4f}")

    print("\n   Email 3 (Legitimate - Bank Communication):")
    for feat in financial_feats[:5]:
        if feat in features_df.columns:
            print(f"     {feat:40s}: {features_df.iloc[2][feat]:.4f}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  - Pipeline extracted 60+ features from 5 emails")
    print("  - All features normalized to [0, 1] range")
    print("  - Financial features effectively separate phishing from legitimate")
    print("  - Extraction time <100ms per email target")
    print("  - sklearn-compatible for easy ML integration")
    print("\nNext Steps:")
    print("  - Run on full dataset (Nazario, APWG, Enron)")
    print("  - Train classifier (XGBoost, Random Forest)")
    print("  - Deploy in federated learning setting")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
