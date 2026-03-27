"""
Streamlit interface for security analysts.

Provides batch processing, detailed metrics, and export capabilities
for bank security analysts triaging phishing emails.
"""

import streamlit as st
import pandas as pd
from typing import List
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from src.explainability.legacy.utils.data_structures import (
    EmailData,
    EmailAddress,
    URL,
    Attachment,
    ModelOutput,
    EmailCategory
)
from src.explainability.legacy.generators.human_aligned import HumanAlignedGenerator
from src.explainability.legacy.utils.formatters import format_explanation_for_analyst


def main():
    """Main Streamlit app for analysts."""
    st.set_page_config(
        page_title="Security Analyst Dashboard",
        page_icon="🛡️",
        layout="wide"
    )

    st.title("🛡️ Security Analyst Dashboard")
    st.markdown("""
    Advanced phishing detection and explanation system for security professionals.
    Batch process emails, analyze patterns, and export reports.
    """)

    # Sidebar
    st.sidebar.header("Settings")

    # Mode selection
    mode = st.sidebar.radio(
        "Mode",
        ["Single Email Analysis", "Batch Processing", "Pattern Analysis"]
    )

    # Export format
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["JSON", "CSV", "PDF Report"]
    )

    if mode == "Single Email Analysis":
        single_email_mode()
    elif mode == "Batch Processing":
        batch_processing_mode()
    else:
        pattern_analysis_mode()


def single_email_mode():
    """Single email detailed analysis mode."""
    st.header("Single Email Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Quick paste area
        email_text = st.text_area(
            "Paste Email Content",
            height=300,
            placeholder="Paste full email headers and body here..."
        )

        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    with col2:
        st.subheader("Quick Statistics")
        if email_text:
            st.metric("Character Count", len(email_text))
            st.metric("Line Count", len(email_text.split('\n')))

            # Detect URLs
            from src.explainability.legacy.utils.text_processing import extract_urls
            urls = extract_urls(email_text)
            st.metric("URLs Found", len(urls))

            # Detect email addresses
            from src.explainability.legacy.utils.text_processing import extract_email_addresses
            emails = extract_email_addresses(email_text)
            st.metric("Email Addresses", len(emails))

    if analyze_btn and email_text:
        with st.spinner("Analyzing email..."):
            # Parse email (simplified for demo)
            email = parse_email_text(email_text)
            prediction = get_analyst_prediction(email)

            # Generate explanation
            generator = HumanAlignedGenerator()
            explanation = generator.generate_with_timing(email, prediction)

        # Display detailed report
        display_analyst_report(explanation)

        # Export button
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Export JSON"):
                export_explanation(explanation, "json")
        with col2:
            if st.button("📥 Export CSV"):
                export_explanation(explanation, "csv")
        with col3:
            if st.button("📥 Export Report"):
                export_explanation(explanation, "report")


def batch_processing_mode():
    """Batch processing mode for triaging multiple emails."""
    st.header("Batch Email Processing")

    # Upload area
    uploaded_file = st.file_uploader(
        "Upload Email File",
        type=['csv', 'json', 'txt'],
        help="Upload a file containing multiple emails to analyze"
    )

    if uploaded_file:
        st.info(f"Uploaded: {uploaded_file.name}")

        # Parse uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                emails_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(emails_df)} emails")

                # Display preview
                st.subheader("Email Preview")
                st.dataframe(emails_df.head())

                # Analyze button
                if st.button("Analyze Batch", type="primary"):
                    analyze_batch(emails_df)

            elif uploaded_file.name.endswith('.json'):
                import json
                emails_data = json.load(uploaded_file)
                st.success(f"Loaded {len(emails_data)} emails")

                if st.button("Analyze Batch", type="primary"):
                    analyze_batch_json(emails_data)

        except Exception as e:
            st.error(f"Error parsing file: {e}")

    # Manual batch input
    st.markdown("---")
    st.subheader("Or Paste Multiple Emails")

    batch_text = st.text_area(
        "Paste emails (one per line, separated by ---)",
        height=200
    )

    if st.button("Analyze Pasted Batch"):
        st.info("Batch analysis not implemented for demo")


def pattern_analysis_mode():
    """Pattern analysis across multiple explanations."""
    st.header("Pattern Analysis")

    st.markdown("""
    Analyze patterns across multiple phishing emails to identify
    campaigns, trends, and common attack vectors.
    """)

    # Sample data for demo
    st.subheader("Sample Phishing Campaigns Detected")

    campaigns = {
        "Netflix Account Verification": 23,
        "Bank Security Alert": 15,
        "IRS Tax Refund": 8,
        "Microsoft Office 365": 12,
        "Amazon Order Scam": 18
    }

    # Display as metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Campaign", "Netflix Account Verification", "23 emails")
    with col2:
        st.metric("Total Analyzed", "76", "Last 7 days")
    with col3:
        st.metric("Detection Rate", "94.2%", "+2.1%")

    # Campaign breakdown
    st.markdown("### Campaign Breakdown")
    campaign_df = pd.DataFrame(list(campaigns.items()), columns=["Campaign", "Count"])
    st.bar_chart(campaign_df.set_index("Campaign"))

    # Feature importance aggregation
    st.markdown("### Common Suspicious Features")
    features = {
        "Suspicious Sender": 45,
        "Urgency Language": 38,
        "Suspicious URLs": 52,
        "Dangerous Attachments": 23,
        "Grammar Issues": 31,
        "Pressure Tactics": 27
    }

    feature_df = pd.DataFrame(list(features.items()), columns=["Feature", "Frequency"])
    st.dataframe(feature_df.sort_values("Frequency", ascending=False))


@st.cache_data
def parse_email_text(text: str) -> EmailData:
    """Parse email text into EmailData object."""
    lines = text.split('\n')

    # Extract headers
    sender_email = "unknown@example.com"
    sender_name = None
    subject = "No Subject"
    body_lines = []
    in_body = False

    for line in lines:
        if line.lower().startswith('from:'):
            from src.explainability.legacy.utils.text_processing import extract_email_parts
            sender_name, sender_email = extract_email_parts(line[5:].strip())
        elif line.lower().startswith('subject:'):
            subject = line[8:].strip()
        elif in_body or line.strip() == '':
            in_body = True
            body_lines.append(line)

    body = '\n'.join(body_lines).strip()

    # Extract URLs
    from src.explainability.legacy.utils.text_processing import extract_urls
    urls_str = extract_urls(body)
    urls = [
        URL(
            original=url,
            domain=url.split('/')[2] if '://' in url else url.split('/')[0],
            has_https='https://' in url
        )
        for url in urls_str
    ]

    return EmailData(
        sender=EmailAddress(
            display_name=sender_name,
            email=sender_email
        ),
        recipients=[],
        subject=subject,
        body=body,
        urls=urls,
        attachments=[]
    )


def get_analyst_prediction(email: EmailData) -> ModelOutput:
    """Get prediction for analyst mode (more detailed)."""
    # Analyze components
    sender_suspicious = (
        len(email.sender.email) < 10 or
        '@example.com' in email.sender.email or
        any(bad in email.sender.email.lower() for bad in ['@temp.', '@trash.', 'Suspicious'])
    )

    subject_suspicious = any(
        word in email.subject.lower()
        for word in ['urgent', 'verify', 'account', 'suspended', 'immediate']
    )

    body_suspicious = any(
        word in email.body.lower()
        for word in ['password', 'verify', 'click', 'login', 'account']
    )

    url_suspicious = any(
        not url.has_https or 'example.com' in url.domain
        for url in email.urls
    )

    score = sum([sender_suspicious, subject_suspicious, body_suspicious, url_suspicious])

    if score >= 3:
        return ModelOutput(
            predicted_label=EmailCategory.PHISHING,
            confidence=0.89,
            model_name="PhishingDetector-Pro"
        )
    elif score >= 1:
        return ModelOutput(
            predicted_label=EmailCategory.SUSPICIOUS,
            confidence=0.72,
            model_name="PhishingDetector-Pro"
        )
    else:
        return ModelOutput(
            predicted_label=EmailCategory.SAFE,
            confidence=0.91,
            model_name="PhishingDetector-Pro"
        )


def display_analyst_report(explanation):
    """Display detailed analyst report."""
    st.markdown("---")
    st.markdown("## Security Analysis Report")

    # Key metrics at top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Prediction",
            explanation.model_prediction.predicted_label.value.upper(),
            delta_color="normal"
        )
    with col2:
        st.metric(
            "Confidence",
            f"{explanation.model_prediction.confidence:.2%}"
        )
    with col3:
        st.metric(
            "Generation Time",
            f"{explanation.generation_time_ms:.0f}ms"
        )
    with col4:
        st.metric(
            "Model",
            explanation.model_prediction.model_name or "Unknown"
        )

    # Component scores
    st.markdown("### Component Risk Scores")

    components = []
    if explanation.sender_explanation:
        components.append({
            "Component": "Sender",
            "Risk": "HIGH" if explanation.sender_explanation.is_suspicious else "LOW",
            "Confidence": f"{explanation.sender_explanation.confidence:.3f}",
            "Reasons": "; ".join(explanation.sender_explanation.reasons)
        })

    if explanation.subject_explanation:
        components.append({
            "Component": "Subject",
            "Risk": "HIGH" if explanation.subject_explanation.is_suspicious else "LOW",
            "Confidence": f"{explanation.subject_explanation.confidence:.3f}",
            "Reasons": "; ".join(explanation.subject_explanation.reasons)
        })

    if explanation.body_explanation:
        components.append({
            "Component": "Body",
            "Risk": "HIGH" if explanation.body_explanation.is_suspicious else "LOW",
            "Confidence": f"{explanation.body_explanation.confidence:.3f}",
            "Reasons": "; ".join(explanation.body_explanation.reasons)
        })

    if explanation.url_explanation:
        components.append({
            "Component": "URLs",
            "Risk": "HIGH" if explanation.url_explanation.is_suspicious else "LOW",
            "Confidence": f"{explanation.url_explanation.confidence:.3f}",
            "Reasons": f"{len(explanation.url_explanation.suspicious_urls)} suspicious URLs"
        })

    if explanation.attachment_explanation:
        components.append({
            "Component": "Attachments",
            "Risk": "HIGH" if explanation.attachment_explanation.is_suspicious else "LOW",
            "Confidence": f"{explanation.attachment_explanation.confidence:.3f}",
            "Reasons": f"{len(explanation.attachment_explanation.dangerous_attachments)} dangerous attachments"
        })

    st.dataframe(pd.DataFrame(components), use_container_width=True)

    # Feature importance
    if explanation.feature_importance:
        st.markdown("### Top Contributing Features")
        feature_df = pd.DataFrame(
            explanation.feature_importance.top_features[:10],
            columns=["Feature", "Importance"]
        )
        st.dataframe(feature_df, use_container_width=True)

    # Detailed explanation
    with st.expander("📄 Full Detailed Report", expanded=False):
        st.markdown(format_explanation_for_analyst(explanation))


def analyze_batch(emails_df: pd.DataFrame):
    """Analyze batch of emails."""
    st.info("Batch processing in progress...")

    progress_bar = st.progress(0)
    results = []

    for idx, row in emails_df.iterrows():
        # Create email from row (simplified)
        email = EmailData(
            sender=EmailAddress(
                display_name=None,
                email=row.get('sender', 'unknown@example.com')
            ),
            recipients=[],
            subject=row.get('subject', ''),
            body=row.get('body', ''),
            urls=[],
            attachments=[]
        )

        # Get prediction
        prediction = get_analyst_prediction(email)

        # Generate explanation
        generator = HumanAlignedGenerator()
        explanation = generator.generate_with_timing(email, prediction)

        results.append({
            'Index': idx,
            'Prediction': explanation.model_prediction.predicted_label.value,
            'Confidence': explanation.model_prediction.confidence,
            'Generation Time (ms)': explanation.generation_time_ms
        })

        progress_bar.progress((idx + 1) / len(emails_df))

    # Display results
    st.success(f"✅ Analyzed {len(results)} emails")

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        phishing_count = (results_df['Prediction'] == 'phishing').sum()
        st.metric("Phishing Detected", phishing_count)
    with col2:
        suspicious_count = (results_df['Prediction'] == 'suspicious').sum()
        st.metric("Suspicious", suspicious_count)
    with col3:
        avg_time = results_df['Generation Time (ms)'].mean()
        st.metric("Avg Generation Time", f"{avg_time:.0f}ms")


def analyze_batch_json(emails_data: list):
    """Analyze batch from JSON."""
    st.info(f"Processing {len(emails_data)} emails...")
    st.warning("JSON batch processing not fully implemented in demo")


def export_explanation(explanation, format_type: str):
    """Export explanation in various formats."""
    if format_type == "json":
        import json
        # Create export dict
        export_data = {
            "prediction": explanation.model_prediction.predicted_label.value,
            "confidence": explanation.model_prediction.confidence,
            "generation_time_ms": explanation.generation_time_ms,
            "summary": explanation.get_summary()
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name="explanation.json",
            mime="application/json"
        )

    elif format_type == "csv":
        # Create CSV export
        import io
        output = io.StringIO()
        output.write("Component,Risk,Confidence\n")

        if explanation.sender_explanation:
            output.write(f"Sender,{explanation.sender_explanation.is_suspicious},{explanation.sender_explanation.confidence}\n")
        if explanation.subject_explanation:
            output.write(f"Subject,{explanation.subject_explanation.is_suspicious},{explanation.subject_explanation.confidence}\n")

        st.download_button(
            "Download CSV",
            data=output.getvalue(),
            file_name="explanation.csv",
            mime="text/csv"
        )

    else:
        st.info("PDF export not implemented in demo")


if __name__ == "__main__":
    main()
