"""
Streamlit user interface for end users.

Provides non-technical, actionable explanations for phishing detection.
"""

import streamlit as st
from typing import Optional
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
from src.explainability.legacy.utils.formatters import (
    format_explanation_for_user,
    format_confidence_score,
    format_risk_level
)


def main():
    """Main Streamlit app for users."""
    st.set_page_config(
        page_title="Phishing Email Analyzer",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Phishing Email Analyzer")
    st.markdown("""
    Understand why an email was flagged as suspicious. This tool provides
    clear, actionable explanations following how humans actually check emails.
    """)

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This tool analyzes emails for phishing attempts and explains
    what makes them suspicious in plain language.

    **Privacy Note**: Your email is analyzed locally. No data is sent to external servers.
    """)

    # Input section
    st.header("Enter Email Details")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Sender input
        st.subheader("Sender Information")
        sender_display = st.text_input("Display Name", "")
        sender_email = st.text_input("Email Address", "")

        # Subject
        st.subheader("Subject Line")
        subject = st.text_input("Subject", "")

        # Body
        st.subheader("Email Body")
        body = st.text_area("Content", height=200)

    with col2:
        # URLs
        st.subheader("Links (Optional)")
        urls_input = st.text_area("Enter URLs (one per line)", height=100)

        # Attachments
        st.subheader("Attachments (Optional)")
        attachment_names = st.text_input("Attachment filenames (comma-separated)", "")

    # Analyze button
    analyze_button = st.button("🔍 Analyze Email", type="primary", use_container_width=True)

    if analyze_button:
        if not sender_email or not subject or not body:
            st.error("Please fill in at least the sender email, subject, and body.")
        else:
            # Create email data
            email = create_email_from_input(
                sender_display, sender_email, subject, body,
                urls_input, attachment_names
            )

            # Get model prediction (simulated)
            prediction = get_simulated_prediction(email)

            # Generate explanation
            with st.spinner("Analyzing email..."):
                generator = HumanAlignedGenerator()
                explanation = generator.generate_with_timing(email, prediction)

            # Display results
            display_explanation(explanation)

            # Generation time
            if explanation.generation_time_ms < 500:
                st.success(f"✅ Analysis completed in {explanation.generation_time_ms:.0f}ms")
            else:
                st.warning(f"⚠️ Analysis took {explanation.generation_time_ms:.0f}ms (target: <500ms)")

    # Footer
    st.divider()
    st.markdown("""
    <small>
    <b>Research Context:</b> This tool is part of a research project on human-aligned
    explainability for phishing detection. Reference: "Eyes on the Phish(er)" (CHI 2025).
    </small>
    """, unsafe_allow_html=True)


def create_email_from_input(
    sender_display: str,
    sender_email: str,
    subject: str,
    body: str,
    urls_input: str,
    attachment_names: str
) -> EmailData:
    """Create EmailData object from user input."""
    # Parse sender
    sender = EmailAddress(
        display_name=sender_display if sender_display else None,
        email=sender_email,
        is_suspicious=False  # Will be analyzed
    )

    # Parse URLs
    urls = []
    if urls_input:
        for url_str in urls_input.strip().split('\n'):
            url_str = url_str.strip()
            if url_str:
                urls.append(URL(
                    original=url_str,
                    domain=url_str.split('/')[2] if '://' in url_str else url_str.split('/')[0],
                    has_https='https://' in url_str,
                    is_suspicious=False  # Will be analyzed
                ))

    # Parse attachments
    attachments = []
    if attachment_names:
        for att_name in attachment_names.split(','):
            att_name = att_name.strip()
            if att_name:
                # Determine file type
                file_ext = Path(att_name).suffix.lower()
                dangerous_extensions = {'.exe', '.scr', '.bat', '.doc', '.docm', '.zip'}

                attachments.append(Attachment(
                    filename=att_name,
                    file_type=file_ext,
                    size_bytes=0,  # Unknown
                    has_macros=file_ext in {'.doc', '.docm', '.xls', '.xlsm'},
                    is_dangerous=file_ext in dangerous_extensions
                ))

    return EmailData(
        sender=sender,
        recipients=[],
        subject=subject,
        body=body,
        urls=urls,
        attachments=attachments,
        category=EmailCategory.SAFE
    )


def get_simulated_prediction(email: EmailData) -> ModelOutput:
    """Get simulated model prediction for demo."""
    # Simple heuristic for demo
    suspicious_indicators = sum([
        email.sender.is_suspicious,
        len(email.subject) > 0 and any(
            word in email.subject.lower()
            for word in ['urgent', 'immediate', 'verify', 'suspended']
        ),
        len(email.body) > 0 and any(
            word in email.body.lower()
            for word in ['password', 'verify', 'click', 'account']
        ),
        any(url.is_suspicious for url in email.urls),
        any(att.is_dangerous for att in email.attachments)
    ])

    if suspicious_indicators >= 2:
        return ModelOutput(
            predicted_label=EmailCategory.PHISHING,
            confidence=0.82 + (suspicious_indicators * 0.03),
            model_name="DemoClassifier"
        )
    elif suspicious_indicators == 1:
        return ModelOutput(
            predicted_label=EmailCategory.SUSPICIOUS,
            confidence=0.65,
            model_name="DemoClassifier"
        )
    else:
        return ModelOutput(
            predicted_label=EmailCategory.SAFE,
            confidence=0.88,
            model_name="DemoClassifier"
        )


def display_explanation(explanation):
    """Display explanation in user-friendly format."""
    st.header("Analysis Results")

    # Status banner
    pred = explanation.model_prediction
    if pred.predicted_label == EmailCategory.PHISHING:
        st.error(f"### ⚠️ This email appears to be PHISHING")
    elif pred.predicted_label == EmailCategory.SUSPICIOUS:
        st.warning(f"### ⚡ This email is SUSPICIOUS")
    else:
        st.success(f"### ✅ This email appears to be SAFE")

    # Confidence
    confidence_str = format_confidence_score(pred.confidence)
    st.info(f"**Confidence Level**: {confidence_str} ({pred.confidence:.1%})")

    # Summary
    st.markdown("### What We Found")
    st.markdown(explanation.get_summary())

    # Detailed analysis (expandable sections)
    st.markdown("---")
    st.markdown("### Detailed Analysis")

    # Sender
    if explanation.sender_explanation:
        with st.expander("📧 Sender Analysis", expanded=explanation.sender_explanation.is_suspicious):
            display_sender_explanation(explanation.sender_explanation)

    # Subject
    if explanation.subject_explanation:
        with st.expander("📝 Subject Line Analysis", expanded=explanation.subject_explanation.is_suspicious):
            display_subject_explanation(explanation.subject_explanation)

    # Body
    if explanation.body_explanation:
        with st.expander("📄 Email Content Analysis", expanded=explanation.body_explanation.is_suspicious):
            display_body_explanation(explanation.body_explanation)

    # URLs
    if explanation.url_explanation:
        with st.expander("🔗 Link Analysis", expanded=explanation.url_explanation.is_suspicious):
            display_url_explanation(explanation.url_explanation)

    # Attachments
    if explanation.attachment_explanation:
        with st.expander("📎 Attachment Analysis", expanded=explanation.attachment_explanation.is_suspicious):
            display_attachment_explanation(explanation.attachment_explanation)

    # Counterfactuals
    if explanation.counterfactuals:
        st.markdown("---")
        st.markdown("### What Would Make This Email Safe?")
        for i, cf in enumerate(explanation.counterfactuals[:3], 1):
            st.markdown(f"{i}. {cf.get_summary()}")

    # Actionable advice
    st.markdown("---")
    st.markdown("### What Should You Do?")
    from src.explainability.legacy.utils.formatters import get_actionable_advice
    st.markdown(get_actionable_advice(explanation))

    # Feedback
    st.markdown("---")
    st.markdown("### Was this explanation helpful?")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👍 Yes, helpful"):
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎 No, not helpful"):
            st.info("Thank you! We'll work to improve.")
    with col3:
        if st.button("❓ Still confused"):
            st.info("Please contact security@yourbank.com for assistance.")


def display_sender_explanation(sender_exp):
    """Display sender explanation."""
    if sender_exp.is_suspicious:
        st.error(f"Suspicious Sender ({format_risk_level(True, sender_exp.confidence)})")
        for reason in sender_exp.reasons:
            st.markdown(f"- ⚠️ {reason}")
    else:
        st.success("✅ Sender appears legitimate")


def display_subject_explanation(subject_exp):
    """Display subject explanation."""
    if subject_exp.is_suspicious:
        st.warning(f"Suspicious Subject ({format_risk_level(True, subject_exp.confidence)})")
        for reason in subject_exp.reasons:
            st.markdown(f"- ⚠️ {reason}")
    else:
        st.success("✅ Subject line appears normal")


def display_body_explanation(body_exp):
    """Display body explanation."""
    if body_exp.is_suspicious:
        st.warning(f"Suspicious Content ({format_risk_level(True, body_exp.confidence)})")
        for reason in body_exp.reasons:
            st.markdown(f"- ⚠️ {reason}")
    else:
        st.success("✅ Email content appears normal")


def display_url_explanation(url_exp):
    """Display URL explanation."""
    if url_exp.is_suspicious:
        st.error(f"Dangerous Links Found ({format_risk_level(True, url_exp.confidence)})")
        for url_info in url_exp.suspicious_urls:
            st.markdown(f"- ⚠️ **{url_info['url']}**")
            st.caption(f"Reason: {url_info['reason']}")
    else:
        if url_exp.safe_urls:
            st.success(f"✅ All {len(url_exp.safe_urls)} links appear safe")
        else:
            st.info("No links found in this email")


def display_attachment_explanation(att_exp):
    """Display attachment explanation."""
    if att_exp.is_suspicious:
        st.error(f"Dangerous Attachments ({format_risk_level(True, att_exp.confidence)})")
        for att in att_exp.dangerous_attachments:
            st.markdown(f"- ⚠️ **{att['filename']}**")
            st.caption(f"Reason: {att['reason']}")
        st.error("⚠️ **DO NOT OPEN** these attachments!")
    else:
        st.success("✅ No dangerous attachments detected")


if __name__ == "__main__":
    main()
