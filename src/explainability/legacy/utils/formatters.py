"""
Formatting utilities for non-technical user output.

Converts technical explanations into human-readable, actionable formats.
"""

from typing import List, Dict, Any
from src.explainability.legacy.utils.data_structures import Explanation


def format_confidence_score(confidence: float) -> str:
    """
    Format confidence score in non-technical terms.

    Args:
        confidence: Confidence value (0.0 to 1.0)

    Returns:
        Human-readable confidence description
    """
    if confidence >= 0.95:
        return "very high"
    elif confidence >= 0.80:
        return "high"
    elif confidence >= 0.60:
        return "moderate"
    elif confidence >= 0.40:
        return "low"
    else:
        return "very low"


def format_risk_level(is_suspicious: bool, confidence: float) -> str:
    """
    Format risk level for user.

    Args:
        is_suspicious: Whether component is flagged as suspicious
        confidence: Confidence score

    Returns:
        Human-readable risk level
    """
    if not is_suspicious:
        return "Safe"

    conf = format_confidence_score(confidence)
    return f"Suspicious ({conf} confidence)"


def format_explanation_for_user(explanation: Explanation) -> str:
    """
    Format explanation for non-technical end users.

    Uses simple language, actionable items, and no jargon.

    Args:
        explanation: Explanation object

    Returns:
        Formatted explanation string
    """
    lines = []

    # Header
    confidence_str = format_confidence_score(explanation.model_prediction.confidence)
    lines.append(f"## Email Analysis Result")
    lines.append(f"**Status**: {explanation.model_prediction.predicted_label.value.upper()}")
    lines.append(f"**Confidence**: {confidence_str} ({explanation.model_prediction.confidence:.1%})")
    lines.append("")

    # Summary
    lines.append("### Summary")
    lines.append(explanation.get_summary())
    lines.append("")

    # Sender explanation
    if explanation.sender_explanation:
        lines.append(format_sender_for_user(explanation.sender_explanation))

    # Subject explanation
    if explanation.subject_explanation:
        lines.append(format_subject_for_user(explanation.subject_explanation))

    # Body explanation
    if explanation.body_explanation:
        lines.append(format_body_for_user(explanation.body_explanation))

    # URL explanation
    if explanation.url_explanation:
        lines.append(format_url_for_user(explanation.url_explanation))

    # Attachment explanation
    if explanation.attachment_explanation:
        lines.append(format_attachment_for_user(explanation.attachment_explanation))

    # Counterfactuals
    if explanation.counterfactuals:
        lines.append("### What Could Make This Email Safe")
        for i, cf in enumerate(explanation.counterfactuals[:3], 1):
            lines.append(f"{i}. {cf.get_summary()}")
        lines.append("")

    # Comparative explanation
    if explanation.comparative:
        lines.append("### Known Phishing Patterns")
        lines.append(explanation.comparative.get_summary())
        lines.append("")

    # Actionable advice
    lines.append("### What Should You Do?")
    lines.append(get_actionable_advice(explanation))
    lines.append("")

    return "\n".join(lines)


def format_sender_for_user(sender_exp) -> str:
    """Format sender explanation for users."""
    lines = ["### Sender Analysis", ""]

    if sender_exp.is_suspicious:
        lines.append(f"⚠️ **Risk Level**: {format_risk_level(True, sender_exp.confidence)}")
        lines.append("")
        lines.append("**Issues Detected:**")

        for reason in sender_exp.reasons:
            lines.append(f"- {reason}")

        if sender_exp.display_name_mismatch:
            lines.append("- Display name doesn't match email address")

        if sender_exp.lookalike_domain:
            lines.append("- Email domain looks similar to a well-known company")

    else:
        lines.append(f"✅ **Risk Level**: Safe")
        lines.append("Sender appears legitimate.")

    lines.append("")
    return "\n".join(lines)


def format_subject_for_user(subject_exp) -> str:
    """Format subject explanation for users."""
    lines = ["### Subject Line Analysis", ""]

    if subject_exp.is_suspicious:
        lines.append(f"⚠️ **Risk Level**: {format_risk_level(True, subject_exp.confidence)}")
        lines.append("")
        lines.append("**Issues Detected:**")

        for reason in subject_exp.reasons:
            lines.append(f"- {reason}")

        if subject_exp.urgency_keywords:
            lines.append(f"- Contains urgency words: {', '.join(subject_exp.urgency_keywords)}")

        if subject_exp.unusual_formatting:
            lines.append(f"- Unusual formatting: {', '.join(subject_exp.unusual_formatting)}")

    else:
        lines.append(f"✅ **Risk Level**: Safe")
        lines.append("Subject line appears normal.")

    lines.append("")
    return "\n".join(lines)


def format_body_for_user(body_exp) -> str:
    """Format body explanation for users."""
    lines = ["### Email Content Analysis", ""]

    if body_exp.is_suspicious:
        lines.append(f"⚠️ **Risk Level**: {format_risk_level(True, body_exp.confidence)}")
        lines.append("")
        lines.append("**Issues Detected:**")

        for reason in body_exp.reasons:
            lines.append(f"- {reason}")

        if body_exp.social_engineering_tactics:
            lines.append("**Tactics Used:**")
            for tactic in body_exp.social_engineering_tactics:
                lines.append(f"  - {tactic}")

        if body_exp.grammar_issues:
            lines.append("**Quality Issues:**")
            for issue in body_exp.grammar_issues:
                lines.append(f"  - {issue}")

    else:
        lines.append(f"✅ **Risk Level**: Safe")
        lines.append("Email content appears normal.")

    lines.append("")
    return "\n".join(lines)


def format_url_for_user(url_exp) -> str:
    """Format URL explanation for users."""
    lines = ["### Link Analysis", ""]

    if url_exp.is_suspicious:
        lines.append(f"⚠️ **Risk Level**: {format_risk_level(True, url_exp.confidence)}")
        lines.append("")
        lines.append("**Suspicious Links Found:**")

        for url_info in url_exp.suspicious_urls:
            lines.append(f"- **{url_info.get('url', 'Unknown')}**")
            lines.append(f"  Reason: {url_info.get('reason', 'Unknown')}")

    else:
        lines.append(f"✅ **Risk Level**: Safe")
        if url_exp.safe_urls:
            lines.append(f"All links ({len(url_exp.safe_urls)}) appear safe.")
        else:
            lines.append("No links found in this email.")

    lines.append("")
    return "\n".join(lines)


def format_attachment_for_user(attachment_exp) -> str:
    """Format attachment explanation for users."""
    lines = ["### Attachment Analysis", ""]

    if attachment_exp.is_suspicious:
        lines.append(f"⚠️ **Risk Level**: {format_risk_level(True, attachment_exp.confidence)}")
        lines.append("")
        lines.append("**Dangerous Attachments:**")

        for att in attachment_exp.dangerous_attachments:
            lines.append(f"- **{att.get('filename', 'Unknown')}**")
            lines.append(f"  Reason: {att.get('reason', 'Unknown')}")

        lines.append("")
        lines.append("**⚠️ DO NOT OPEN these attachments!**")

    else:
        lines.append(f"✅ **Risk Level**: Safe")
        if not attachment_exp.dangerous_attachments:
            lines.append("No dangerous attachments detected.")
        else:
            lines.append("Attachments appear safe.")

    lines.append("")
    return "\n".join(lines)


def get_actionable_advice(explanation: Explanation) -> str:
    """
    Generate actionable advice based on explanation.

    Args:
        explanation: Explanation object

    Returns:
        Actionable advice string
    """
    advice = []

    if explanation.model_prediction.predicted_label.value == "phishing":
        advice.append("❌ **Do not click** any links in this email")
        advice.append("❌ **Do not download** any attachments")
        advice.append("❌ **Do not reply** to this email")
        advice.append("✅ **Report** this email to your IT security team")
        advice.append("✅ **Delete** this email after reporting")

        if explanation.url_explanation and explanation.url_explanation.suspicious_urls:
            advice.append("✅ **Verify** suspicious requests by contacting the organization directly through official channels")

    elif explanation.model_prediction.predicted_label.value == "suspicious":
        advice.append("⚠️ **Exercise caution** with this email")
        advice.append("✅ **Verify** the sender through another channel")
        advice.append("✅ **Check** links before clicking (hover to see actual URL)")
        advice.append("✅ **Contact** IT security if unsure")

    else:  # Safe
        advice.append("✅ Email appears safe to open and interact with")
        advice.append("✅ Always remain vigilant for suspicious emails")

    return "\n".join(advice)


def format_explanation_for_analyst(explanation: Explanation) -> str:
    """
    Format explanation for security analysts.

    More technical, includes metrics and detailed scores.

    Args:
        explanation: Explanation object

    Returns:
        Formatted explanation string
    """
    lines = []

    # Header
    lines.append("## Security Analysis Report")
    lines.append(f"**Email ID**: {explanation.email.email_id or 'N/A'}")
    lines.append(f"**Prediction**: {explanation.model_prediction.predicted_label.value.upper()}")
    lines.append(f"**Confidence**: {explanation.model_prediction.confidence:.4f}")
    lines.append(f"**Model**: {explanation.model_prediction.model_name or 'Unknown'}")
    lines.append(f"**Generation Time**: {explanation.generation_time_ms:.1f}ms")
    lines.append("")

    # Component scores
    lines.append("### Component Analysis")
    lines.append("")

    components = [
        ("Sender", explanation.sender_explanation),
        ("Subject", explanation.subject_explanation),
        ("Body", explanation.body_explanation),
        ("URLs", explanation.url_explanation),
        ("Attachments", explanation.attachment_explanation),
    ]

    for name, comp in components:
        if comp:
            risk = "HIGH" if comp.is_suspicious else "LOW"
            lines.append(f"**{name}**: {risk} RISK (conf: {comp.confidence:.3f})")

            if comp.is_suspicious and comp.reasons:
                lines.append(f"  Reasons: {', '.join(comp.reasons)}")

    lines.append("")

    # Feature importance
    if explanation.feature_importance:
        lines.append("### Top Contributing Features")
        lines.append("")

        for i, (feat, score) in enumerate(explanation.feature_importance.top_features[:10], 1):
            lines.append(f"{i}. {feat}: {score:.4f}")

        lines.append("")

    # Attention visualization
    if explanation.attention_visualization:
        lines.append("### Model Attention (Top Tokens)")
        lines.append("")

        for token, attention in explanation.attention_visualization.get_top_attended_tokens(10):
            lines.append(f"- {token}: {attention:.4f}")

        lines.append("")

    # Counterfactuals
    if explanation.counterfactuals:
        lines.append("### Counterfactual Analysis")
        lines.append("")

        for i, cf in enumerate(explanation.counterfactuals[:3], 1):
            lines.append(f"{i}. {cf.get_summary()}")

        lines.append("")

    # Threat intelligence
    if explanation.comparative and explanation.comparative.similar_campaigns:
        lines.append("### Threat Intelligence")
        lines.append("")

        lines.append(explanation.comparative.get_summary())
        lines.append("")

    return "\n".join(lines)


def format_highlighted_text(text: str, suspicious_spans: List[tuple]) -> str:
    """
    Format text with highlighted suspicious portions.

    Args:
        text: Original text
        suspicious_spans: List of (start, end, reason) tuples

    Returns:
        Text with HTML-like highlighting
    """
    if not suspicious_spans:
        return text

    # Sort by start position
    suspicious_spans = sorted(suspicious_spans, key=lambda x: x[0])

    result = []
    last_end = 0

    for start, end, reason in suspicious_spans:
        # Add text before highlight
        result.append(text[last_end:start])

        # Add highlighted text
        result.append(f"<mark class='suspicious' title='{reason}'>{text[start:end]}</mark>")

        last_end = end

    # Add remaining text
    result.append(text[last_end:])

    return "".join(result)
