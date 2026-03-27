"""
Email preprocessing for transformer-based phishing detection.
Handles special token injection and text truncation strategies.
"""
import re
from typing import Dict, List, Tuple, Optional


def preprocess_email(
    email: Dict[str, str],
    use_special_tokens: bool = True,
    special_tokens: Optional[List[str]] = None
) -> str:
    """
    Preprocess email by injecting special tokens for structure.

    Args:
        email: Dictionary with email data (subject, body, sender, url)
        use_special_tokens: Whether to inject special tokens
        special_tokens: List of special tokens to use

    Returns:
        Preprocessed text string
    """
    if special_tokens is None:
        special_tokens = ["[SUBJECT]", "[BODY]", "[URL]", "[SENDER]"]

    subject = email.get('subject', '')
    body = email.get('body', email.get('text', ''))
    sender = email.get('sender', '')
    url = email.get('url', '')

    # Clean text
    subject = clean_text(subject)
    body = clean_text(body)
    sender = clean_text(sender)
    url = clean_text(url)

    if use_special_tokens:
        # Inject special tokens for structure
        parts = []
        if subject:
            parts.append(f"{special_tokens[0]} {subject}")
        if body:
            parts.append(f"{special_tokens[1]} {body}")
        if url:
            parts.append(f"{special_tokens[2]} {url}")
        if sender:
            parts.append(f"{special_tokens[3]} {sender}")

        return " ".join(parts)
    else:
        # Simple concatenation
        return f"{subject} {body}".strip()


def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize newlines
    text = re.sub(r'\n+', ' ', text)

    return text


def truncate_text(
    tokens: List[str],
    max_length: int,
    strategy: str = "head_tail"
) -> List[str]:
    """
    Truncate text using different strategies.

    Args:
        tokens: List of tokens
        max_length: Maximum number of tokens to keep
        strategy: Truncation strategy ('head_only', 'tail_only', 'head_tail')

    Returns:
        Truncated list of tokens
    """
    if len(tokens) <= max_length:
        return tokens

    if strategy == "head_only":
        # Keep only the head (beginning)
        return tokens[:max_length]

    elif strategy == "tail_only":
        # Keep only the tail (end)
        return tokens[-max_length:]

    elif strategy == "head_tail":
        # Keep both head and tail, drop middle
        # Reserve space for special tokens (3 for truncation marker)
        keep_each = (max_length - 3) // 2
        head = tokens[:keep_each]
        tail = tokens[-keep_each:]
        # Insert truncation marker
        return head + ["[TRUNCATED]"] + tail

    else:
        raise ValueError(f"Unknown truncation strategy: {strategy}")


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from email text.

    Args:
        text: Email text

    Returns:
        List of URLs found in the text
    """
    # Simple URL regex pattern
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls


def extract_email_addresses(text: str) -> List[str]:
    """
    Extract email addresses from text.

    Args:
        text: Email text

    Returns:
        List of email addresses found
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails


def analyze_email_structure(text: str) -> Dict[str, int]:
    """
    Analyze email structure for feature extraction.

    Args:
        text: Email text

    Returns:
        Dictionary with structural features
    """
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'url_count': len(extract_urls_from_text(text)),
        'email_count': len(extract_email_addresses(text)),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
    }
