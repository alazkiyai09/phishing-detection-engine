"""
Text processing utilities for email analysis.

Provides tokenization, URL extraction, and text normalization functions.
"""

import re
from typing import List, Tuple, Optional, Set
from urllib.parse import urlparse
import email.utils

from src.explainability.legacy.utils.data_structures import EmailAddress, URL


def tokenize_email(email_data: 'EmailData') -> dict:
    """
    Tokenize email into components for model input.

    Args:
        email_data: Email to tokenize

    Returns:
        Dictionary with tokenized components
    """
    tokens = {
        'sender': email_data.sender.email,
        'subject': email_data.subject,
        'body': email_data.body,
        'urls': [url.original for url in email_data.urls],
        'attachments': [att.filename for att in email_data.attachments]
    }

    # Tokenize body text (simple whitespace tokenization)
    tokens['body_tokens'] = email_data.body.split()

    return tokens


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text using regex.

    Args:
        text: Text to search for URLs

    Returns:
        List of URLs found
    """
    # URL pattern from RFC 3986
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    urls = re.findall(url_pattern, text)

    # Also catch www. without http
    www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    www_urls = re.findall(www_pattern, text)

    # Prepend http:// to www urls
    www_urls = [f"http://{url}" for url in www_urls]

    return list(set(urls + www_urls))


def extract_email_addresses(text: str) -> List[str]:
    """
    Extract email addresses from text.

    Args:
        text: Text to search

    Returns:
        List of email addresses found
    """
    # RFC 5322 compliant email regex (simplified)
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return list(set(re.findall(email_pattern, text)))


def normalize_text(text: str) -> str:
    """
    Normalize text for processing.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Normalize unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text


def extract_email_parts(email_address: str) -> Tuple[Optional[str], str]:
    """
    Extract display name and email address from email string.

    Args:
        email_address: Email string like "John Doe <john@example.com>"

    Returns:
        Tuple of (display_name, email)
    """
    display_name, addr = email.utils.parseaddr(email_address)
    return (display_name if display_name else None, addr)


def parse_url(url_string: str) -> dict:
    """
    Parse URL into components.

    Args:
        url_string: URL string to parse

    Returns:
        Dictionary with URL components
    """
    try:
        parsed = urlparse(url_string)
        return {
            'scheme': parsed.scheme,
            'domain': parsed.netloc,
            'path': parsed.path,
            'query': parsed.query,
            'fragment': parsed.fragment,
            'has_https': parsed.scheme == 'https'
        }
    except Exception:
        return {
            'scheme': None,
            'domain': url_string,
            'path': None,
            'query': None,
            'fragment': None,
            'has_https': False
        }


def extract_domain_from_email(email_address: str) -> str:
    """
    Extract domain from email address.

    Args:
        email_address: Email address

    Returns:
        Domain string
    """
    _, addr = email.utils.parseaddr(email_address)
    if '@' in addr:
        return addr.split('@')[1]
    return addr


def check_lookalike_domain(domain: str, legitimate_domains: Set[str]) -> bool:
    """
    Check if domain is a lookalike of legitimate domain.

    Args:
        domain: Domain to check
        legitimate_domains: Set of known legitimate domains

    Returns:
        True if lookalike detected
    """
    domain = domain.lower().replace('www.', '')

    for legit in legitimate_domains:
        legit = legit.lower().replace('www.', '')

        # Direct match
        if domain == legit:
            return False

        # Check for character substitutions
        substitutions = {
            '0': 'o', '1': 'l', '1': 'i', '5': 's',
            '8': 'b', '|': 'l', '@': 'a'
        }

        for num, char in substitutions.items():
            if num in domain:
                test_domain = domain.replace(num, char)
                if test_domain == legit:
                    return True

        # Check for missing dots
        without_dots = domain.replace('.', '')
        legit_without_dots = legit.replace('.', '')
        if without_dots == legit_without_dots and len(domain) < len(legit):
            return True

        # Check for typos (Levenshtein distance 1-2)
        if levenshtein_distance(domain, legit) <= 2:
            return True

    return False


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def extract_keywords(text: str, keyword_sets: dict) -> dict:
    """
    Extract keywords from text based on predefined sets.

    Args:
        text: Text to search
        keyword_sets: Dict of category -> list of keywords

    Returns:
        Dict of category -> list of found keywords
    """
    text_lower = text.lower()
    found = {}

    for category, keywords in keyword_sets.items():
        found[category] = [kw for kw in keywords if kw.lower() in text_lower]

    return found


def detect_urgency_keywords(text: str) -> List[str]:
    """
    Detect urgency/profiling keywords in text.

    Args:
        text: Text to analyze

    Returns:
        List of urgency keywords found
    """
    urgency_keywords = [
        'urgent', 'immediately', 'asap', 'right away', 'at once',
        'deadline', 'expires', 'expiring', 'limited time',
        'act now', 'don\'t wait', 'hurry', 'time sensitive',
        'final notice', 'last chance', 'account suspended',
        'verify immediately', 'confirm now', 'payment overdue',
        'unusual activity', 'security alert', 'suspended'
    ]

    text_lower = text.lower()
    return [kw for kw in urgency_keywords if kw in text_lower]


def detect_pressure_language(text: str) -> List[str]:
    """
    Detect pressure language in text.

    Args:
        text: Text to analyze

    Returns:
        List of pressure phrases found
    """
    pressure_phrases = [
        'you must', 'required to', 'mandatory', 'compulsory',
        'or else', 'otherwise', 'failure to', 'legal action',
        'immediate attention', 'serious consequences',
        'will be terminated', 'will be closed', 'will be suspended'
    ]

    text_lower = text.lower()
    return [phrase for phrase in pressure_phrases if phrase in text_lower]


def detect_social_engineering(text: str) -> List[str]:
    """
    Detect social engineering tactics in text.

    Args:
        text: Text to analyze

    Returns:
        List of tactics found
    """
    tactics = {
        'authority': ['ceo', 'manager', 'director', 'hr', 'it department', 'security team'],
        'urgency': detect_urgency_keywords(text),
        'fear': ['legal', 'lawsuit', 'police', 'fbi', 'investigation', 'fraud', 'unauthorized'],
        'curiosity': ['secret', 'confidential', 'exclusive', 'special offer', 'prize'],
        'greed': ['free money', 'inheritance', 'lottery', 'winner', 'claim', 'reward'],
        'help': ['help needed', 'assist', 'favor', 'stranded', 'emergency', 'trouble']
    }

    found = []
    for tactic, keywords in tactics.items():
        for kw in keywords:
            if kw.lower() in text.lower():
                found.append(f"{tactic}: {kw}")

    return found


def detect_grammar_issues(text: str) -> List[str]:
    """
    Detect common grammar issues in text.

    Args:
        text: Text to analyze

    Returns:
        List of issues found
    """
    issues = []

    # Check for excessive exclamation marks
    if '!!!' in text:
        issues.append("excessive exclamation marks")

    # Check for ALL CAPS words (excluding common acronyms)
    words = text.split()
    acronyms = {'https', 'http', 'www', 'com', 'org', 'net', 'edu', 'gov', 'id', 'ssn', 'atm', 'pdf'}
    caps_words = [w for w in words if w.isupper() and w.lower() not in acronyms and len(w) > 2]
    if len(caps_words) > 2:
        issues.append("excessive capitalization")

    # Check for multiple spaces
    if '  ' in text:
        issues.append("irregular spacing")

    # Check for missing spaces after punctuation
    if re.search(r'[.,!?][a-zA-Z]', text):
        issues.append("missing spaces after punctuation")

    return issues
