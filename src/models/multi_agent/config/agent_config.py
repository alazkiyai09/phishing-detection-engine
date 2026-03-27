"""
Configuration for agent weights, thresholds, and coordination logic.
"""

from typing import Dict

# Agent weights for weighted voting
# Higher weight = agent's opinion counts more
AGENT_WEIGHTS: Dict[str, float] = {
    "url_analyst": 1.2,      # URL analysis is highly indicative
    "content_analyst": 1.0,   # Content analysis is important
    "header_analyst": 1.1,   # Header analysis is reliable
    "visual_analyst": 0.8,   # Visual analysis is supplementary (placeholder)
}

# Confidence threshold for considering an agent's verdict as "high confidence"
CONFIDENCE_THRESHOLD: float = 0.7

# Threshold for conflict resolution
# If agents disagree by more than this margin, use special logic
CONFLICT_RESOLUTION_THRESHOLD: float = 0.5

# Coordinator configuration
COORDINATION_CONFIG = {
    "min_agents_required": 2,  # Minimum agents needed for decision
    "unanimous_threshold": 0.9,  # If all agents agree with this confidence, skip conflict resolution
    "majority_threshold": 0.6,   # Simple majority threshold
    "tie_breaker": "url_analyst",  # Agent to trust in case of tie
}

# URL Analyst specific configuration
URL_ANALYST_CONFIG = {
    "suspicious_patterns": [
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP addresses
        r"[a-z0-9\-]+\.tk|\.ml|\.ga|\.cf",  # Suspicious TLDs
        r".*paypal.*login.*",  # Brand impersonation
        r".*verify.*account.*",  # Verification scam
    ],
    "legitimate_domains": [
        "google.com",
        "microsoft.com",
        "amazon.com",
        "apple.com",
        "facebook.com",
    ],
}

# Financial domain-specific patterns
FINANCIAL_PATTERNS = {
    "bank_names": [
        "chase", "bank of america", "wells fargo", "citi", "citibank",
        "capital one", "hsbc", "barclays", "deutsche bank", "hsbc",
    ],
    "urgency_keywords": [
        "immediate action required", "urgent notice", "account suspended",
        "verify immediately", "wire transfer", "payment due",
    ],
    "credential_keywords": [
        "password", "ssn", "social security", "account number",
        "routing number", "credit card", "login credentials",
    ],
}

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    "openai": {
        "rate": 3.0,  # requests per second
        "capacity": 10,  # burst capacity
    },
    "ollama": {
        "rate": 10.0,  # local model, higher rate
        "capacity": 20,
    },
}

# Cache configuration
CACHE_CONFIG = {
    "enabled": True,
    "max_size": 1000,
    "default_ttl": 3600,  # 1 hour
    "persistent": False,
    "cache_dir": ".cache/llm",
}
