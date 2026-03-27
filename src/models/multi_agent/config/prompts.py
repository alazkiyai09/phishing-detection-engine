"""
Prompt templates for all agents.
Each prompt is designed to elicit structured JSON output.
"""

# System prompt that establishes the LLM's role
SYSTEM_PROMPT = """You are an expert security analyst specializing in phishing email detection.
You analyze emails and determine if they are legitimate or phishing attempts.
Your responses must be objective, detailed, and backed by evidence."""

# Output format specification
JSON_OUTPUT_FORMAT = """
Respond with a valid JSON object following this exact structure:
{
    "is_phishing": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your analysis",
    "evidence": ["list", "of", "specific", "evidence"]
}

IMPORTANT: Your response must be valid JSON only, with no additional text.
"""

# URL Analyst Prompt
URL_ANALYST_PROMPT = """You are the URL Analyst Agent. Your task is to analyze URLs found in emails for phishing indicators.

Analyze the following URLs from an email:
{urls}

Email context:
- Subject: {subject}
- Sender: {sender}
- Body preview: {body_preview}

{json_format}

Look for:
1. IP addresses instead of domain names
2. Typosquatting (e.g., g0ogle.com instead of google.com)
3. Suspicious TLDs (e.g., .tk, .ml, .ga, .cf)
4. Misleading subdomains (e.g., secure.verify.fake.com)
5. Brand impersonation in URLs
6. URL shorteners hiding destinations
7. Mismatch between visible text and actual URL

Provide your analysis as JSON.
"""

# Content Analyst Prompt
CONTENT_ANALYST_PROMPT = """You are the Content Analyst Agent. Your task is to analyze email content for social engineering and phishing tactics.

Email to analyze:
Subject: {subject}
Sender: {sender}
Body: {body}

{json_format}

Look for:
1. Urgency tactics (immediate action required, account suspended)
2. Authority abuse (appearing to be from executive, IT, bank)
3. Threat language (account closure, legal consequences)
4. Requests for sensitive information (passwords, SSN, account numbers)
5. Generic greetings (Dear Customer vs personalized)
6. Poor grammar and spelling (often indicates phishing)
7. Inconsistent branding or formatting
8. Financial pressure (wire transfers, payment requests)
9. Credential harvesting patterns (login verification, password reset)

Provide your analysis as JSON.
"""

# Header Analyst Prompt
HEADER_ANALYST_PROMPT = """You are the Header Analyst Agent. Your task is to analyze email headers for authentication failures and spoofing indicators.

Email headers:
{headers}

Email metadata:
- Subject: {subject}
- Sender: {sender}

{json_format}

Look for:
1. SPF (Sender Policy Framework) failures
2. DKIM (DomainKeys Identified Mail) signature issues
3. DMARC policy failures
4. Mismatched Return-Path and From addresses
5. Suspicious routing paths (unusual hops)
6. X-Originating-IP mismatches
7. Reply-To address different from From address
8. Missing authentication headers
9. Suspicious Received headers (e.g., from compromised servers)

Provide your analysis as JSON.
"""

# Visual Analyst Prompt (Placeholder)
VISUAL_ANALYST_PROMPT = """You are the Visual Analyst Agent. Your task is to analyze the visual appearance of emails for phishing indicators.

[NOTE: This is a placeholder for future visual analysis using screenshots or HTML rendering]

Email metadata:
- Subject: {subject}
- Sender: {sender}
- Body text: {body}

{json_format}

In a full implementation, you would analyze:
1. Logo quality and placement
2. Color scheme consistency with brand
3. Layout and formatting
4. HTML structure for hidden elements
5. Image quality and resolution
6. Visual inconsistencies

For now, provide a basic analysis based on the text content.
"""

# Coordinator Prompt
COORDINATOR_PROMPT = """You are the Coordinator Agent. Your task is to aggregate analyses from specialist agents and make a final decision.

Agent analyses:
{agent_outputs}

{json_format}

Your role:
1. Review all agent outputs
2. Consider the confidence scores of each agent
3. Weigh agent opinions based on their reliability (URL and Header analysts are typically more reliable)
4. Resolve conflicts between agents
5. Make a final decision: phishing or legitimate

Consider:
- High-confidence verdicts should carry more weight
- If most agents agree with high confidence, follow the majority
- If agents disagree significantly, investigate the reasons and make a judgment call
- When in doubt, err on the side of caution (flag as phishing)

Provide your final decision as JSON with this structure:
{
    "verdict": "phishing" or "legitimate",
    "confidence": 0.0-1.0,
    "explanation": "Clear explanation of the decision and how conflicts were resolved",
    "conflicts_resolved": ["list", "of", "conflict", "resolutions"]
}
"""

# Financial Domain Specialist Prompt
FINANCIAL_ANALYST_PROMPT = """You are the Financial Domain Specialist Agent. Your task is to detect bank impersonation and financial fraud attempts.

Email to analyze:
Subject: {subject}
Sender: {sender}
Body: {body}

{json_format}

Look for:
1. Bank impersonation (fake Chase, Wells Fargo, Citi, etc.)
2. Wire transfer urgency (immediate payment required)
3. Account threats (account suspended, unusual activity)
4. Credential harvesting (verify account info, update details)
5. Transaction verification requests
6. Payment redirection instructions
7. Invoice or payment fraud

Provide your analysis as JSON, also including:
{
    "is_phishing": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "evidence": ["list"],
    "financial_indicators": {
        "bank_impersonation": true/false,
        "wire_urgency": true/false,
        "credential_harvesting": true/false,
        "account_threats": true/false
    }
}
"""


def format_url_prompt(urls: list, subject: str, sender: str, body_preview: str) -> str:
    """Format the URL analyst prompt with email data."""
    body_truncated = body_preview[:500] if len(body_preview) > 500 else body_preview
    urls_text = "\n".join(f"- {url}" for url in urls) if urls else "(No URLs found)"

    return URL_ANALYST_PROMPT.format(
        urls=urls_text,
        subject=subject,
        sender=sender,
        body_preview=body_truncated,
        json_format=JSON_OUTPUT_FORMAT,
    )


def format_content_prompt(subject: str, sender: str, body: str) -> str:
    """Format the content analyst prompt with email data."""
    return CONTENT_ANALYST_PROMPT.format(
        subject=subject,
        sender=sender,
        body=body,
        json_format=JSON_OUTPUT_FORMAT,
    )


def format_header_prompt(headers: dict, subject: str, sender: str) -> str:
    """Format the header analyst prompt with email data."""
    headers_text = "\n".join(f"{k}: {v}" for k, v in headers.items()) if headers else "(No headers available)"

    return HEADER_ANALYST_PROMPT.format(
        headers=headers_text,
        subject=subject,
        sender=sender,
        json_format=JSON_OUTPUT_FORMAT,
    )


def format_visual_prompt(subject: str, sender: str, body: str) -> str:
    """Format the visual analyst prompt with email data."""
    return VISUAL_ANALYST_PROMPT.format(
        subject=subject,
        sender=sender,
        body=body[:1000],  # Truncate for visual placeholder
        json_format=JSON_OUTPUT_FORMAT,
    )


def format_coordinator_prompt(agent_outputs: list) -> str:
    """Format the coordinator prompt with agent outputs."""
    import json

    outputs_text = "\n\n".join(
        f"### {output['agent_name']}\n"
        f"Verdict: {output['verdict']}\n"
        f"Confidence: {output['confidence']}\n"
        f"Reasoning: {output['reasoning']}\n"
        f"Evidence: {', '.join(output.get('evidence', []))}"
        for output in agent_outputs
    )

    return COORDINATOR_PROMPT.format(
        agent_outputs=outputs_text,
        json_format=JSON_OUTPUT_FORMAT,
    )


def format_financial_prompt(subject: str, sender: str, body: str) -> str:
    """Format the financial analyst prompt with email data."""
    return FINANCIAL_ANALYST_PROMPT.format(
        subject=subject,
        sender=sender,
        body=body,
        json_format=JSON_OUTPUT_FORMAT,
    )
