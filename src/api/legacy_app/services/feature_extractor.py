"""
Feature extraction service integrating Day 1 pipeline.

This module wraps the PhishingFeaturePipeline from the Day 1 project
and provides a clean interface for the API.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import email
from email import policy
from email.parser import BytesParser
import re
from datetime import datetime

import pandas as pd
import numpy as np

from src.api.legacy_app.config import settings
from src.api.legacy_app.utils.logger import get_logger

logger = get_logger(__name__)

# Add Day 1 project to path
DAY1_PATH = Path(settings.DAY1_PIPELINE_PATH)
if DAY1_PATH.exists():
    sys.path.insert(0, str(DAY1_PATH))
    logger.info(f"Added Day 1 path to sys.path: {DAY1_PATH}")
else:
    logger.warning(f"Day 1 path not found: {DAY1_PATH}")

# Import Day 1 components
try:
    from src.transformers_backup.phishing_pipeline import PhishingFeaturePipeline
    from src.utils.email_parser import SafeEmailParser
    DAY1_AVAILABLE = True
    logger.info("Day 1 components imported successfully")
except ImportError as e:
    DAY1_AVAILABLE = False
    logger.warning(f"Failed to import Day 1 components: {e}")


class FeatureExtractionService:
    """
    Service for extracting features from emails using Day 1 pipeline.

    Handles both raw EML format and pre-parsed email data.
    """

    def __init__(self):
        """Initialize feature extraction service."""
        self.pipeline = None
        self.email_parser = None
        self._initialized = False

        if DAY1_AVAILABLE:
            try:
                self.pipeline = PhishingFeaturePipeline()
                self.email_parser = SafeEmailParser()
                self._initialized = True
                logger.info("Feature extraction service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize feature extraction: {e}", exc_info=True)

    @property
    def is_available(self) -> bool:
        """Check if feature extraction service is available."""
        return self._initialized and self.pipeline is not None

    async def extract_from_raw_email(self, raw_email: str) -> Dict[str, Any]:
        """
        Extract features from raw RFC 822 email (EML format).

        Args:
            raw_email: Raw email content as string

        Returns:
            Dictionary with parsed email data and extracted features
        """
        if not self.is_available:
            raise RuntimeError("Feature extraction service not available")

        start_time = datetime.now()

        try:
            # Parse raw email
            parsed_email = self._parse_raw_email(raw_email)

            # Extract features
            features = await self.extract_from_parsed_email(parsed_email)

            # Add parsing metadata
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "parsed_email": parsed_email,
                "features": features,
                "processing_time_ms": processing_time_ms
            }

        except Exception as e:
            logger.error(f"Failed to extract features from raw email: {e}", exc_info=True)
            raise

    async def extract_from_parsed_email(self, parsed_email: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from pre-parsed email data.

        Args:
            parsed_email: Dictionary with email fields (headers, body, urls, etc.)

        Returns:
            Dictionary with extracted features (normalized to [0, 1])
        """
        if not self.is_available:
            raise RuntimeError("Feature extraction service not available")

        try:
            # Use simplified feature extraction
            features = self._extract_features_simple(parsed_email)

            logger.debug(f"Extracted {len(features)} features")

            return {
                "features": features,
                "feature_names": list(features.keys()),
                "n_features": len(features)
            }

        except Exception as e:
            logger.error(f"Failed to extract features from parsed email: {e}", exc_info=True)
            raise

    def _parse_raw_email(self, raw_email: str) -> Dict[str, Any]:
        """
        Parse raw RFC 822 email.

        Args:
            raw_email: Raw email content

        Returns:
            Parsed email dictionary
        """
        try:
            # Use Day 1 email parser if available
            if self.email_parser:
                parsed = self.email_parser.parse_email(raw_email)
                return self._normalize_parsed_email(parsed)

            # Fallback to Python email parser
            msg = email.message_from_string(raw_email, policy=policy.default)

            return {
                "headers": dict(msg.items()),
                "subject": msg.get("subject", ""),
                "from_addr": msg.get("from", ""),
                "to_addrs": self._extract_email_addresses(msg.get("to", "")),
                "cc_addrs": self._extract_email_addresses(msg.get("cc", "")),
                "reply_to": msg.get("reply-to", ""),
                "body_text": self._extract_body_text(msg),
                "body_html": self._extract_body_html(msg),
                "attachments": [],  # TODO: Extract attachment metadata
                "urls": self._extract_urls_from_email(msg)
            }

        except Exception as e:
            logger.error(f"Failed to parse raw email: {e}", exc_info=True)
            raise ValueError(f"Invalid email format: {e}")

    def _normalize_parsed_email(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parsed email to standard format.

        Ensures all expected fields are present.
        """
        return {
            "headers": parsed.get("headers", {}),
            "subject": parsed.get("subject", ""),
            "from_addr": parsed.get("from_addr", ""),
            "to_addrs": parsed.get("to_addrs", []),
            "cc_addrs": parsed.get("cc_addrs", []),
            "reply_to": parsed.get("reply_to", ""),
            "body_text": parsed.get("body_text", ""),
            "body_html": parsed.get("body_html", ""),
            "urls": parsed.get("urls", []),
            "attachments": parsed.get("attachments", [])
        }

    def _parsed_email_to_dataframe(self, parsed_email: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert parsed email to DataFrame format expected by pipeline.

        Args:
            parsed_email: Parsed email dictionary

        Returns:
            DataFrame with one row
        """
        # Create single-row DataFrame with all required fields
        data = {
            "body": parsed_email.get("body_text", ""),
            "headers": parsed_email.get("headers", {}),
            "subject": parsed_email.get("subject", ""),
            "from_addr": parsed_email.get("from_addr", ""),
            "body_html": parsed_email.get("body_html", ""),
            "attachments": parsed_email.get("attachments", [])
        }

        return pd.DataFrame([data])

    def _extract_email_addresses(self, field: str) -> List[str]:
        """Extract email addresses from header field."""
        if not field:
            return []

        # Simple email extraction
        pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        return re.findall(pattern, str(field))

    def _extract_body_text(self, msg) -> str:
        """Extract plain text body from email message."""
        text_parts = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text_parts.append(part.get_payload(decode=True).decode("utf-8", errors="ignore"))
        return "\n".join(text_parts)

    def _extract_body_html(self, msg) -> str:
        """Extract HTML body from email message."""
        html_parts = []
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                html_parts.append(part.get_payload(decode=True).decode("utf-8", errors="ignore"))
        return "\n".join(html_parts)

    def _extract_urls_from_email(self, msg) -> List[Dict[str, str]]:
        """Extract URLs from email body."""
        urls = []
        body = self._extract_body_text(msg)

        # Simple URL extraction
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        matches = re.findall(url_pattern, body)

        for url in matches:
            urls.append({
                "original": url,
                "suspicious": False  # TODO: Add basic suspicious detection
            })

        return urls

    def _extract_features_simple(self, parsed_email: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract simplified features compatible with XGBoost model.

        Returns dictionary of 70 features matching the trained model.
        """
        body_text = parsed_email.get("body_text", "")
        subject = parsed_email.get("subject", "")
        from_addr = parsed_email.get("from_addr", "")
        headers = parsed_email.get("headers", {})
        urls = parsed_email.get("urls", [])

        # URL features (1-10)
        url_count = len(urls)
        has_ip_url = 1.0 if any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', u.get("original", "")) for u in urls) else 0.0

        url_lengths = [len(u.get("original", "")) for u in urls] if urls else [0]
        avg_url_length = sum(url_lengths) / len(url_lengths) if url_lengths else 0.0
        max_url_length = max(url_lengths) if url_lengths else 0.0

        suspicious_tlds = ['.xyz', '.top', '.zip', '.tk', '.ga', '.cf', '.gq']
        has_suspicious_tld = 1.0 if any(any(tld in u.get("original", "").lower() for tld in suspicious_tlds) for u in urls) else 0.0
        has_https = 1.0 if any(u.get("original", "").startswith("https://") for u in urls) else 0.0

        avg_subdomain_count = 0.0
        for u in urls:
            domain = u.get("original", "").split("://")[-1].split("/")[0]
            subdomain_count = domain.count('.') - 1
            if url_count > 0:
                avg_subdomain_count += subdomain_count / url_count

        url_shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']
        has_url_shortener = 1.0 if any(any(short in u.get("original", "").lower() for short in url_shorteners) for u in urls) else 0.0

        special_chars = "!@#$%^&*()_+=-[]{}|;':\",./<>?"
        special_char_ratio = 0.0
        if urls:
            total_chars = sum(len(u.get("original", "")) for u in urls)
            special_count = sum(sum(1 for c in u.get("original", "") if c in special_chars) for u in urls)
            special_char_ratio = special_count / total_chars if total_chars > 0 else 0.0

        has_port_specified = 1.0 if any(':' in u.get("original", "").split('://')[-1] for u in urls) else 0.0

        # Header features (11-18)
        spf_pass = 0.0  # Would need actual SPF check
        spf_fail = 0.0
        dkim_present = 0.0
        dkim_valid = 0.0
        dmarc_pass = 0.0
        dmarc_fail = 0.0
        hop_count = float(headers.get('hop_count', 0))
        reply_to_mismatch = 1.0 if parsed_email.get("reply_to") and parsed_email.get("reply_to") != from_addr else 0.0

        # Sender features (19-28)
        has_priority_flag = 1.0 if headers.get('priority') or headers.get('x-priority') else 0.0
        has_authentication_results = 1.0 if 'authentication-results' in [h.lower() for h in headers.keys()] else 0.0

        freemail_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
        domain = from_addr.split('@')[-1] if '@' in from_addr else ''
        is_freemail = 1.0 if domain.lower() in freemail_domains else 0.0

        display_name = from_addr.split('<')[0].strip() if '<' in from_addr else ''
        display_name_mismatch = 0.0  # Would need comparison
        display_name_has_bank = 1.0 if any(bank in display_name.lower() for bank in ['chase', 'bank', 'wells fargo', 'citi']) else 0.0

        domain_age_days = 365.0  # Placeholder - would need WHOIS lookup
        has_numbers_in_domain = 1.0 if any(c.isdigit() for c in domain) else 0.0
        email_address_length = float(len(from_addr))
        domain_length = float(len(domain))
        sender_name_length = float(len(display_name))
        has_reply_to_path = 1.0 if parsed_email.get("reply_to") else 0.0
        suspicious_pattern = 1.0 if 'secure' in domain.lower() or 'verify' in domain.lower() else 0.0

        # Content features (29-38)
        urgency_keywords = ['urgent', 'immediately', 'hurry', 'expires', 'deadline', 'act now']
        urgency_keyword_count = float(sum(1 for kw in urgency_keywords if kw.lower() in body_text.lower()))

        cta_button_count = float(body_text.lower().count('click') + body_text.lower().count('button'))
        threat_keywords = ['suspend', 'terminate', 'close', 'legal action', 'consequence']
        threat_language_count = float(sum(1 for kw in threat_keywords if kw.lower() in body_text.lower()))

        financial_terms = ['payment', 'invoice', 'transaction', 'transfer', 'billing']
        financial_term_count = float(sum(1 for kw in financial_terms if kw.lower() in body_text.lower()))

        immediate_action_count = float(body_text.lower().count('now') + body_text.lower().count('immediately'))
        verification_request_count = float(body_text.lower().count('verify') + body_text.lower().count('confirm'))
        click_here_count = float(body_text.lower().count('click here'))
        password_request_count = float(sum(1 for kw in ['password', 'pin', 'ssn', 'social security'] if kw.lower() in body_text.lower()))
        account_suspended_count = float(body_text.lower().count('suspend') + body_text.lower().count('account'))

        # Structural features (39-49)
        url_in_body_count = float(len(re.findall(r'http[s]?://\S+', body_text)))
        html_text_ratio = 0.5  # Placeholder
        has_attachments = 1.0 if parsed_email.get("attachments") else 0.0
        attachment_count = float(len(parsed_email.get("attachments", [])))
        has_executable_attachment = 0.0
        has_office_attachment = 0.0
        embedded_image_count = 0.0
        external_image_count = 0.0
        has_forms = 0.0
        has_javascript = 0.0
        email_size_kb = float(len(body_text) / 1024)

        # Linguistic features (50-58)
        spelling_error_rate = 0.05  # Placeholder
        grammar_score_proxy = 0.5
        formality_score = 0.5
        reading_ease_score = 0.5
        sentence_count = float(len([s for s in body_text.split('.') if s.strip()]))
        avg_sentence_length = 0.0
        sentences = [s.strip() for s in body_text.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        exclamation_mark_count = float(body_text.count('!'))
        question_mark_count = float(body_text.count('?'))
        all_caps_words = [w for w in body_text.split() if w.isupper() and len(w) > 1]
        all_caps_ratio = float(len(all_caps_words) / len(body_text.split())) if body_text.split() else 0.0
        punctuation_ratio = 0.0
        if body_text:
            punctuation_chars = sum(1 for c in body_text if c in '!@#$%^&*()_+-=[]{}|;:\'",.<>?/')
            punctuation_ratio = float(punctuation_chars / len(body_text))

        # Financial features (59-70)
        bank_names = ['chase', 'bank of america', 'wells fargo', 'citi', 'capital one']
        bank_impersonation_score = float(sum(1 for bank in bank_names if bank in body_text.lower() or bank in from_addr.lower()))

        wire_urgency_keywords = ['wire transfer', 'swift', 'iban', 'ach']
        wire_urgency_score = float(sum(1 for kw in wire_urgency_keywords if kw.lower() in body_text.lower()))

        credential_keywords = ['ssn', 'social security', 'account number', 'routing number', 'credit card']
        credential_harvesting_score = float(sum(1 for kw in credential_keywords if kw.lower() in body_text.lower()))

        invoice_keywords = ['invoice', 'receipt', 'payment due', 'bill']
        invoice_terminology_density = float(sum(1 for kw in invoice_keywords if kw.lower() in body_text.lower()))

        account_number_request = 1.0 if 'account number' in body_text.lower() else 0.0
        routing_number_request = 1.0 if 'routing' in body_text.lower() else 0.0
        ssn_request = 1.0 if any(kw in body_text.lower() for kw in ['ssn', 'social security']) else 0.0

        payment_urgency_keywords = ['payment overdue', 'immediate payment', 'pay now', 'past due']
        payment_urgency_score = float(sum(1 for kw in payment_urgency_keywords if kw.lower() in body_text.lower()))

        financial_institutions = float(sum(1 for bank in bank_names if bank in body_text.lower()))

        wire_transfer_keywords_list = ['wire', 'transfer', 'swift', 'iban', 'ach']
        wire_transfer_keywords = float(sum(1 for kw in wire_transfer_keywords_list if kw.lower() in body_text.lower()))

        # Return all 70 features
        return {
            "url_count": url_count,
            "has_ip_url": has_ip_url,
            "avg_url_length": min(avg_url_length / 100.0, 1.0),  # Normalize
            "max_url_length": min(max_url_length / 200.0, 1.0),
            "has_suspicious_tld": has_suspicious_tld,
            "has_https": has_https,
            "avg_subdomain_count": min(avg_subdomain_count / 5.0, 1.0),
            "has_url_shortener": has_url_shortener,
            "special_char_ratio": min(special_char_ratio, 1.0),
            "has_port_specified": has_port_specified,
            "spf_pass": spf_pass,
            "spf_fail": spf_fail,
            "dkim_present": dkim_present,
            "dkim_valid": dkim_valid,
            "dmarc_pass": dmarc_pass,
            "dmarc_fail": dmarc_fail,
            "hop_count": min(hop_count / 10.0, 1.0),
            "reply_to_mismatch": reply_to_mismatch,
            "has_priority_flag": has_priority_flag,
            "has_authentication_results": has_authentication_results,
            "is_freemail": is_freemail,
            "display_name_mismatch": display_name_mismatch,
            "display_name_has_bank": display_name_has_bank,
            "domain_age_days": min(domain_age_days / 3650.0, 1.0),  # Normalize by 10 years
            "has_numbers_in_domain": has_numbers_in_domain,
            "email_address_length": min(email_address_length / 100.0, 1.0),
            "domain_length": min(domain_length / 50.0, 1.0),
            "sender_name_length": min(sender_name_length / 50.0, 1.0),
            "has_reply_to_path": has_reply_to_path,
            "suspicious_pattern": suspicious_pattern,
            "urgency_keyword_count": min(urgency_keyword_count / 10.0, 1.0),
            "cta_button_count": min(cta_button_count / 20.0, 1.0),
            "threat_language_count": min(threat_language_count / 10.0, 1.0),
            "financial_term_count": min(financial_term_count / 10.0, 1.0),
            "immediate_action_count": min(immediate_action_count / 10.0, 1.0),
            "verification_request_count": min(verification_request_count / 10.0, 1.0),
            "click_here_count": min(click_here_count / 10.0, 1.0),
            "password_request_count": min(password_request_count / 10.0, 1.0),
            "account_suspended_count": min(account_suspended_count / 10.0, 1.0),
            "url_in_body_count": min(url_in_body_count / 20.0, 1.0),
            "html_text_ratio": html_text_ratio,
            "has_attachments": has_attachments,
            "attachment_count": min(attachment_count / 10.0, 1.0),
            "has_executable_attachment": has_executable_attachment,
            "has_office_attachment": has_office_attachment,
            "embedded_image_count": min(embedded_image_count / 10.0, 1.0),
            "external_image_count": min(external_image_count / 10.0, 1.0),
            "has_forms": has_forms,
            "has_javascript": has_javascript,
            "email_size_kb": min(email_size_kb / 100.0, 1.0),
            "spelling_error_rate": spelling_error_rate,
            "grammar_score_proxy": grammar_score_proxy,
            "formality_score": formality_score,
            "reading_ease_score": reading_ease_score,
            "sentence_count": min(sentence_count / 50.0, 1.0),
            "avg_sentence_length": min(avg_sentence_length / 30.0, 1.0),
            "exclamation_mark_count": min(exclamation_mark_count / 20.0, 1.0),
            "question_mark_count": min(question_mark_count / 20.0, 1.0),
            "all_caps_ratio": all_caps_ratio,
            "punctuation_ratio": min(punctuation_ratio, 1.0),
            "bank_impersonation_score": min(bank_impersonation_score / 5.0, 1.0),
            "wire_urgency_score": min(wire_urgency_score / 5.0, 1.0),
            "credential_harvesting_score": min(credential_harvesting_score / 10.0, 1.0),
            "invoice_terminology_density": min(invoice_terminology_density / 5.0, 1.0),
            "account_number_request": account_number_request,
            "routing_number_request": routing_number_request,
            "ssn_request": ssn_request,
            "payment_urgency_score": min(payment_urgency_score / 5.0, 1.0),
            "financial_institution_mentions": min(financial_institutions / 5.0, 1.0),
            "wire_transfer_keywords": min(wire_transfer_keywords / 5.0, 1.0)
        }

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about extracted features.

        Returns:
            Dictionary with feature metadata
        """
        if not self.is_available:
            return {
                "available": False,
                "feature_count": 0,
                "categories": []
            }

        try:
            feature_names = self.pipeline.get_feature_names()

            # Group features by category
            categories = {
                "url": [f for f in feature_names if f.startswith("url_")],
                "header": [f for f in feature_names if f.startswith("spf_") or f.startswith("dkim_") or f.startswith("dmarc_") or f in ["hop_count", "reply_to_mismatch"]],
                "sender": [f for f in feature_names if any(x in f for x in ["freemail", "display_name", "domain_", "email_address", "sender_name"])],
                "content": [f for f in feature_names if any(x in f for x in ["urgency", "cta_button", "threat", "financial", "immediate", "verification", "click_here", "password", "account_suspended"])],
                "structural": [f for f in feature_names if any(x in f for x in ["html", "attachment", "image", "form", "javascript", "email_size"])],
                "linguistic": [f for f in feature_names if any(x in f for x in ["spelling", "grammar", "formality", "reading", "sentence", "exclamation", "question", "caps", "punctuation"])],
                "financial": [f for f in feature_names if any(x in f for x in ["bank", "wire", "credential", "invoice", "account_number", "routing", "ssn", "payment"])]
            }

            return {
                "available": True,
                "feature_count": len(feature_names),
                "categories": {k: len(v) for k, v in categories.items()},
                "all_features": feature_names
            }

        except Exception as e:
            logger.error(f"Failed to get feature info: {e}", exc_info=True)
            return {
                "available": False,
                "error": str(e)
            }


# Global feature extraction service instance
feature_extraction_service = FeatureExtractionService()
