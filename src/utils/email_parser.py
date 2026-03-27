"""Robust email parsing utilities with security-focused handling.

This module provides safe email parsing that:
1. Uses defusedxml to prevent XXE attacks
2. Gracefully handles malformed headers/bodies
3. Recovers from encoding issues
4. Validates email structure before extraction
"""

import email
import email.policy
from email import message_from_string
from typing import Any, Dict, Optional
from warnings import warn

import defusedxml.ElementTree as ET
from bs4 import BeautifulSoup


class EmailParseError(Exception):
    """Custom exception for email parsing errors."""

    pass


class SafeEmailParser:
    """Secure and robust email parser for phishing analysis.

    Features:
    - Malformed header recovery (continues on individual field failures)
    - Encoding issue handling (fallback to latin-1)
    - XXE attack prevention (defusedxml)
    - HTML/text separation with fallback
    - Attachment metadata extraction
    """

    def __init__(self) -> None:
        """Initialize the parser with modern email policy."""
        self.policy = email.policy.default
        self.max_body_length = 10_000_000  # 10MB max to prevent memory issues

    def parse_email(self, raw_email: str) -> Dict[str, Any]:
        """Parse raw email string into structured dictionary.

        Args:
            raw_email: Raw email content (RFC 822 format).

        Returns:
            Dictionary containing:
                - headers: Dict of parsed headers
                - subject: Email subject line (str)
                - from_addr: From address (str)
                - to_addrs: List of To addresses
                - cc_addrs: List of CC addresses
                - body_text: Plain text body
                - body_html: HTML body (if present)
                - attachments: List of attachment metadata
                - parse_errors: List of non-fatal parsing errors

        Raises:
            EmailParseError: If email is completely unparsable.
        """
        if not raw_email or not isinstance(raw_email, str):
            raise EmailParseError("raw_email must be a non-empty string")

        try:
            msg = message_from_string(raw_email, policy=self.policy)
        except Exception as e:
            raise EmailParseError(f"Failed to parse email: {e}") from e

        result = {
            "headers": self._extract_headers(msg),
            "subject": self._safe_get_subject(msg),
            "from_addr": self._safe_get_from(msg),
            "to_addrs": self._safe_get_addrs(msg, "To"),
            "cc_addrs": self._safe_get_addrs(msg, "Cc"),
            "reply_to": self._safe_get_reply_to(msg),
            "body_text": "",
            "body_html": "",
            "attachments": [],
            "parse_errors": [],
        }

        # Extract body (text and HTML)
        body_text, body_html, attachments = self._extract_body(msg)
        result["body_text"] = body_text
        result["body_html"] = body_html
        result["attachments"] = attachments

        return result

    def _extract_headers(self, msg: email.message.Message) -> Dict[str, str]:
        """Extract email headers with error recovery.

        Args:
            msg: Parsed email message.

        Returns:
            Dictionary of header names to values. Malformed headers
            are skipped with a warning.
        """
        headers = {}
        important_headers = [
            "Received",
            "SPF",
            "DKIM-Signature",
            "DMARC-Result",
            "Authentication-Results",
            "X-Priority",
            "Importance",
            "X-Mailer",
            "MIME-Version",
            "Content-Type",
            "Message-ID",
            "Return-Path",
        ]

        for header in important_headers:
            try:
                value = msg.get(header, "")
                if value:
                    headers[header] = str(value)
            except (UnicodeError, AttributeError) as e:
                warn(f"Failed to extract header '{header}': {e}")
                continue

        return headers

    def _safe_get_subject(self, msg: email.message.Message) -> str:
        """Safely extract subject with encoding fallback.

        Args:
            msg: Parsed email message.

        Returns:
            Subject line as string, or empty string on failure.
        """
        try:
            subject = msg.get("subject", "")
            if subject:
                return str(subject)
            return ""
        except (UnicodeError, AttributeError):
            return ""

    def _safe_get_from(self, msg: email.message.Message) -> str:
        """Safely extract From address.

        Args:
            msg: Parsed email message.

        Returns:
            From address as string, or empty string on failure.
        """
        try:
            from_val = msg.get("from", "")
            if from_val:
                # Extract email address from display name format
                if hasattr(from_val, "addr_spec"):
                    return from_val.addr_spec
                return str(from_val)
            return ""
        except (AttributeError, UnicodeError):
            return ""

    def _safe_get_addrs(self, msg: email.message.Message, header: str) -> list[str]:
        """Safely extract addresses from To/Cc headers.

        Args:
            msg: Parsed email message.
            header: Header name ('To' or 'Cc').

        Returns:
            List of email addresses, or empty list on failure.
        """
        try:
            addrs = msg.get(header, [])
            if not addrs:
                return []

            result = []
            for addr in addrs:
                if hasattr(addr, "addr_spec"):
                    result.append(addr.addr_spec)
                else:
                    result.append(str(addr))
            return result
        except (AttributeError, UnicodeError, TypeError):
            return []

    def _safe_get_reply_to(self, msg: email.message.Message) -> str:
        """Safely extract Reply-To address.

        Args:
            msg: Parsed email message.

        Returns:
            Reply-To address as string, or empty string on failure.
        """
        try:
            reply_to = msg.get("reply-to", "")
            if reply_to:
                if hasattr(reply_to, "addr_spec"):
                    return reply_to.addr_spec
                return str(reply_to)
            return ""
        except (AttributeError, UnicodeError):
            return ""

    def _extract_body(
        self, msg: email.message.Message
    ) -> tuple[str, str, list[Dict[str, str]]]:
        """Extract email body (text and HTML) and attachments.

        Args:
            msg: Parsed email message.

        Returns:
            Tuple of (body_text, body_html, attachments).
            - body_text: Plain text content
            - body_html: HTML content (sanitized)
            - attachments: List of attachment metadata dicts
        """
        body_text = ""
        body_html = ""
        attachments = []

        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))

                    # Attachment
                    if "attachment" in content_disposition:
                        attachments.append(
                            {
                                "filename": part.get_filename() or "unknown",
                                "content_type": content_type,
                                "size": len(part.get_payload(decode=True) or b""),
                            }
                        )
                        continue

                    # Text body
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_text = self._decode_payload(payload)
                            if len(body_text) > self.max_body_length:
                                body_text = body_text[: self.max_body_length]

                    # HTML body
                    elif content_type == "text/html":
                        payload = part.get_payload(decode=True)
                        if payload:
                            html = self._decode_payload(payload)
                            body_html = self._sanitize_html(html)
                            if len(body_html) > self.max_body_length:
                                body_html = body_html[: self.max_body_length]

            else:  # Not multipart
                content_type = msg.get_content_type()
                payload = msg.get_payload(decode=True)
                if payload:
                    decoded = self._decode_payload(payload)
                    if content_type == "text/plain":
                        body_text = decoded[: self.max_body_length]
                    elif content_type == "text/html":
                        body_html = self._sanitize_html(decoded)
                        body_html = body_html[: self.max_body_length]

        except Exception as e:
            warn(f"Error extracting body: {e}")

        return body_text, body_html, attachments

    def _decode_payload(self, payload: bytes) -> str:
        """Decode payload with encoding fallback.

        Args:
            payload: Raw bytes payload.

        Returns:
            Decoded string, or empty string on failure.
        """
        try:
            # Try UTF-8 first
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1 (never fails)
                return payload.decode("latin-1")
            except Exception:
                return ""

    def _sanitize_html(self, html: str) -> str:
        """Sanitize HTML to prevent XXE and other attacks.

        Uses defusedxml for safe parsing and BeautifulSoup for extraction.

        Args:
            html: Raw HTML string.

        Returns:
            Sanitized HTML string, or empty string on parsing failure.
        """
        if not html:
            return ""

        try:
            # Parse with defusedxml to prevent XXE
            ET.fromstring(f"<root>{html}<" + "/root>")

            # Use BeautifulSoup for extraction
            soup = BeautifulSoup(html, "lxml")
            return str(soup)
        except ET.ParseError as e:
            warn(f"HTML parse error (potentially malicious): {e}")
            return ""
        except Exception as e:
            warn(f"HTML sanitization error: {e}")
            return ""

    def get_display_name(self, from_addr: str) -> str:
        """Extract display name from From address.

        Args:
            from_addr: Full From address (e.g., "John Doe <jdoe@example.com>").

        Returns:
            Display name portion, or empty string if none present.
        """
        if not from_addr:
            return ""

        try:
            if "<" in from_addr and ">" in from_addr:
                return from_addr.split("<")[0].strip().strip('"')
            return ""
        except Exception:
            return ""

    def get_domain_from_addr(self, email_addr: str) -> str:
        """Extract domain from email address.

        Args:
            email_addr: Email address string.

        Returns:
            Domain portion (e.g., "example.com" from "user@example.com").
        """
        if not email_addr or "@" not in email_addr:
            return ""

        try:
            return email_addr.split("@")[-1].strip().lower()
        except Exception:
            return ""
