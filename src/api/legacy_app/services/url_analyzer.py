"""
URL analysis service for quick phishing detection.

Provides fast URL-only analysis without full email processing.
"""
import re
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import tldextract
import dns.resolver

from src.api.legacy_app.config import settings
from src.api.legacy_app.utils.logger import get_logger

logger = get_logger(__name__)

# Suspicious TLDs
SUSPICIOUS_TLDS = {
    '.xyz', '.top', '.tk', '.ga', '.cf', '.gq', '.ml',
    '.cc', '.pw', '.club', '.online', '.site', '.info'
}

# Legitimate financial domains (examples)
LEGITIMATE_BANKS = {
    'chase.com', 'wellsfargo.com', 'bankofamerica.com',
    'citi.com', 'usbank.com', 'pnc.com', 'capitalone.com',
    'tdbank.com', 'schwab.com', 'fidelity.com'
}


class URLAnalyzer:
    """
    Fast URL analysis for phishing detection.

    Analyzes URLs for common phishing indicators without full ML models.
    """

    def __init__(self):
        """Initialize URL analyzer."""
        self.suspicious_tlds = SUSPICIOUS_TLDS
        self.legitimate_banks = LEGITIMATE_BANKS
        logger.info("URL analyzer initialized")

    async def analyze_url(
        self,
        url: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single URL for phishing indicators.

        Args:
            url: URL to analyze
            context: Optional context (sender, subject, etc.)

        Returns:
            Dictionary with analysis results
        """
        import time
        start_time = time.time()

        try:
            # Parse URL
            parsed = urlparse(url)

            # Extract domain parts
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}"
            subdomain = extracted.subdomain

            # Run all checks
            checks = {
                "has_ip_address": self._check_ip_address(extracted.domain),
                "suspicious_tld": self._check_suspicious_tld(extracted.suffix),
                "has_port": parsed.port is not None,
                "suspicious_subdomain": self._check_suspicious_subdomain(subdomain),
                "url_length": len(url),
                "domain_length": len(extracted.domain),
                "subdomain_count": len(subdomain.split('.')) if subdomain else 0,
                "has_https": parsed.scheme == "https",
                "special_char_ratio": self._calculate_special_char_ratio(url),
                "has_suspicious_words": self._check_suspicious_words(url),
                "bank_impersonation": self._check_bank_impersonation(domain),
                "typosquatting": self._check_typosquatting(domain),
                "url_shortener": self._check_url_shortener(url)
            }

            # Calculate risk score
            risk_score = self._calculate_risk_score(checks)

            # Determine verdict
            verdict = self._determine_verdict(risk_score, checks)

            # Generate explanation
            explanation = self._generate_explanation(checks, risk_score, context)

            processing_time_ms = (time.time() - start_time) * 1000

            return {
                "url": url,
                "verdict": verdict,
                "risk_score": risk_score,
                "checks": checks,
                "explanation": explanation,
                "processing_time_ms": processing_time_ms
            }

        except Exception as e:
            logger.error(f"Failed to analyze URL: {e}", exc_info=True)
            raise ValueError(f"URL analysis failed: {e}")

    def _check_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address."""
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return bool(re.match(ip_pattern, domain))

    def _check_suspicious_tld(self, suffix: str) -> bool:
        """Check if TLD is commonly used in phishing."""
        return f".{suffix}" in self.suspicious_tlds

    def _check_suspicious_subdomain(self, subdomain: str) -> bool:
        """Check for suspicious subdomain patterns."""
        if not subdomain:
            return False

        suspicious_patterns = [
            'secure', 'verify', 'confirm', 'account', 'login',
            'signin', 'auth', 'banking', 'chase', 'wells',
            'fargo', 'support', 'service', 'alert'
        ]

        subdomain_lower = subdomain.lower()
        return any(pattern in subdomain_lower for pattern in suspicious_patterns)

    def _calculate_special_char_ratio(self, url: str) -> float:
        """Calculate ratio of special characters in URL."""
        special_chars = set('!@#$%^&*()_+=-{}[]|\\:;"\'<>?/~`')
        special_count = sum(1 for c in url if c in special_chars)
        return special_count / len(url) if url else 0.0

    def _check_suspicious_words(self, url: str) -> List[str]:
        """Check for suspicious words in URL."""
        suspicious_words = [
            'verify', 'confirm', 'account', 'login', 'signin',
            'suspend', 'block', 'secure', 'update', 'urgent',
            'immediate', 'banking', 'authenticate'
        ]

        found = []
        url_lower = url.lower()

        for word in suspicious_words:
            if word in url_lower:
                found.append(word)

        return found

    def _check_bank_impersonation(self, domain: str) -> Optional[Dict[str, Any]]:
        """Check if domain is impersonating a legitimate bank."""
        domain_lower = domain.lower()

        for bank in self.legitimate_banks:
            # Extract bank name from domain
            bank_name = bank.replace('.com', '')
            if bank_name in domain_lower and domain != bank:
                return {
                    "impersonated": bank,
                    "similarity": self._calculate_similarity(domain, bank),
                    "actual_domain": bank
                }

        return None

    def _check_typosquatting(self, domain: str) -> Optional[Dict[str, Any]]:
        """Check for typosquatting (common misspellings)."""
        domain_lower = domain.lower()

        # Common typosquatting patterns
        typos = {
            'chase': ['chase', 'chase-bank', 'chasebank'],
            'wellsfargo': ['wellfargo', 'wells-fargo', 'welsfargo'],
            'bankofamerica': ['bank-of-america', 'bankamerica'],
        }

        for legitimate, typos_list in typos.items():
            for typo in typos_list:
                if typo in domain_lower and legitimate not in domain_lower:
                    return {
                        "legitimate": f"{legitimate}.com",
                        "typo": typo,
                        "detected": True
                    }

        return None

    def _check_url_shortener(self, url: str) -> bool:
        """Check if URL uses a URL shortening service."""
        shorteners = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co',
            'ow.ly', 'is.gd', 'buff.ly', 'bit.do'
        ]

        return any(shortener in url for shortener in shorteners)

    def _calculate_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate string similarity (simple Levenshtein-like)."""
        # Simple character overlap ratio
        set1 = set(domain1.lower())
        set2 = set(domain2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _calculate_risk_score(self, checks: Dict[str, Any]) -> int:
        """
        Calculate overall risk score (0-100).

        Weighted sum of risk factors.
        """
        score = 0

        # High-risk indicators (30 points each)
        if checks["has_ip_address"]:
            score += 30
        if checks["bank_impersonation"]:
            score += 30
        if checks["typosquatting"]:
            score += 30

        # Medium-risk indicators (15 points each)
        if checks["suspicious_tld"]:
            score += 15
        if checks["has_suspicious_words"]:
            score += min(15, len(checks["has_suspicious_words"]) * 5)

        # Low-risk indicators (5 points each)
        if checks["suspicious_subdomain"]:
            score += 5
        if checks["has_port"]:
            score += 5
        if checks["url_shortener"]:
            score += 5

        # HTTPS reduces risk
        if checks["has_https"]:
            score = max(0, score - 10)

        return min(100, score)

    def _determine_verdict(self, risk_score: int, checks: Dict[str, Any]) -> str:
        """Determine verdict based on risk score."""
        if risk_score >= 70:
            return "PHISHING"
        elif risk_score >= 40:
            return "SUSPICIOUS"
        else:
            return "LEGITIMATE"

    def _generate_explanation(
        self,
        checks: Dict[str, Any],
        risk_score: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation."""
        reasons = []

        # High-risk factors
        if checks["has_ip_address"]:
            reasons.append("URL contains IP address instead of domain name")

        if checks["bank_impersonation"]:
            imp = checks["bank_impersonation"]
            reasons.append(f"Domain appears to impersonate {imp['impersonated']}")

        if checks["typosquatting"]:
            typo = checks["typosquatting"]
            reasons.append(f"Possible typosquatting of {typo['legitimate']}")

        if checks["suspicious_tld"]:
            reasons.append(f"Uses suspicious top-level domain")

        if checks["suspicious_subdomain"]:
            reasons.append("Contains suspicious subdomain patterns")

        if checks["has_suspicious_words"]:
            words = ", ".join(checks["has_suspicious_words"])
            reasons.append(f"Contains suspicious words: {words}")

        if not reasons:
            reasons.append("No significant risk indicators detected")

        # Context-aware explanation
        if context:
            sender = context.get("sender", "")
            if sender:
                sender_domain = sender.split("@")[-1] if "@" in sender else ""
                if sender_domain and "suspicious" in " ".join(reasons).lower():
                    reasons.append(f"Sender domain ({sender_domain}) differs from URL")

        return ". ".join(reasons) + "."


# Global URL analyzer instance
url_analyzer = URLAnalyzer()
