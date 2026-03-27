"""
URL component analyzer.

Analyzes URLs in email for suspicious patterns including:
- Domain age and reputation
- HTTPS vs HTTP
- Suspicious paths and parameters
- Lookalike domains
- URL shorteners
"""

from typing import List, Set, Optional
from urllib.parse import urlparse, parse_qs
import re

from src.explainability.legacy.utils.data_structures import URL, URLExplanation, EmailData
from src.explainability.legacy.utils.text_processing import check_lookalike_domain


class URLAnalyzer:
    """
    Analyze URLs for suspicious patterns.

    URLs are typically checked after body content by humans.
    """

    # Known legitimate domains
    LEGITIMATE_DOMAINS = {
        'google.com', 'amazon.com', 'apple.com', 'microsoft.com',
        'facebook.com', 'linkedin.com', 'twitter.com',
        'chase.com', 'bankofamerica.com', 'wellsfargo.com',
        'paypal.com', 'ebay.com', 'netflix.com'
    }

    # Known URL shorteners
    URL_SHORTENERS = {
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co',
        'ow.ly', 'is.gd', 'buff.ly', 'bit.do'
    }

    # Suspicious TLDs
    SUSPICIOUS_TLDS = {
        '.xyz', '.top', '.zip', '.mov', '.tk', '.ml', '.ga',
        '.cf', '.gq', '.cc', '.pw'
    }

    # Suspicious keywords in URLs
    SUSPICIOUS_KEYWORDS = [
        'login', 'signin', 'verify', 'confirm', 'account',
        'update', 'secure', 'banking', 'payment', 'wallet',
        'credential', 'password', 'auth', 'token'
    ]

    # Suspicious file extensions
    SUSPICIOUS_EXTENSIONS = [
        '.exe', '.scr', '.bat', '.cmd', '.pif',
        '.zip', '.rar', '.js', '.vbs'
    ]

    def __init__(
        self,
        legitimate_domains: Optional[Set[str]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize URL analyzer.

        Args:
            legitimate_domains: Set of known legitimate domains
            strict_mode: If True, more conservative in marking as safe
        """
        self.legitimate_domains = legitimate_domains or self.LEGITIMATE_DOMAINS
        self.strict_mode = strict_mode

    def analyze(self, email: EmailData) -> URLExplanation:
        """
        Analyze URLs in email.

        Args:
            email: Email data to analyze

        Returns:
            URLExplanation with analysis results
        """
        urls = email.urls
        reasons = []
        suspicious_urls = []
        safe_urls = []
        is_suspicious = False

        # Analyze each URL
        for url_obj in urls:
            url_analysis = self._analyze_single_url(url_obj)

            if url_analysis['is_suspicious']:
                is_suspicious = True
                suspicious_urls.append(url_analysis)
                reasons.append(url_analysis['reason'])
            else:
                safe_urls.append(url_obj.original)

        # If no URLs found, that's neutral
        if not urls:
            pass

        # Calculate confidence
        confidence = self._calculate_confidence(is_suspicious, len(suspicious_urls))

        return URLExplanation(
            is_suspicious=is_suspicious,
            confidence=confidence,
            reasons=reasons,
            suspicious_urls=suspicious_urls,
            safe_urls=safe_urls
        )

    def _analyze_single_url(self, url_obj: URL) -> dict:
        """
        Analyze a single URL.

        Returns:
            Dict with analysis results
        """
        is_suspicious = False
        reasons = []

        # Parse URL
        try:
            parsed = urlparse(url_obj.original)
        except Exception:
            return {
                'url': url_obj.original,
                'is_suspicious': True,
                'reason': 'Invalid URL format'
            }

        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # Check for HTTPS
        if not url_obj.has_https and parsed.scheme != 'https':
            is_suspicious = True
            reasons.append('Unencrypted connection (HTTP instead of HTTPS)')

        # Check for lookalike domain
        if check_lookalike_domain(domain, self.legitimate_domains):
            is_suspicious = True
            reasons.append('Domain mimics well-known company')

        # Check for suspicious TLD
        if any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS):
            is_suspicious = True
            reasons.append('Uses uncommon domain extension')

        # Check for URL shortener
        if self._is_url_shortener(domain):
            is_suspicious = True
            reasons.append('URL shortener hides actual destination')

        # Check for suspicious keywords in path
        suspicious_in_path = [kw for kw in self.SUSPICIOUS_KEYWORDS if kw in path]
        if suspicious_in_path:
            is_suspicious = True
            reasons.append(f'Suspicious keywords in path: {", ".join(suspicious_in_path)}')

        # Check for IP address instead of domain
        if self._is_ip_address(domain):
            is_suspicious = True
            reasons.append('Uses IP address instead of domain name')

        # Check for suspicious file extensions
        if any(path.endswith(ext) for ext in self.SUSPICIOUS_EXTENSIONS):
            is_suspicious = True
            reasons.append('Links to executable file')

        # Check for excessive subdomains
        if self._has_excessive_subdomains(domain):
            is_suspicious = True
            reasons.append('Excessive subdomains (suspicious structure)')

        # Check for suspicious parameters
        suspicious_params = self._check_suspicious_parameters(parsed)
        if suspicious_params:
            is_suspicious = True
            reasons.append(f'Suspicious URL parameters: {", ".join(suspicious_params)}')

        # Build reason string
        reason_str = '; '.join(reasons) if reasons else 'Suspicious pattern detected'

        return {
            'url': url_obj.original,
            'domain': domain,
            'is_suspicious': is_suspicious,
            'reason': reason_str
        }

    def _is_url_shortener(self, domain: str) -> bool:
        """Check if domain is a URL shortener."""
        return domain in self.URL_SHORTENERS

    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address."""
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return bool(re.match(ip_pattern, domain))

    def _has_excessive_subdomains(self, domain: str) -> bool:
        """Check for excessive subdomains."""
        parts = domain.split('.')
        return len(parts) > 4

    def _check_suspicious_parameters(self, parsed_url) -> List[str]:
        """Check for suspicious URL parameters."""
        suspicious = []

        try:
            params = parse_qs(parsed_url.query)

            # Check for suspicious parameter names
            param_names = [k.lower() for k in params.keys()]

            suspicious_params = ['token', 'auth', 'password', 'credential', 'session']
            found = [p for p in param_names if p in suspicious_params]

            if found:
                suspicious.extend(found)

        except Exception:
            pass

        return suspicious

    def _calculate_confidence(self, is_suspicious: bool, num_suspicious: int) -> float:
        """
        Calculate confidence score.

        Args:
            is_suspicious: Whether any URLs are suspicious
            num_suspicious: Number of suspicious URLs

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not is_suspicious:
            return 0.85

        # Base confidence increases with number of suspicious URLs
        base = 0.60
        increase = min(num_suspicious * 0.10, 0.30)

        return min(base + increase, 0.95)

    def analyze_multiple(self, emails: List[EmailData]) -> List[URLExplanation]:
        """
        Analyze URLs for multiple emails.

        Args:
            emails: List of emails to analyze

        Returns:
            List of URL explanations
        """
        return [self.analyze(email) for email in emails]
