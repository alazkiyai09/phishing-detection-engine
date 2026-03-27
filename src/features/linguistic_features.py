"""Linguistic feature extraction for phishing detection.

Extracts NLP-based features from email text:
- Spelling error rate
- Grammar score (simplified)
- Formal vs informal language
- Sentiment analysis
- Text complexity metrics
- Punctuation patterns
"""

import re
import time
from typing import List

import numpy as np
import pandas as pd
import textstat

from .base import BaseExtractor


class LinguisticFeatureExtractor(BaseExtractor):
    """Extract linguistic features for phishing detection.

    Features extracted:
        - spelling_error_rate: Misspelled words ratio (normalized)
        - grammar_score_proxy: Grammar quality score (normalized)
        - formality_score: Language formality level (normalized)
        - reading_ease_score: Text readability (normalized)
        - sentence_count: Number of sentences (normalized)
        - avg_sentence_length: Average sentence length (normalized)
        - exclamation_mark_count: Exclamation usage (normalized)
        - question_mark_count: Question usage (normalized)
        - all_caps_ratio: All-caps words ratio (normalized)
        - punctuation_ratio: Punctuation density (normalized)
    """

    # Max values for normalization
    MAX_SENTENCES = 50
    MAX_SENTENCE_LENGTH = 50
    MAX_PUNCTUATION_COUNT = 20

    # Commonly misspelled words (for phishing-specific patterns)
    COMMON_MISPELLINGS = {
        "verfiy": "verify",
        "securty": "security",
        "acount": "account",
        "infromation": "information",
        "confimration": "confirmation",
        "passowrd": "password",
        "loggin": "login",
        "bankk": "bank",
        "credntials": "credentials",
        "autorization": "authorization",
    }

    def __init__(self) -> None:
        """Initialize linguistic feature extractor."""
        super().__init__()
        self.feature_names = [
            "spelling_error_rate",
            "grammar_score_proxy",
            "formality_score",
            "reading_ease_score",
            "sentence_count",
            "avg_sentence_length",
            "exclamation_mark_count",
            "question_mark_count",
            "all_caps_ratio",
            "punctuation_ratio",
        ]

        # Regex patterns
        self.word_pattern = re.compile(r"\b[a-zA-Z]+\b")
        self.sentence_pattern = re.compile(r"[.!?]+")
        self.exclamation_pattern = re.compile(r"!")
        self.question_pattern = re.compile(r"\?")
        self.all_caps_pattern = re.compile(r"\b[A-Z]{2,}\b")
        self.punctuation_pattern = re.compile(r"[^\w\s]")

    def fit(self, emails: pd.DataFrame) -> "LinguisticFeatureExtractor":
        """Fit the linguistic extractor.

        Stateless extractor - validates input structure only.

        Args:
            emails: DataFrame with 'body' and 'subject' columns.

        Returns:
            self: Fitted extractor.
        """
        self._validate_input(emails)
        self._is_fitted = True
        return self

    def transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Transform emails into linguistic features.

        Args:
            emails: DataFrame with 'body' and 'subject' columns.

        Returns:
            DataFrame with linguistic features (n_emails, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("LinguisticFeatureExtractor must be fitted before transform")

        results = []

        for idx, row in emails.iterrows():
            start_time = time.time()

            body = str(row.get("body", ""))
            subject = str(row.get("subject", ""))

            # Combine subject and body
            text = f"{subject} {body}"

            features = self._extract_linguistic_features(text)
            results.append(features)

            self.extraction_times.append((time.time() - start_time) * 1000)  # ms

        return pd.DataFrame(results, columns=self.feature_names)

    def _extract_linguistic_features(self, text: str) -> dict[str, float]:
        """Extract all linguistic features.

        Args:
            text: Email text (subject + body).

        Returns:
            Dictionary of feature names to values in [0, 1].
        """
        if not text or len(text.strip()) < 10:
            return {name: 0.0 for name in self.feature_names}

        words = self._extract_words(text)
        if not words:
            return {name: 0.0 for name in self.feature_names}

        features = {
            "spelling_error_rate": self._calculate_spelling_error_rate(words),
            "grammar_score_proxy": self._calculate_grammar_score(text),
            "formality_score": self._calculate_formality_score(text),
            "reading_ease_score": self._calculate_reading_ease(text),
            "sentence_count": self._count_sentences(text),
            "avg_sentence_length": self._avg_sentence_length(text),
            "exclamation_mark_count": self._count_exclamations(text),
            "question_mark_count": self._count_questions(text),
            "all_caps_ratio": self._all_caps_ratio(words),
            "punctuation_ratio": self._punctuation_ratio(text),
        }

        return features

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text.

        Args:
            text: Input text.

        Returns:
            List of words.
        """
        return self.word_pattern.findall(text)

    def _calculate_spelling_error_rate(self, words: List[str]) -> float:
        """Calculate spelling error rate.

        Checks against common misspellings dictionary.

        Args:
            words: List of words.

        Returns:
            Normalized error rate in [0, 1].
        """
        if not words:
            return 0.0

        error_count = 0
        total_count = 0

        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 3:  # Only check words longer than 3 chars
                total_count += 1
                if word_lower in self.COMMON_MISPELLINGS:
                    error_count += 1

        if total_count == 0:
            return 0.0

        # Normalize: more errors = higher score (more suspicious)
        # Max expected error rate ~20%
        error_rate = error_count / total_count
        return min(1.0, error_rate / 0.2)

    def _calculate_grammar_score(self, text: str) -> float:
        """Calculate grammar quality score (proxy).

        Uses heuristics as proxy for grammar quality:
- Proper sentence capitalization
- No double spaces
- No excessive punctuation

        Args:
            text: Input text.

        Returns:
            Normalized grammar score in [0, 1].
        """
        if not text:
            return 0.0

        grammar_issues = 0
        total_checks = 0

        # Check for double spaces
        if "  " in text:
            grammar_issues += 1
        total_checks += 1

        # Check for sentences starting with lowercase
        sentences = self.sentence_pattern.split(text)
        valid_sentences = [s for s in sentences if len(s.strip()) > 0]
        if valid_sentences:
            lowercase_starts = sum(
                1 for s in valid_sentences if s.strip() and s.strip()[0].islower()
            )
            if lowercase_starts > len(valid_sentences) / 2:
                grammar_issues += 1
        total_checks += 1

        # Check for multiple punctuation marks (??? or !!!)
        if re.search(r"[!?]{2,}", text):
            grammar_issues += 1
        total_checks += 1

        if total_checks == 0:
            return 0.0

        # More issues = higher score (more suspicious)
        return min(1.0, grammar_issues / total_checks)

    def _calculate_formality_score(self, text: str) -> float:
        """Calculate language formality score.

        Informal language can indicate phishing (unprofessional).

        Args:
            text: Input text.

        Returns:
            Normalized formality score in [0, 1].
            Higher = more informal (more suspicious).
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        informal_indicators = [
            "hey",
            "hi",
            "hello",
            "dear friend",
            "dear customer",
            "urgent",
            "act now",
            "don't wait",
            "immediately",
            "asap",
        ]

        formal_indicators = [
            "dear mr",
            "dear mrs",
            "dear ms",
            "sincerely",
            "regards",
            "respectfully",
        ]

        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)

        # Score: informal - formal (normalized to [0, 1])
        # Higher = more informal (suspicious)
        score = (informal_count - formal_count + 5) / 10  # Shift to [0, 1]
        return max(0.0, min(1.0, score))

    def _calculate_reading_ease(self, text: str) -> float:
        """Calculate text readability score.

        Uses textstat Flesch Reading Ease.

        Args:
            text: Input text.

        Returns:
            Normalized readability score in [0, 1].
        """
        if not text or len(text) < 50:
            return 0.0

        try:
            # Flesch Reading Ease: 0-100 (higher = easier)
            # Invert: harder to read = more suspicious
            ease = textstat.flesch_reading_ease(text)

            # Normalize to [0, 1]: Lower ease = higher score
            # Range: 0 (very easy) to 100 (very hard)
            # We want: easy = 0, hard = 1
            return max(0.0, min(1.0, (100 - ease) / 100))

        except Exception:
            return 0.0

    def _count_sentences(self, text: str) -> float:
        """Count number of sentences.

        Args:
            text: Input text.

        Returns:
            Normalized count in [0, 1].
        """
        if not text:
            return 0.0

        sentences = [s for s in self.sentence_pattern.split(text) if len(s.strip()) > 0]
        count = len(sentences)

        return min(1.0, count / self.MAX_SENTENCES)

    def _avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words.

        Args:
            text: Input text.

        Returns:
            Normalized average length in [0, 1].
        """
        if not text:
            return 0.0

        sentences = [s for s in self.sentence_pattern.split(text) if len(s.strip()) > 0]

        if not sentences:
            return 0.0

        word_counts = [len(self._extract_words(s)) for s in sentences]
        if not word_counts:
            return 0.0

        avg_length = np.mean(word_counts)
        return min(1.0, avg_length / self.MAX_SENTENCE_LENGTH)

    def _count_exclamations(self, text: str) -> float:
        """Count exclamation marks.

        Excessive exclamations indicate urgency pressure.

        Args:
            text: Input text.

        Returns:
            Normalized count in [0, 1].
        """
        if not text:
            return 0.0

        count = len(self.exclamation_pattern.findall(text))
        return min(1.0, count / 5)  # Max 5 for normalization

    def _count_questions(self, text: str) -> float:
        """Count question marks.

        Args:
            text: Input text.

        Returns:
            Normalized count in [0, 1].
        """
        if not text:
            return 0.0

        count = len(self.question_pattern.findall(text))
        return min(1.0, count / 5)  # Max 5 for normalization

    def _all_caps_ratio(self, words: List[str]) -> float:
        """Calculate ratio of all-caps words.

        ALL CAPS is often used for emphasis in phishing.

        Args:
            words: List of words.

        Returns:
            Normalized ratio in [0, 1].
        """
        if not words:
            return 0.0

        all_caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)
        ratio = all_caps_count / len(words)

        # Normalize: max expected ratio ~20%
        return min(1.0, ratio / 0.2)

    def _punctuation_ratio(self, text: str) -> float:
        """Calculate punctuation density.

        Args:
            text: Input text.

        Returns:
            Normalized ratio in [0, 1].
        """
        if not text:
            return 0.0

        text_len = len(text)
        if text_len == 0:
            return 0.0

        punctuation_count = len(self.punctuation_pattern.findall(text))
        ratio = punctuation_count / text_len

        # Normalize: max expected ratio ~20%
        return min(1.0, ratio / 0.2)
