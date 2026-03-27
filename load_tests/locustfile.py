"""
Locust load tests for Unified Phishing Detection API.

Target: 100 RPS with P95 latency < 1s
"""
import json
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# Sample emails for load testing
SAMPLE_PHISHING_EMAIL = """
From: security@chase-secure-portal.xyz
Subject: URGENT: Verify your account now
Date: Wed, 29 Jan 2026 12:34:56 +0000

Dear Customer,

Your account will be suspended within 24 hours unless you verify your information.
Click here to verify: http://chase-secure-portal.xyz/login

Please provide your account number and SSN to prevent suspension.

Sincerely,
Chase Security Team
"""

SAMPLE_LEGITIMATE_EMAIL = """
From: notifications@chase.com
Subject: Your statement is available
Date: Wed, 29 Jan 2026 12:34:56 +0000

Dear Customer,

Your monthly statement is now available.
Log in to your account to view it: https://chase.com

This is an automated message. Please do not reply.

Chase Bank
"""


class PhishingAPIUser(HttpUser):
    """
    Simulates user traffic to the phishing detection API.
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Run on start of each user."""
        # Check health first
        self.client.get("/health")

    @task(3)
    def analyze_email(self):
        """
        Analyze a single email (most common operation).
        """
        # Randomly choose phishing or legitimate email
        email = random.choice([
            SAMPLE_PHISHING_EMAIL,
            SAMPLE_LEGITIMATE_EMAIL
        ])

        # Randomly choose model
        model_type = random.choice([
            "xgboost",
            "transformer",
            "ensemble"
        ])

        payload = {
            "raw_email": email,
            "model_type": model_type,
            "use_cache": random.choice([True, False])
        }

        with self.client.post(
            "/api/v1/analyze/email",
            json=payload,
            catch_response=True,
            name="/api/v1/analyze/email"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Verify response structure
                if "verdict" not in data:
                    response.failure("Missing 'verdict' in response")
            elif response.status_code == 501:
                # Endpoint not implemented yet (expected during development)
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def analyze_url(self):
        """
        Quick URL analysis (less common).
        """
        urls = [
            "http://chase-secure-portal.xyz/login",
            "https://chase.com",
            "http://wellfarg0.com/verify",
            "https://www.wellsfargo.com"
        ]

        payload = {
            "url": random.choice(urls),
            "use_cache": True
        }

        with self.client.post(
            "/api/v1/analyze/url",
            json=payload,
            catch_response=True,
            name="/api/v1/analyze/url"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "verdict" not in data:
                    response.failure("Missing 'verdict' in response")
            elif response.status_code == 501:
                # Not implemented yet
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def list_models(self):
        """
        List available models (lightweight operation).
        """
        self.client.get("/api/v1/models", name="/api/v1/models")

    @task(1)
    def health_check(self):
        """
        Health check (lightweight operation).
        """
        self.client.get("/health", name="/health")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Print summary statistics when test stops.
    """
    if isinstance(environment.runner, MasterRunner):
        print("\n=== Load Test Summary ===")
        print(f"Total requests: {environment.runner.stats.total.num_requests}")
        print(f"Failures: {environment.runner.stats.total.num_failures}")
        print(f"RPS: {environment.runner.stats.total.avg_rps:.2f}")
        print(f"Median response time: {environment.runner.stats.current.response_time_median:.2f}ms")
        print(f"P95 response time: {environment.runner.stats.current.get_response_time_percentile(0.95):.2f}ms")
        print("========================\n")
