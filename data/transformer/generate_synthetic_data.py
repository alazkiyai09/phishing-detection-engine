#!/usr/bin/env python3
"""
Generate synthetic phishing and legitimate emails for testing and development.
"""
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict
import string


class SyntheticEmailGenerator:
    """Generate realistic synthetic emails for phishing detection."""

    # Phishing templates with urgency and deception tactics
    PHISHING_TEMPLATES = [
        {
            'subject': 'URGENT: Verify your account immediately',
            'body_templates': [
                "Dear Customer, Your account has been suspended due to suspicious activity. Click here to verify: {url}",
                "Immediate action required! Your account will be closed in 24 hours. Verify now: {url}",
                "Security Alert! Unusual login detected. Confirm your identity: {url}",
                "Your account has been compromised. Reset your password now: {url}"
            ],
            'senders': ['security@bank-security.com', 'support@verify-account.com', 'alert@secure-login.net'],
            'urls': ['http://bank-secure.com/verify', 'http://account-verify.net/login', 'http://secure-login.info/confirm']
        },
        {
            'subject': 'Congratulations! You\'ve won $1,000,000',
            'body_templates': [
                "Congratulations! You've been selected as a winner. Claim your prize now: {url}",
                "URGENT: You have 48 hours to claim your $1,000,000 prize. Click here: {url}",
                "Amazing news! You're our lucky winner. Transfer fee required to claim: {url}"
            ],
            'senders': ['claims@lottery-winner.com', 'prize@international-lottery.net', 'winner@mega-prize.com'],
            'urls': ['http://claim-prize.net/winner', 'http://lottery-winner.com/claim', 'http://mega-prize.info/collect']
        },
        {
            'subject': 'Invoice Overdue - Immediate Payment Required',
            'body_templates': [
                "URGENT: Your invoice is overdue. Pay now to avoid legal action: {url}",
                "Final Notice: Payment required within 24 hours or account will be suspended: {url}",
                "Overdue invoice: Immediate payment required to avoid service interruption: {url}"
            ],
            'senders': ['billing@invoice-collection.com', 'accounts@payment-urgent.net', 'finance@overdue-invoice.com'],
            'urls': ['http://pay-invoice.net/overdue', 'http://billing-urgent.com/pay', 'http://invoice-payment.info/now']
        },
        {
            'subject': 'Job Opportunity - Work From Home - $5000/week',
            'body_templates': [
                "Make $5000/week working from home! No experience needed. Sign up now: {url}",
                "URGENT: Limited positions available. Earn $5000 weekly from home: {url}",
                "Work from home opportunity! $5000/week part-time. Apply now: {url}"
            ],
            'senders': ['jobs@work-from-home.com', 'careers@easy-money.net', 'hiring@remote-work.info'],
            'urls': ['http://work-from-home.net/apply', 'http://easy-money.jobs/signup', 'http://remote-work.info/join']
        }
    ]

    # Legitimate email templates
    LEGITIMATE_TEMPLATES = [
        {
            'subject': 'Weekly Team Update - Project Status',
            'body_templates': [
                "Hi Team, Please find attached the weekly project status report. Let's discuss in tomorrow's meeting at 10 AM. Best regards, {sender_name}",
                "Hello everyone, Here's the update on our current projects. Please review before our standup. Thanks, {sender_name}",
                "Team, Attached is the sprint summary. Please review and let me know if you have questions. Best, {sender_name}"
            ],
            'senders': ['john.smith@company.com', 'sarah.johnson@techcorp.com', 'mike.williams@enterprise.net'],
            'sender_names': ['John Smith', 'Sarah Johnson', 'Mike Williams']
        },
        {
            'subject': 'Meeting Reminder: Q4 Planning Session',
            'body_templates': [
                "Hi, This is a reminder about our Q4 planning session tomorrow at 2 PM in Conference Room B. Please bring your team's roadmap. Thanks, {sender_name}",
                "Hello, Just a reminder about the planning meeting tomorrow. See you there! Best, {sender_name}",
                "Team, Don't forget tomorrow's planning session at 2 PM. Conference Room B. Regards, {sender_name}"
            ],
            'senders': ['admin@company.com', 'operations@techcorp.com', 'planning@enterprise.net'],
            'sender_names': ['Admin Team', 'Operations', 'Planning Committee']
        },
        {
            'subject': 'Your Order #12345 Has Shipped',
            'body_templates': [
                "Hi, Your order #12345 has shipped and should arrive by Friday. Track your package with the carrier. Thanks, {sender_name}",
                "Hello, Good news! Your order has shipped and will arrive on Friday. You can track it online. Best, {sender_name}",
                "Your order is on its way! Expected delivery: Friday. Tracking number included in email. Regards, {sender_name}"
            ],
            'senders': ['shipping@retailer.com', 'orders@onlinestore.net', 'fulfillment@ecommerce.com'],
            'sender_names': ['Shipping Team', 'Order Fulfillment', 'Customer Service']
        },
        {
            'subject': 'Newsletter: Industry Insights for October',
            'body_templates': [
                "Hi, Here are this week's top industry insights and trends. Read more on our blog. Best regards, {sender_name}",
                "Hello, Check out the latest industry news and analysis in this week's newsletter. Thanks, {sender_name}",
                "This week in tech: Major developments and insights. Full story available on our website. Regards, {sender_name}"
            ],
            'senders': ['newsletter@industrynews.com', 'updates@techinsights.net', 'weekly@digest.com'],
            'sender_names': ['Editorial Team', 'News Desk', 'Content Team']
        }
    ]

    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        random.seed(seed)

    def generate_phishing_email(self) -> Dict[str, str]:
        """Generate a synthetic phishing email."""
        template = random.choice(self.PHISHING_TEMPLATES)

        subject = template['subject']
        body = random.choice(template['body_templates'])
        sender = random.choice(template['senders'])
        url = random.choice(template['urls'])

        # Format body with URL
        body = body.format(url=url)

        # Add some random variations
        if random.random() > 0.5:
            urgency_words = ['URGENT:', 'IMMEDIATE ACTION REQUIRED:', 'FINAL NOTICE:', 'LAST CHANCE:']
            subject = f"{random.choice(urgency_words)} {subject}"

        return {
            'subject': subject,
            'body': body,
            'sender': sender,
            'url': url,
            'label': 1,
            'label_text': 'Phishing Email'
        }

    def generate_legitimate_email(self) -> Dict[str, str]:
        """Generate a synthetic legitimate email."""
        template = random.choice(self.LEGITIMATE_TEMPLATES)

        subject = template['subject']
        body = random.choice(template['body_templates'])
        sender = random.choice(template['senders'])
        sender_name = random.choice(template.get('sender_names', ['Team']))

        # Format body with sender name
        body = body.format(sender_name=sender_name)

        return {
            'subject': subject,
            'body': body,
            'sender': sender,
            'url': '',
            'label': 0,
            'label_text': 'Safe Email'
        }

    def generate_dataset(
        self,
        n_samples: int = 1000,
        phishing_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate a synthetic dataset.

        Args:
            n_samples: Total number of samples to generate
            phishing_ratio: Proportion of phishing emails (0.0 to 1.0)

        Returns:
            DataFrame with synthetic emails
        """
        n_phishing = int(n_samples * phishing_ratio)
        n_legitimate = n_samples - n_phishing

        emails = []

        # Generate phishing emails
        for _ in range(n_phishing):
            emails.append(self.generate_phishing_email())

        # Generate legitimate emails
        for _ in range(n_legitimate):
            emails.append(self.generate_legitimate_email())

        # Shuffle
        random.shuffle(emails)

        df = pd.DataFrame(emails)

        # Create combined text field (subject + body)
        df['text'] = df['subject'] + ' ' + df['body']

        return df


def main():
    """Generate and save synthetic datasets."""
    generator = SyntheticEmailGenerator(seed=42)

    # Generate full synthetic dataset (for development/testing)
    print("📧 Generating synthetic phishing email dataset...")
    df = generator.generate_dataset(n_samples=1000, phishing_ratio=0.5)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    output_file = output_dir / "synthetic_phishing_emails.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Generated {len(df)} synthetic emails")
    print(f"   Phishing: {(df['label'] == 1).sum()}")
    print(f"   Legitimate: {(df['label'] == 0).sum()}")
    print(f"   Saved to: {output_file}")

    # Generate small test dataset (for unit tests)
    test_df = generator.generate_dataset(n_samples=100, phishing_ratio=0.5)
    test_file = output_dir / "test_phishing_emails.csv"
    test_df.to_csv(test_file, index=False)

    print(f"\n✅ Generated {len(test_df)} test emails")
    print(f"   Saved to: {test_file}")

    # Show samples
    print("\n📧 Sample Phishing Email:")
    phishing_sample = df[df['label'] == 1].iloc[0]
    print(f"   Subject: {phishing_sample['subject']}")
    print(f"   Body: {phishing_sample['body'][:100]}...")

    print("\n📧 Sample Legitimate Email:")
    legit_sample = df[df['label'] == 0].iloc[0]
    print(f"   Subject: {legit_sample['subject']}")
    print(f"   Body: {legit_sample['body'][:100]}...")


if __name__ == "__main__":
    main()
