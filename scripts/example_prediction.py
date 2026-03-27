#!/usr/bin/env python3
"""
Quick prediction example - test a trained model on sample emails.
Run this after training completes to verify your model works.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def predict_with_example():
    """Run prediction on example emails without loading trained model."""

    print("=" * 70)
    print("Phishing Email Detection - Quick Prediction Demo")
    print("=" * 70)
    print()

    # Check if trained model exists
    checkpoint_path = Path("checkpoints/bert/best_model.pt")

    if not checkpoint_path.exists():
        print("⚠️  No trained model found yet!")
        print()
        print("To train a model, run:")
        print("  bash train_all_models.sh")
        print()
        print("Or train a single model:")
        print("  python3 train.py --model bert --data data/processed/phishing_emails_2k.csv --epochs 1 --batch-size 8 --no-wandb")
        print()
        return

    print("✅ Found trained model at checkpoints/bert/best_model.pt")
    print()

    # Import here to avoid errors if model doesn't exist
    try:
        import torch
        from src.inference.predictor import Predictor
        from src.data.tokenizer import TokenizerWrapper
        from src.models.factory import create_model

        # Load model
        print("Loading BERT model...")
        model = create_model(model_type='bert', num_labels=2)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        tokenizer = TokenizerWrapper(model_name='bert-base-uncased')
        predictor = Predictor(model=model, tokenizer=tokenizer, device='cpu')

        print("✅ Model loaded successfully!")
        print()

        # Test emails
        test_emails = [
            {
                'name': 'Obvious Phishing',
                'text': 'URGENT: Your account will be suspended! Click here immediately to verify: http://verify-bank-login.com'
            },
            {
                'name': 'Legitimate Work Email',
                'text': 'Hi team, reminder about our weekly meeting tomorrow at 2pm. Please come prepared with your updates.'
            },
            {
                'name': 'Suspicious Email',
                'text': 'Dear customer, we detected suspicious activity. Click here to verify your account within 24 hours or it will be permanently suspended.'
            },
            {
                'name': 'Legitimate Newsletter',
                'text': 'Your weekly newsletter is here! Check out the latest articles and updates from our team.'
            }
        ]

        print("-" * 70)
        print("Running predictions on test emails...")
        print("-" * 70)
        print()

        for email in test_emails:
            result = predictor.predict(email['text'])

            print(f"📧 {email['name']}")
            print(f"   Text: {email['text'][:80]}...")
            print(f"   Prediction: {result['label_text']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Probabilities: Legitimate={result['probabilities'][0]:.2%}, Phishing={result['probabilities'][1]:.2%}")

            # Visual indicator
            if result['label'] == 1:
                print(f"   🚨 FLAGGED AS PHISHING")
            else:
                print(f"   ✅ LEGITIMATE")

            print()

        print("=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        print()
        print("To use this model in your own code:")
        print()
        print("  from src.inference.predictor import Predictor")
        print("  from src.models.factory import create_model")
        print("  import torch")
        print()
        print("  model = create_model(model_type='bert', num_labels=2)")
        print("  checkpoint = torch.load('checkpoints/bert/best_model.pt')")
        print("  model.load_state_dict(checkpoint['model_state_dict'])")
        print("  model.eval()")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    predict_with_example()
