#!/bin/bash
# Train all models and generate comparison report

echo "========================================"
echo "Training All Transformer Models"
echo "========================================"

# Train BERT
echo ""
echo "1. Training BERT..."
bash experiments/scripts/train_bert.sh

# Train RoBERTa
echo ""
echo "2. Training RoBERTa..."
bash experiments/scripts/train_roberta.sh

# Train DistilBERT
echo ""
echo "3. Training DistilBERT..."
bash experiments/scripts/train_distilbert.sh

# Train LoRA-BERT
echo ""
echo "4. Training LoRA-BERT..."
bash experiments/scripts/train_lora.sh

echo ""
echo "========================================"
echo "All models trained!"
echo "========================================"
echo ""
echo "Generating comparison report..."
python3 -c "
import torch
import pandas as pd
from pathlib import Path

models = ['bert', 'roberta', 'distilbert', 'lora_bert']
results = []

for model in models:
    checkpoint_dir = Path(f'checkpoints/{model}')
    if checkpoint_dir.exists():
        # Try to load best model
        best_model_path = checkpoint_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location='cpu')
            val_metrics = checkpoint.get('val_metrics', {})
            results.append({
                'Model': model.upper(),
                'AUPRC': f\"{val_metrics.get('auprc', 0):.4f}\",
                'AUROC': f\"{val_metrics.get('auroc', 0):.4f}\",
                'Accuracy': f\"{val_metrics.get('accuracy', 0):.4f}\",
                'Precision': f\"{val_metrics.get('precision', 0):.4f}\",
                'Recall': f\"{val_metrics.get('recall', 0):.4f}\",
                'F1': f\"{val_metrics.get('f1', 0):.4f}\",
            })

df = pd.DataFrame(results)
print('\\n' + '='*80)
print('MODEL COMPARISON TABLE')
print('='*80)
print(df.to_string(index=False))
print('='*80)
df.to_csv('model_comparison.csv', index=False)
print('\\n💾 Saved to model_comparison.csv')
"

echo ""
echo "✅ Comparison complete!"
