"""
Export utilities: ONNX export and LoRA adapter saving.
"""
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from ..models.base import BaseTransformerClassifier
from ..data.tokenizer import TokenizerWrapper


def export_to_onnx(
    model: BaseTransformerClassifier,
    tokenizer: TokenizerWrapper,
    output_path: str,
    example_input: Optional[Dict[str, torch.Tensor]] = None,
    opset_version: int = 14
) -> None:
    """
    Export model to ONNX format for deployment.

    Args:
        model: PyTorch model to export
        tokenizer: Tokenizer instance
        output_path: Path to save ONNX model
        example_input: Example input for tracing (created if None)
        opset_version: ONNX opset version
    """
    model.eval()
    device = next(model.parameters()).device

    # Create example input if not provided
    if example_input is None:
        example_input = {
            'input_ids': torch.randint(0, 30522, (1, 512), device=device),
            'attention_mask': torch.ones(1, 512, device=device)
        }

    # Move to device
    example_input = {k: v.to(device) for k, v in example_input.items()}

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX with error handling
    try:
        torch.onnx.export(
            model,
            (example_input['input_ids'], example_input['attention_mask']),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"💾 Model exported to ONNX: {output_path}")
    except RuntimeError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"ONNX export failed: {e}")
        logger.warning("Falling back to TorchScript export")

        # Fallback to TorchScript
        torchscript_path = str(Path(output_path).with_suffix('.pt'))
        try:
            # Use tracing for TorchScript export
            model.eval()
            traced_model = torch.jit.trace(
                model,
                (example_input['input_ids'], example_input['attention_mask'])
            )
            traced_model.save(torchscript_path)
            print(f"💾 Model exported to TorchScript: {torchscript_path}")
            print(f"⚠️  Note: TorchScript format may have compatibility limitations")
        except Exception as ts_error:
            logger.error(f"TorchScript export also failed: {ts_error}")
            raise RuntimeError(f"Both ONNX and TorchScript export failed. ONNX: {e}, TorchScript: {ts_error}")

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified successfully")

    # Save tokenizer info
    tokenizer_config_path = Path(output_path).parent / "tokenizer_config.json"
    import json
    with open(tokenizer_config_path, 'w') as f:
        json.dump({
            'model_name': tokenizer.model_name,
            'max_length': 512,
            'special_tokens': ["[SUBJECT]", "[BODY]", "[URL]", "[SENDER]", "[TRUNCATED]"]
        }, f)
    print(f"💾 Tokenizer config saved: {tokenizer_config_path}")


def test_onnx_model(
    onnx_path: str,
    example_input: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Test ONNX model with ONNXRuntime.

    Args:
        onnx_path: Path to ONNX model
        example_input: Example input dict with numpy arrays

    Returns:
        Model outputs
    """
    # Create ONNX Runtime session
    sess = ort.InferenceSession(onnx_path)

    # Run inference
    outputs = sess.run(
        None,
        {
            'input_ids': example_input['input_ids'],
            'attention_mask': example_input['attention_mask']
        }
    )

    print("✅ ONNX Runtime inference successful")
    print(f"   Output shape: {outputs[0].shape}")

    return {'logits': outputs[0]}


def save_lora_adapters(
    model,
    save_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save LoRA adapter weights separately for federated aggregation.

    Args:
        model: Model with LoRA adapters
        save_path: Directory to save adapters
        metadata: Optional metadata to save
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if model has save_adapters method (LoRABERTClassifier)
    if hasattr(model, 'save_adapters'):
        model.save_adapters(str(save_dir))
    else:
        # Manually extract LoRA parameters
        lora_state_dict = {}
        for name, param in model.named_parameters():
            if 'lora' in name and param.requires_grad:
                lora_state_dict[name] = param.data.cpu()

        torch.save(lora_state_dict, save_dir / 'lora_adapters.pt')
        print(f"💾 LoRA adapters saved to {save_dir}")

    # Save metadata
    if metadata:
        import json
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        print(f"💾 Metadata saved to {save_dir}/metadata.json")


def load_lora_adapters(
    model,
    load_path: str
) -> None:
    """
    Load LoRA adapter weights.

    Args:
        model: Model to load adapters into
        load_path: Directory with adapter weights
    """
    load_dir = Path(load_path)

    # Check if model has load_adapters method
    if hasattr(model, 'load_adapters'):
        model.load_adapters(str(load_dir))
    else:
        # Manually load LoRA parameters
        lora_state_dict = torch.load(load_dir / 'lora_adapters.pt', weights_only=True)

        model_state_dict = model.state_dict()
        for name, param in lora_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)

        print(f"📥 LoRA adapters loaded from {load_dir}")


def compare_model_sizes(
    pytorch_model: BaseTransformerClassifier,
    onnx_path: str
) -> Dict[str, float]:
    """
    Compare sizes of PyTorch and ONNX models.

    Args:
        pytorch_model: PyTorch model
        onnx_path: Path to ONNX model

    Returns:
        Dictionary with size information in MB
    """
    from ..utils.memory import get_model_size

    # PyTorch model size
    torch_size = get_model_size(pytorch_model)

    # ONNX model size
    onnx_file = Path(onnx_path)
    onnx_size_mb = onnx_file.stat().st_size / 1024**2

    return {
        'pytorch_total_mb': torch_size['total_size_mb'],
        'pytorch_trainable_mb': torch_size['trainable_param_size_mb'],
        'onnx_mb': onnx_size_mb,
        'compression_ratio': torch_size['total_size_mb'] / onnx_size_mb
    }


def create_deployment_package(
    model: BaseTransformerClassifier,
    tokenizer: TokenizerWrapper,
    output_dir: str,
    export_onnx: bool = True
) -> None:
    """
    Create a complete deployment package.

    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        output_dir: Output directory
        export_onnx: Whether to export ONNX model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'num_labels': model.num_labels
    }, output_path / 'pytorch_model.bin')
    print(f"💾 PyTorch model saved to {output_path}/pytorch_model.bin")

    # Save tokenizer
    tokenizer.save_pretrained(str(output_path / 'tokenizer'))

    # Export ONNX
    if export_onnx:
        export_to_onnx(
            model,
            tokenizer,
            str(output_path / 'model.onnx')
        )

    # Save inference script
    inference_script = """
import torch
from transformers import AutoTokenizer

def load_model(model_path):
    model = torch.load(f"{model_path}/pytorch_model.bin")
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs
"""
    with open(output_path / 'inference.py', 'w') as f:
        f.write(inference_script)
    print(f"💾 Inference script saved to {output_path}/inference.py")

    print(f"\n✅ Deployment package created at {output_path}")
