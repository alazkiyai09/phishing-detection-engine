from src.models.transformer.lora_adapter import LoRAConfig


def build_training_plan(model_name: str = "distilbert") -> dict:
    return {"model": model_name, "adapter": LoRAConfig().__dict__, "status": "planned"}
