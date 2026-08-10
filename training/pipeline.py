from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "starter-musicology-model"
    max_epochs: int = 3


def build_training_summary(config: TrainingConfig | None = None) -> dict[str, object]:
    resolved = config or TrainingConfig()
    return {
        "model_name": resolved.model_name,
        "max_epochs": resolved.max_epochs,
        "status": "configured",
    }
