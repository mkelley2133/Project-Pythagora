from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "starter-musicology-model"
    max_epochs: int = 3


def validate_config(config: TrainingConfig) -> None:
    if not config.model_name:
        raise ValueError("model_name must not be empty")
    if config.max_epochs < 1:
        raise ValueError("max_epochs must be >= 1")


def build_training_summary(config: TrainingConfig | None = None) -> dict[str, object]:
    resolved = config or TrainingConfig()
    validate_config(resolved)
    return {
        "model_name": resolved.model_name,
        "max_epochs": resolved.max_epochs,
        "status": "configured",
    }
