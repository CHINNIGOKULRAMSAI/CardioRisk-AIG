from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str
    expected_balanced_accuracy: float
