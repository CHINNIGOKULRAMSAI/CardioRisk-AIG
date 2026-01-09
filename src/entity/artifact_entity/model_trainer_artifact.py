from dataclasses import dataclass

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    best_model_name: str
    best_accuracy: float
