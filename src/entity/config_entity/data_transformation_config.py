from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    transformed_train_path: str
    transformed_test_path: str
    preprocessor_object_path: str
    target_column: str
