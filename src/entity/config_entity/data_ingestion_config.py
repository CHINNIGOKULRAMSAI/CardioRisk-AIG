from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    database_name: str
    collection_name: str
    feature_store_file_path: str
    training_file_path: str
    testing_file_path: str
    train_test_split_ratio: float = 0.2
