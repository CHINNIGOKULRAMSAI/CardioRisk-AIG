from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    schema_file_path: str
    validation_report_path: str
