from dataclasses import dataclass

@dataclass
class DataValidationArtifact:
    validation_status: bool
    report_file_path: str
