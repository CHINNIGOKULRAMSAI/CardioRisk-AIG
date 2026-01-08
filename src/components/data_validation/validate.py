import os
import sys
import yaml
import json
import pandas as pd
from typing import Dict

from src.exception.exception import CVDException
from src.logger.logging import logging
from src.entity.config_entity.data_validation_config import DataValidationConfig
from src.entity.artifact_entity.data_validation_artifact import DataValidationArtifact


class DataValidation:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: DataValidationConfig
    ):
        try:
            self.dataframe = dataframe
            self.config = config
        except Exception as e:
            raise CVDException(e, sys)

    def _read_schema(self) -> Dict:
        try:
            with open(self.config.schema_file_path, "r") as f:
                schema = yaml.safe_load(f)
            return schema
        except Exception as e:
            raise CVDException(e, sys)

    def _validate_schema(self) -> Dict:
        logging.info("Starting schema validation")

        schema = self._read_schema()
        report = {
            "missing_columns": [],
            "invalid_columns": []
        }

        for column, rules in schema["columns"].items():

            if column not in self.dataframe.columns:
                report["missing_columns"].append(column)
                continue

            col_data = self.dataframe[column]

            if "allowed" in rules:
                invalid = ~col_data.isin(rules["allowed"])
                if invalid.any():
                    report["invalid_columns"].append(
                        f"{column}: invalid categorical values"
                    )

            if "min" in rules:
                if (col_data < rules["min"]).any():
                    report["invalid_columns"].append(
                        f"{column}: values below min"
                    )

            if "max" in rules:
                if (col_data > rules["max"]).any():
                    report["invalid_columns"].append(
                        f"{column}: values above max"
                    )

        logging.info("Schema validation completed")
        return report

    def _check_missing_values(self) -> Dict:
        logging.info("Checking missing values")
        return self.dataframe.isnull().sum().to_dict()

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            schema_report = self._validate_schema()
            missing_report = self._check_missing_values()

            validation_status = (
                len(schema_report["missing_columns"]) == 0 and
                len(schema_report["invalid_columns"]) == 0
            )

            report = {
                "schema_report": schema_report,
                "missing_values": missing_report,
                "validation_status": validation_status
            }

            os.makedirs(
                os.path.dirname(self.config.validation_report_path),
                exist_ok=True
            )

            with open(self.config.validation_report_path, "w") as f:
                json.dump(report, f, indent=4)

            logging.info(
                f"Validation report saved at "
                f"{self.config.validation_report_path}"
            )

            return DataValidationArtifact(
                validation_status=validation_status,
                report_file_path=self.config.validation_report_path
            )

        except Exception as e:
            raise CVDException(e, sys)
