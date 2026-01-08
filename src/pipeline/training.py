from src.components.data_ingestion.ingest import DataIngestion
from src.components.data_validation.validate import DataValidation
from src.entity.config_entity.data_ingestion_config import DataIngestionConfig
from src.entity.config_entity.data_validation_config import DataValidationConfig
import pandas as pd


ingestion_config = DataIngestionConfig(
    database_name="CVD",
    collection_name="CVD_Data",
    feature_store_file_path="data/feature_store/cvd.csv",
    training_file_path="data/train/train.csv",
    testing_file_path="data/test/test.csv",
    train_test_split_ratio=0.2
)

ingestion = DataIngestion(ingestion_config)
ingestion_artifact = ingestion.initiate_data_ingestion()


validation_config = DataValidationConfig(
    schema_file_path="src/constants/schema.yaml",
    validation_report_path="artifacts/data_validation/report.json"
)

validator = DataValidation(
    dataframe=pd.read_csv(ingestion_artifact.feature_store_path, sep=';'),
    config=validation_config
)

validation_artifact = validator.initiate_data_validation()

if not validation_artifact.validation_status:
    raise Exception("❌ Data Validation Failed. Check report.")
