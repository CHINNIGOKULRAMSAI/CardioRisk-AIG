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



import pandas as pd
from src.components.data_transformation.transform import DataTransformation
from src.entity.config_entity.data_transformation_config import DataTransformationConfig

# Load train/test data
train_df = pd.read_csv("data/train/train.csv", sep=';')
test_df = pd.read_csv("data/test/test.csv", sep=';')

# Transformation config
transformation_config = DataTransformationConfig(
    transformed_train_path="artifacts/transformed/train.npy",
    transformed_test_path="artifacts/transformed/test.npy",
    preprocessor_object_path="artifacts/preprocessor/preprocessor.pkl",
    target_column="cardio"
)

transformer = DataTransformation(
    train_df=train_df,
    test_df=test_df,
    config=transformation_config
)

transformation_artifact = transformer.initiate_data_transformation()



from src.components.model_training.training import ModelTrainer
from src.entity.config_entity.model_trainer_config import ModelTrainerConfig
import numpy as np

# Load transformed data
train_array = np.load(transformation_artifact.transformed_train_path)
test_array = np.load(transformation_artifact.transformed_test_path)

trainer_config = ModelTrainerConfig(
    trained_model_file_path="artifacts/model/model.pkl",
    expected_balanced_accuracy=0.70
)

trainer = ModelTrainer(
    train_array=train_array,
    test_array=test_array,
    config=trainer_config
)

trainer_artifact = trainer.initiate_model_trainer()
