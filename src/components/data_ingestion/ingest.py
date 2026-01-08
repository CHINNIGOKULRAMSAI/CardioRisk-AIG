import os
import sys
import pandas as pd
import numpy as np
import pymongo
import certifi
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.exception.exception import CVDException
from src.logger.logging import logging
from src.entity.config_entity.data_ingestion_config import DataIngestionConfig
from src.entity.artifact_entity.data_ingestion_artifact import DataIngestionArtifact

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config
        except Exception as e:
            raise CVDException(e, sys)

    def _fetch_from_mongodb(self) -> pd.DataFrame:
        try:
            logging.info("Connecting to MongoDB")

            client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=ca
            )

            collection = client[self.config.database_name][
                self.config.collection_name
            ]

            df = pd.DataFrame(list(collection.find()))
            client.close()

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)

            logging.info(f"Fetched {df.shape[0]} records from MongoDB")

            return df

        except Exception as e:
            raise CVDException(e, sys)

    def _save_feature_store(self, df: pd.DataFrame) -> None:
        try:
            os.makedirs(
                os.path.dirname(self.config.feature_store_file_path),
                exist_ok=True
            )

            df.to_csv(
                self.config.feature_store_file_path,
                index=False,
                header=True
            )

            logging.info(
                f"Feature store saved at "
                f"{self.config.feature_store_file_path}"
            )

        except Exception as e:
            raise CVDException(e, sys)

    def _split_and_save(self, df: pd.DataFrame) -> None:
        try:
            logging.info("Performing train-test split")

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.train_test_split_ratio,
                random_state=42
            )

            os.makedirs(
                os.path.dirname(self.config.training_file_path),
                exist_ok=True
            )

            train_df.to_csv(
                self.config.training_file_path,
                index=False,
                header=True
            )

            test_df.to_csv(
                self.config.testing_file_path,
                index=False,
                header=True
            )

            logging.info("Train-test split completed")
            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")

        except Exception as e:
            raise CVDException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion")

            df = self._fetch_from_mongodb()
            self._save_feature_store(df)
            self._split_and_save(df)

            artifact = DataIngestionArtifact(
                feature_store_path=self.config.feature_store_file_path,
                train_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path
            )

            logging.info("Data Ingestion completed successfully")

            return artifact

        except Exception as e:
            raise CVDException(e, sys)
