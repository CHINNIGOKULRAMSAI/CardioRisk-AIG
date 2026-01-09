import os
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.exception.exception import CVDException
from src.logger.logging import logging
from src.entity.config_entity.data_transformation_config import DataTransformationConfig
from src.entity.artifact_entity.data_transformation_artifact import DataTransformationArtifact


class DataTransformation:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: DataTransformationConfig
    ):
        try:
            self.train_df = train_df
            self.test_df = test_df
            self.config = config
        except Exception as e:
            raise CVDException(e, sys)

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Applying feature engineering")

            # BMI feature
            df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

            return df

        except Exception as e:
            raise CVDException(e, sys)

    def _get_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        try:
            numerical_features = X.columns.tolist()

            numeric_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, numerical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CVDException(e, sys)

    def _save_object(self, file_path: str, obj) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        except Exception as e:
            raise CVDException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            # Separate input & target
            X_train = self.train_df.drop(columns=[self.config.target_column])
            y_train = self.train_df[self.config.target_column]

            X_test = self.test_df.drop(columns=[self.config.target_column])
            y_test = self.test_df[self.config.target_column]

            # Drop 'id' column if it exists
            X_train = X_train.drop(columns=['id'], errors='ignore')
            X_test = X_test.drop(columns=['id'], errors='ignore')

            # Feature engineering
            X_train = self._feature_engineering(X_train)
            X_test = self._feature_engineering(X_test)

            # Preprocessing
            preprocessor = self._get_preprocessor(X_train)

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combine X and y
            train_array = np.c_[X_train_transformed, y_train.to_numpy()]
            test_array = np.c_[X_test_transformed, y_test.to_numpy()]

            # Save transformed data
            os.makedirs(
                os.path.dirname(self.config.transformed_train_path),
                exist_ok=True
            )

            np.save(self.config.transformed_train_path, train_array)
            np.save(self.config.transformed_test_path, test_array)

            # Save preprocessor
            self._save_object(
                self.config.preprocessor_object_path,
                preprocessor
            )

            logging.info("Data transformation completed successfully")

            return DataTransformationArtifact(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                preprocessor_object_path=self.config.preprocessor_object_path
            )

        except Exception as e:
            raise CVDException(e, sys)
