import os
import sys
import pickle
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier

from src.exception.exception import CVDException
from src.logger.logging import logging
from src.entity.config_entity.model_trainer_config import ModelTrainerConfig
from src.entity.artifact_entity.model_trainer_artifact import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        train_array: np.ndarray,
        test_array: np.ndarray,
        config: ModelTrainerConfig
    ):
        try:
            self.X_train = train_array[:, :-1]
            self.y_train = train_array[:, -1]

            self.X_test = test_array[:, :-1]
            self.y_test = test_array[:, -1]

            self.config = config

        except Exception as e:
            raise CVDException(e, sys)

    def _save_model(self, model) -> None:
        try:
            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path),
                exist_ok=True
            )
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            raise CVDException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting RandomForest-only model training")

            # ---------------- BASE MODEL ----------------
            rf = RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

            # ---------------- STRONG BUT SAFE PARAMS ----------------
            param_dist = {
                "n_estimators": [400, 600, 800],
                "max_depth": [18, 22, 26, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True]
            }

            cv = StratifiedKFold(
                n_splits=3,          # fast + reliable
                shuffle=True,
                random_state=42
            )

            search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                n_iter=8,            # 🔥 focused search
                scoring="balanced_accuracy",
                cv=cv,
                verbose=2,
                n_jobs=-1,
                random_state=42
            )

            search.fit(self.X_train, self.y_train)

            best_model = search.best_estimator_

            # ---------------- EVALUATION ----------------
            y_pred = best_model.predict(self.X_test)

            bal_acc = balanced_accuracy_score(self.y_test, y_pred)
            acc = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)

            logging.info(f"RandomForest Accuracy: {acc}")
            logging.info(f"RandomForest Balanced Accuracy: {bal_acc}")
            logging.info(f"RandomForest Precision: {precision}")
            logging.info(f"RandomForest Recall: {recall}")

            logging.info(
                f"RandomForest Classification Report:\n"
                f"{classification_report(self.y_test, y_pred)}"
            )

            if bal_acc < self.config.expected_balanced_accuracy:
                raise CVDException(
                    f"Balanced accuracy {bal_acc} below expected threshold",
                    sys
                )

            self._save_model(best_model)

            logging.info(
                f"Final RandomForest model saved "
                f"with balanced accuracy: {bal_acc}"
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path,
                best_model_name="RandomForestClassifier",
                best_accuracy=bal_acc
            )

        except Exception as e:
            raise CVDException(e, sys)
