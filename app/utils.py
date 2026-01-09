import pickle
import pandas as pd

MODEL_PATH = "artifacts/model/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor/preprocessor.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_preprocessor():
    with open(PREPROCESSOR_PATH, "rb") as f:
        return pickle.load(f)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # BMI feature (same as training)
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    return df
