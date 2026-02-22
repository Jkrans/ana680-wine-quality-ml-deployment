# train.py
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "wine-quality"
RED_PATH = DATA_DIR / "winequality-red.csv"
WHITE_PATH = DATA_DIR / "winequality-white.csv"


def load_data() -> pd.DataFrame:
    if not RED_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RED_PATH}")
    if not WHITE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {WHITE_PATH}")

    red = pd.read_csv(RED_PATH, sep=";")
    white = pd.read_csv(WHITE_PATH, sep=";")

    # Combine red + white wine rows
    df = pd.concat([red, white], ignore_index=True)

    # Basic sanity check
    if "quality" not in df.columns:
        raise ValueError("Expected target column 'quality' not found in dataset.")

    return df


def main() -> None:
    df = load_data()

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Regularized linear regression + scaling to avoid numerical overflow warnings
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    joblib.dump(model, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()