import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def build_and_train_pipelines(
    X_train,
    y_train,
    preprocessor,
    model_configs=None
):

    models = {

        "Logistic_Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Random_Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss"
            )
    }

    trained_pipelines = {}

    for name, model in models.items():

        print("Training:", name)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        trained_pipelines[name] = pipeline

    return trained_pipelines


def save_model_pipeline(
    pipeline,
    path
):

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(
            folder,
            exist_ok=True
        )

    joblib.dump(
        pipeline,
        path
    )

    print(
        "Model saved to:",
        path
    )