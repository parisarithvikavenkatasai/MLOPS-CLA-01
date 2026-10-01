import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/raw/test.csv.xlsx"


# ============================================================
# 2. MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Airline_Passenger_Satisfaction"
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

df = pd.read_excel(
    DATA_PATH
)

print(
    "Dataset loaded successfully!"
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Remove unnecessary columns

columns_to_drop = [
    "Unnamed: 0",
    "id"
]

for col in columns_to_drop:

    if col in df.columns:

        df = df.drop(
            columns=[col]
        )


# Convert delay columns to numbers

delay_columns = [
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes"
]

for col in delay_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = df[col].fillna(0)


print(
    "Data cleaning completed."
)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# Total delay

df["TotalDelay"] = (
    df["Departure Delay in Minutes"]
    +
    df["Arrival Delay in Minutes"]
)


# Service columns

service_columns = [

    "Inflight wifi service",

    "Departure/Arrival time convenient",

    "Ease of Online booking",

    "Gate location",

    "Food and drink",

    "Online boarding",

    "Seat comfort",

    "Inflight entertainment",

    "On-board service",

    "Leg room service",

    "Baggage handling",

    "Checkin service",

    "Inflight service",

    "Cleanliness"
]


# Select only columns that exist

existing_service_columns = [

    col

    for col in service_columns

    if col in df.columns
]


# Composite service score

df["CompositeServiceScore"] = (

    df[existing_service_columns]
    .mean(axis=1)

)


print(
    "Feature engineering completed."
)


# ============================================================
# 6. TARGET VARIABLE
# ============================================================

target_column = "satisfaction"


y = df[target_column].map({

    "neutral or dissatisfied": 0,

    "satisfied": 1

})


# Remove target column

X = df.drop(
    columns=[target_column]
)


# ============================================================
# 7. HANDLE CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

# IMPORTANT:
# Text columns such as Gender are categorical.

categorical_columns = (

    X.select_dtypes(

        include=[
            "object",
            "category",
            "string"
        ]

    ).columns.tolist()

)


# Numerical columns

numerical_columns = (

    X.select_dtypes(

        include=[
            "number"
        ]

    ).columns.tolist()

)


print()
print(
    "Numerical features:",
    len(numerical_columns)
)

print(
    "Categorical features:",
    len(categorical_columns)
)


print()
print(
    "Categorical columns:",
    categorical_columns
)


# ============================================================
# 8. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",

            StandardScaler(),

            numerical_columns
        ),

        (
            "cat",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_columns
        )

    ]

)


print()
print(
    "Preprocessor created."
)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (

    train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

)


print()
print(
    "Train/Test split completed."
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ============================================================
# 10. MODELS
# ============================================================

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


# ============================================================
# 11. TRAIN MODELS
# ============================================================

trained_pipelines = {}


print()
print(
    "========== MODEL TRAINING =========="
)


for name, model in models.items():


    print(
        "Training:",
        name
    )


    pipeline = Pipeline([

        (
            "preprocessor",

            preprocessor

        ),

        (
            "model",

            model

        )

    ])


    pipeline.fit(

        X_train,

        y_train

    )


    trained_pipelines[name] = pipeline


print()
print(
    "All models trained successfully."
)


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

rows = []

all_metrics = {}


print()
print(
    "========== MODEL EVALUATION =========="
)


for name, pipeline in trained_pipelines.items():


    # Predictions

    predictions = pipeline.predict(
        X_test
    )


    # Probability predictions

    probabilities = (

        pipeline.predict_proba(
            X_test
        )[:, 1]

    )


    # Accuracy

    accuracy = accuracy_score(

        y_test,

        predictions

    )


    # Precision

    precision = precision_score(

        y_test,

        predictions

    )


    # Recall

    recall = recall_score(

        y_test,

        predictions

    )


    # F1

    f1 = f1_score(

        y_test,

        predictions

    )


    # ROC-AUC

    roc_auc = roc_auc_score(

        y_test,

        probabilities

    )


    # Store metrics

    all_metrics[name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc

    }


    # Table row

    rows.append({

        "Model": name,

        "Accuracy (%)":
            f"{accuracy * 100:.2f}%",

        "Precision (%)":
            f"{precision * 100:.2f}%",

        "Recall (%)":
            f"{recall * 100:.2f}%",

        "F1-Score (%)":
            f"{f1 * 100:.2f}%",

        "ROC-AUC (%)":
            f"{roc_auc * 100:.2f}%"

    })


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(
    rows
)


print()
print(
    "MODEL RESULTS:"
)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE METRICS
# ============================================================

os.makedirs(

    "outputs",

    exist_ok=True

)


results_df.to_csv(

    "outputs/metrics.csv",

    index=False

)


print()
print(
    "Metrics saved to:"
)

print(
    "outputs/metrics.csv"
)


# ============================================================
# 15. FIND BEST MODEL
# ============================================================

best_model_name = max(

    all_metrics,

    key=lambda name:

    all_metrics[name]["Accuracy"]

)


print()
print(
    "Best model:",
    best_model_name
)


# ============================================================
# 16. MLFLOW LOGGING
# ============================================================

print()
print(
    "========== MLFLOW LOGGING =========="
)


for name, pipeline in trained_pipelines.items():


    print()
    print(
        "Logging:",
        name
    )


    with mlflow.start_run(

        run_name=name

    ):


        # ------------------------------------------------
        # PARAMETERS
        # ------------------------------------------------

        mlflow.log_param(

            "model",

            name

        )


        mlflow.log_param(

            "dataset",

            "Airline Passenger Satisfaction"

        )


        mlflow.log_param(

            "test_size",

            0.20

        )


        mlflow.log_param(

            "random_state",

            42

        )


        # ------------------------------------------------
        # METRICS
        # ------------------------------------------------

        metrics = all_metrics[name]


        mlflow.log_metric(

            "accuracy",

            metrics["Accuracy"]

        )


        mlflow.log_metric(

            "precision",

            metrics["Precision"]

        )


        mlflow.log_metric(

            "recall",

            metrics["Recall"]

        )


        mlflow.log_metric(

            "f1",

            metrics["F1"]

        )


        mlflow.log_metric(

            "roc_auc",

            metrics["ROC_AUC"]

        )


        # ------------------------------------------------
        # MODEL LOGGING
        # ------------------------------------------------

        mlflow.sklearn.log_model(

            pipeline,

            name="model",

            skops_trusted_types=[

                "sklearn.tree._tree.Tree",

                "xgboost.core.Booster",

                "xgboost.sklearn.XGBClassifier"

            ]

        )


    print(

        "MLflow logging completed for:",

        name

    )


# ============================================================
# 17. COMPLETED
# ============================================================

print()
print(
    "=========================================="
)

print(
    "MLflow tracking completed successfully!"
)

print(
    "=========================================="
)