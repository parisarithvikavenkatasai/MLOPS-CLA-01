import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Allow Python to find src folder
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.preprocess import (
    clean_raw_data,
    engineer_features,
    encode_target,
    build_preprocessor
)

from src.train import (
    build_and_train_pipelines,
    save_model_pipeline
)

from src.evaluate import (
    compare_models
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = (
    "data/raw/test.csv.xlsx"
)

MODEL_PATH = (
    "models/best_model.joblib"
)

OUTPUT_PATH = (
    "outputs/metrics.csv"
)


# ============================================================
# 2. CREATE FOLDERS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print()
print("==========================================")
print("MLOPS PIPELINE STARTED")
print("==========================================")

print()
print("Loading dataset...")

df = pd.read_excel(
    DATA_PATH
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 4. DATA CLEANING
# ============================================================

print()
print("Cleaning data...")

df = clean_raw_data(
    df,
    id_cols=[
        "Unnamed: 0",
        "id"
    ]
)

print(
    "Data cleaning completed."
)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

print()
print("Engineering features...")

df = engineer_features(
    df
)

print(
    "Feature engineering completed."
)


# ============================================================
# 6. TARGET
# ============================================================

target_column = "satisfaction"

y = encode_target(
    df[target_column]
)

X = df.drop(
    columns=[target_column]
)


# ============================================================
# 7. FIND FEATURES
# ============================================================

categorical_columns = (

    X.select_dtypes(

        include=[
            "object",
            "category",
            "string"
        ]

    ).columns.tolist()

)


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


# ============================================================
# 8. PREPROCESSOR
# ============================================================

preprocessor = build_preprocessor(

    categorical_columns,

    numerical_columns

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
# 10. TRAIN MODELS
# ============================================================

print()
print("==========================================")
print("MODEL TRAINING")
print("==========================================")


trained_pipelines = (

    build_and_train_pipelines(

        X_train,

        y_train,

        preprocessor

    )

)


print()
print(
    "All models trained successfully."
)


# ============================================================
# 11. EVALUATE MODELS
# ============================================================

print()
print("==========================================")
print("MODEL EVALUATION")
print("==========================================")


summary_df, all_metrics, best_model_name = (

    compare_models(

        trained_pipelines,

        X_test,

        y_test

    )

)


print()
print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE METRICS
# ============================================================

summary_df.to_csv(

    OUTPUT_PATH,

    index=False

)


print()
print(
    "Metrics saved to:",
    OUTPUT_PATH
)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

best_pipeline = (

    trained_pipelines[
        best_model_name
    ]

)


save_model_pipeline(

    best_pipeline,

    MODEL_PATH

)


# ============================================================
# 14. FINAL RESULT
# ============================================================

print()
print("==========================================")

print(
    "BEST MODEL:",
    best_model_name
)

print(
    "MODEL SAVED TO:",
    MODEL_PATH
)

print(
    "MLOPS PIPELINE COMPLETED SUCCESSFULLY!"
)

print("==========================================")