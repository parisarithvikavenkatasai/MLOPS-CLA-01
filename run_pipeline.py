import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Allow Python to find files inside src
sys.path.append(os.path.abspath("."))

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

from src.evaluate import compare_models


# ==========================================
# 1. CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

data_path = "data/raw/test.csv.xlsx"

print("\nLoading dataset...")

df = pd.read_excel(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 3. REMOVE ID COLUMNS
# ==========================================

id_columns = [
    "Unnamed: 0",
    "id"
]

df = clean_raw_data(
    df,
    id_cols=id_columns
)

print("\nData cleaning completed.")


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

df = engineer_features(df)

print("Feature engineering completed.")

print("Created features:")
print("- TotalDelay")
print("- CompositeServiceScore")


# ==========================================
# 5. SEPARATE INPUT AND TARGET
# ==========================================

y = encode_target(
    df["satisfaction"]
)

X = df.drop(
    columns=["satisfaction"]
)


# ==========================================
# 6. DEFINE FEATURES
# ==========================================

numerical_columns = [

    "Age",
    "Flight Distance",

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
    "Cleanliness",

    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",

    "TotalDelay",
    "CompositeServiceScore"
]


categorical_columns = [

    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class"
]


# Keep only columns that actually exist
numerical_columns = [
    col for col in numerical_columns
    if col in X.columns
]

categorical_columns = [
    col for col in categorical_columns
    if col in X.columns
]


print(
    "\nNumerical features:",
    len(numerical_columns)
)

print(
    "Categorical features:",
    len(categorical_columns)
)


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTrain/Test split completed.")

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ==========================================
# 8. CREATE PREPROCESSOR
# ==========================================

preprocessor = build_preprocessor(

    categorical_columns,

    numerical_columns

)


print("\nPreprocessor created.")


# ==========================================
# 9. TRAIN MODELS
# ==========================================

print("\n========== MODEL TRAINING ==========")

trained_pipelines = build_and_train_pipelines(

    X_train=X_train,

    y_train=y_train,

    preprocessor=preprocessor,

    model_configs=None

)


print("\nAll models trained successfully.")


# ==========================================
# 10. EVALUATE MODELS
# ==========================================

print("\n========== MODEL EVALUATION ==========")

summary_df, all_metrics, best_model_name = compare_models(

    trained_pipelines,

    X_test,

    y_test

)


print("\nMODEL RESULTS:")
print(summary_df.to_string(index=False))


# ==========================================
# 11. SAVE METRICS
# ==========================================

summary_df.to_csv(

    "outputs/metrics.csv",

    index=False

)

print(
    "\nMetrics saved to:"
)

print(
    "outputs/metrics.csv"
)


# ==========================================
# 12. GET BEST MODEL
# ==========================================

best_pipeline = trained_pipelines[
    best_model_name
]


print(
    "\nBest model:",
    best_model_name
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

predictions = best_pipeline.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    predictions
)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Neutral/Dissatisfied",
        "Satisfied"
    ]

)

display.plot()

plt.title(
    "Confusion Matrix - "
    + best_model_name
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()


print(
    "Confusion matrix saved."
)


# ==========================================
# 14. ROC CURVE
# ==========================================

plt.figure(figsize=(8, 6))


for name, pipeline in trained_pipelines.items():

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png"
)

plt.close()


print(
    "ROC curve saved."
)


# ==========================================
# 15. SAVE BEST MODEL
# ==========================================

model_path = (
    "models/best_model.joblib"
)


save_model_pipeline(

    best_pipeline,

    model_path

)


# ==========================================
# 16. FINAL MESSAGE
# ==========================================

print("\n====================================")
print("PROJECT EXECUTION COMPLETED")
print("====================================")

print(
    "Dataset:",
    data_path
)

print(
    "Best Model:",
    best_model_name
)

print(
    "Saved Model:",
    model_path
)

print(
    "Metrics:",
    "outputs/metrics.csv"
)

print(
    "Confusion Matrix:",
    "outputs/confusion_matrix.png"
)

print(
    "ROC Curve:",
    "outputs/roc_curve.png"
)

print("====================================")