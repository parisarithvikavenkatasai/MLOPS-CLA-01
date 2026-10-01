import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_single_pipeline(
    pipeline,
    X_test,
    y_test
):

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        pipeline.predict_proba(X_test)[:, 1]
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }


def compare_models(
    trained_pipelines,
    X_test,
    y_test
):

    rows = []

    all_metrics = {}

    for name, pipeline in trained_pipelines.items():

        metrics = evaluate_single_pipeline(
            pipeline,
            X_test,
            y_test
        )

        all_metrics[name] = metrics

        rows.append({
            "Model": name,
            "Accuracy (%)":
                f"{metrics['Accuracy'] * 100:.2f}%",

            "Precision (%)":
                f"{metrics['Precision'] * 100:.2f}%",

            "Recall (%)":
                f"{metrics['Recall'] * 100:.2f}%",

            "F1-Score (%)":
                f"{metrics['F1'] * 100:.2f}%",

            "ROC-AUC (%)":
                f"{metrics['ROC_AUC'] * 100:.2f}%"
        })

    summary_df = pd.DataFrame(rows)

    # Select model with highest accuracy
    best_model_name = max(
        all_metrics,
        key=lambda name:
        all_metrics[name]["Accuracy"]
    )

    return (
        summary_df,
        all_metrics,
        best_model_name
    )