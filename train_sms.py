# sms_spam/train_sms.py
import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent / "data" / "raw" / "sms_spam.csv"
EXPERIMENT_NAME = "SMS_Spam_Quickstart"

# Only set experiment if not provided via env/CLI
if not os.environ.get("MLFLOW_EXPERIMENT_NAME") and not os.environ.get("MLFLOW_EXPERIMENT_ID"):
    mlflow.set_experiment(EXPERIMENT_NAME)

def load_data(path: Path) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    # Expect columns: label,text with label in {spam,ham}
    df["label"] = df["label"].str.strip().str.lower()
    y = (df["label"] == "spam").astype(int)  # spam=1, ham=0
    X = df["text"].astype(str)
    return X, y

def plot_confusion(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["ham", "spam"]); ax.set_yticklabels(["ham", "spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "tfidf__ngram_range": (1, 2),
        "tfidf__min_df": 2,
        "clf__C": 2.0,
        "clf__max_iter": 200,
        "clf__class_weight": None,
        "random_state": 42,
    }

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=params["tfidf__ngram_range"], min_df=params["tfidf__min_df"])),
        ("clf", LogisticRegression(C=params["clf__C"], max_iter=params["clf__max_iter"], random_state=params["random_state"]))
    ])

    with mlflow.start_run(run_name="train-sms-spam"):
        mlflow.log_params({
            "tfidf_ngram_range": str(params["tfidf__ngram_range"]),
            "tfidf_min_df": params["tfidf__min_df"],
            "clf_C": params["clf__C"],
            "clf_max_iter": params["clf__max_iter"]
        })

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        mlflow.log_metrics(metrics)

        # Confusion matrix artifact
        out_png = Path("confusion_matrix.png")
        plot_confusion(y_test, y_pred, out_png)
        mlflow.log_artifact(str(out_png))

        # Signature & example (strings in, ints out)
        from mlflow.models.signature import infer_signature
        input_example = pd.Series([
            "WIN a FREE entry to our prize draw!!! Reply WIN now.",
            "Hey, shall we meet at 6pm near the station?"
        ])
        # For signature inference, scikit pipeline accepts array-like of strings
        example_pred = pipeline.predict(input_example)
        signature = infer_signature(input_example.values, example_pred)

        # Pin requirements based on current env of training run
        import pkg_resources
        def pinned(pkg): return f"{pkg}=={pkg_resources.get_distribution(pkg).version}"
        reqs = [
            pinned("mlflow"),
            pinned("scikit-learn"),
            pinned("numpy"),
            pinned("pandas"),
            pinned("scipy"),
            pinned("cloudpickle")
        ]

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example.to_list(),
            pip_requirements=reqs
        )

        run = mlflow.active_run()
        print("Run complete. run_id=", run.info.run_id)

if __name__ == "__main__":
    main()
