#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


LABEL_NAMES = ["HC", "PD"]


# ============================================================
# Load predictions
# ============================================================

def load_predictions(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "subject_id",
        "source",
        "label",
        "prediction",
        "score",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[
            "label",
            "prediction",
            "score",
        ]
    ).copy()

    df["label"] = df["label"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["score"] = df["score"].astype(float)

    return df


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "Sensitivity": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),

        "Specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# Metrics bar chart
# ============================================================

def save_metrics_plot(metrics, output_path):

    names = [
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
    ]

    values = [
        metrics[name]
        for name in names
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        names,
        values,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Metric"
    )

    ax.set_title(
        "TRAIN Model Performance"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Confusion matrix
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    output_path,
    title,
    normalized=False,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    if normalized:

        row_sum = cm.sum(
            axis=1,
            keepdims=True,
        )

        display = np.divide(
            cm.astype(float),
            row_sum,
            out=np.zeros_like(
                cm,
                dtype=float,
            ),
            where=row_sum != 0,
        )

    else:

        display = cm

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        display,
        aspect="equal",
    )

    fig.colorbar(
        im,
        ax=ax,
    )

    ax.set_xticks(
        [0, 1],
        LABEL_NAMES,
    )

    ax.set_yticks(
        [0, 1],
        LABEL_NAMES,
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_title(
        title
    )

    for i in range(2):

        for j in range(2):

            if normalized:

                text = (
                    f"{display[i, j]:.2f}"
                )

            else:

                text = str(
                    int(cm[i, j])
                )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Subject-level analysis
# ============================================================

def make_subject_level(
    df,
    threshold,
):

    subject = (
        df.groupby(
            [
                "subject_id",
                "source",
            ],
            as_index=False,
        )
        .agg(
            true_label=(
                "label",
                "first",
            ),

            n_windows=(
                "label",
                "size",
            ),

            mean_score=(
                "score",
                "mean",
            ),

            median_score=(
                "score",
                "median",
            ),

            pd_window_fraction=(
                "prediction",
                "mean",
            ),
        )
    )

    subject["predicted_label"] = (
        subject["mean_score"]
        >= threshold
    ).astype(int)

    subject["correct"] = (
        subject["true_label"]
        ==
        subject["predicted_label"]
    )

    return subject


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate TRAIN metrics, "
            "graphs and confusion matrices."
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train_predictions.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4067,
        help=(
            "Subject-level threshold. "
            "Default = 0.4067"
        ),
    )

    args = parser.parse_args()

    output_dir = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load
    # ========================================================

    df = load_predictions(
        args.train
    )

    print()
    print("=" * 60)
    print("TRAIN ANALYSIS")
    print("=" * 60)

    print(
        f"Windows: {len(df)}"
    )

    print(
        f"Subjects: "
        f"{df['subject_id'].nunique()}"
    )

    print()
    print("Labels:")

    print(
        df["label"]
        .map({
            0: "HC",
            1: "PD",
        })
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Window level
    # ========================================================

    y_true = df["label"]
    y_pred = df["prediction"]

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print()
    print("=== WINDOW LEVEL ===")

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    # ========================================================
    # Save metrics
    # ========================================================

    metrics_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in metrics.items()
            }
        ]
    )

    metrics_df.to_csv(
        output_dir
        / "train_metrics.csv",
        index=False,
    )

    # ========================================================
    # Metrics graph
    # ========================================================

    save_metrics_plot(
        metrics,
        output_dir
        / "train_metrics.png",
    )

    # ========================================================
    # Window confusion matrix
    # ========================================================

    save_confusion_matrix(
        y_true,
        y_pred,
        output_dir
        / "confusion_matrix_train.png",
        "TRAIN confusion matrix",
        normalized=False,
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        output_dir
        / "confusion_matrix_train_normalized.png",
        "TRAIN normalized confusion matrix",
        normalized=True,
    )

    # ========================================================
    # Subject level
    # ========================================================

    subject = make_subject_level(
        df,
        threshold=args.threshold,
    )

    subject_metrics = calculate_metrics(
        subject["true_label"],
        subject["predicted_label"],
    )

    print()
    print("=== SUBJECT LEVEL ===")

    print(
        f"Threshold: "
        f"{args.threshold:.4f}"
    )

    print(
        f"Subjects: "
        f"{len(subject)}"
    )

    for key, value in subject_metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    # ========================================================
    # Subject CSV
    # ========================================================

    subject.to_csv(
        output_dir
        / "train_subject_performance.csv",
        index=False,
    )

    # ========================================================
    # Subject confusion matrices
    # ========================================================

    save_confusion_matrix(
        subject["true_label"],
        subject["predicted_label"],
        output_dir
        / "confusion_matrix_train_subject.png",
        "TRAIN subject-level confusion matrix",
        normalized=False,
    )

    save_confusion_matrix(
        subject["true_label"],
        subject["predicted_label"],
        output_dir
        / "confusion_matrix_train_subject_normalized.png",
        "TRAIN subject-level normalized confusion matrix",
        normalized=True,
    )

    # ========================================================
    # Subject metrics
    # ========================================================

    subject_metrics_df = pd.DataFrame(
        [
            subject_metrics
        ]
    )

    subject_metrics_df.to_csv(
        output_dir
        / "train_subject_metrics.csv",
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "level": "window",
                "n": len(df),
                **metrics,
            },
            {
                "level": "subject",
                "n": len(subject),
                **subject_metrics,
            },
        ]
    )

    summary.to_csv(
        output_dir
        / "train_summary.csv",
        index=False,
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Results saved to:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()