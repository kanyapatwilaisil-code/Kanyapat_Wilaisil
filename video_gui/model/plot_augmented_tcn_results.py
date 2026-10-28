#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Balanced Accuracy":
            balanced_accuracy_score(y_true, y_pred),

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# METRIC BAR CHART
# ============================================================

def plot_metrics(metrics, title, output):

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
        figsize=(9, 5)
    )

    bars = ax.bar(
        names,
        values
    )

    ax.set_ylim(0, 1.05)

    ax.set_ylabel(
        "Metric Value"
    )

    ax.set_title(
        title
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.02,

            f"{value:.3f}",

            ha="center",
            va="bottom"
        )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    title,
    output
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        cm,
        cmap="Blues"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    classes = [
        "HC",
        "PD"
    ]

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        classes
    )

    ax.set_yticklabels(
        classes
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        title
    )

    threshold = (
        cm.max() / 2
        if cm.max() > 0
        else 0
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),

                ha="center",
                va="center",

                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),

                fontsize=16,
                fontweight="bold"
            )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc(
    y_true,
    scores,
    title,
    output
):

    auc = roc_auc_score(
        y_true,
        scores
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        scores
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        title
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SUBJECT-LEVEL SCORE
# ============================================================

def make_subject_scores(df):

    subject = (
        df
        .groupby("group_id")
        .agg(
            true_label=(
                "label",
                "first"
            ),

            score=(
                "score",
                "mean"
            ),

            n_windows=(
                "score",
                "size"
            )
        )
        .reset_index()
    )

    return subject


# ============================================================
# THRESHOLD GRAPH
# ============================================================

def plot_thresholds(
    window_threshold,
    subject_threshold,
    output
):

    names = [
        "Window-Level\nThreshold",
        "Subject-Level\nThreshold"
    ]

    values = [
        window_threshold,
        subject_threshold
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        names,
        values
    )

    ax.set_ylabel(
        "Threshold"
    )

    ax.set_title(
        "Validation-Derived Decision Thresholds"
    )

    ax.set_ylim(
        0,
        max(values) + 0.1
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.015,

            f"{value:.4f}",

            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validation",
        required=True
    )

    parser.add_argument(
        "--test",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    # Selected from VALIDATION
    parser.add_argument(
        "--window-threshold",
        type=float,
        default=0.5158
    )

    # Selected from VALIDATION subject-level analysis
    parser.add_argument(
        "--subject-threshold",
        type=float,
        default=0.4067
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    validation = pd.read_csv(
        args.validation
    )

    test = pd.read_csv(
        args.test
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    y_val = validation[
        "label"
    ].astype(int).to_numpy()

    score_val = validation[
        "score"
    ].astype(float).to_numpy()

    pred_val = (
        score_val
        >= args.window_threshold
    ).astype(int)

    validation_metrics = calculate_metrics(
        y_val,
        pred_val
    )

    validation_auc = roc_auc_score(
        y_val,
        score_val
    )

    print("\n================================")
    print("VALIDATION - WINDOW LEVEL")
    print("================================")

    print(
        f"ROC-AUC: {validation_auc:.4f}"
    )

    for key in [
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
    ]:

        print(
            f"{key}: "
            f"{validation_metrics[key]:.4f}"
        )

    # ========================================================
    # TEST WINDOW
    # ========================================================

    y_test = test[
        "label"
    ].astype(int).to_numpy()

    score_test = test[
        "score"
    ].astype(float).to_numpy()

    pred_test_window = (
        score_test
        >= args.window_threshold
    ).astype(int)

    test_window_metrics = calculate_metrics(
        y_test,
        pred_test_window
    )

    test_window_auc = roc_auc_score(
        y_test,
        score_test
    )

    print("\n================================")
    print("TEST - WINDOW LEVEL")
    print("================================")

    print(
        f"ROC-AUC: {test_window_auc:.4f}"
    )

    for key in [
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
    ]:

        print(
            f"{key}: "
            f"{test_window_metrics[key]:.4f}"
        )

    print(
        "Confusion Matrix:"
    )

    print(
        f"TN={test_window_metrics['TN']} "
        f"FP={test_window_metrics['FP']} "
        f"FN={test_window_metrics['FN']} "
        f"TP={test_window_metrics['TP']}"
    )

    # ========================================================
    # TEST SUBJECT LEVEL
    # ========================================================

    test_subject = make_subject_scores(
        test
    )

    pred_test_subject = (
        test_subject["score"]
        >= args.subject_threshold
    ).astype(int)

    y_test_subject = (
        test_subject["true_label"]
        .astype(int)
    )

    test_subject_metrics = calculate_metrics(
        y_test_subject,
        pred_test_subject
    )

    test_subject_auc = roc_auc_score(
        y_test_subject,
        test_subject["score"]
    )

    print("\n================================")
    print("TEST - SUBJECT LEVEL")
    print("================================")

    print(
        f"Subjects: "
        f"{len(test_subject)}"
    )

    print(
        f"Threshold: "
        f"{args.subject_threshold:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{test_subject_auc:.4f}"
    )

    for key in [
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
    ]:

        print(
            f"{key}: "
            f"{test_subject_metrics[key]:.4f}"
        )

    print(
        "Confusion Matrix:"
    )

    print(
        f"TN={test_subject_metrics['TN']} "
        f"FP={test_subject_metrics['FP']} "
        f"FN={test_subject_metrics['FN']} "
        f"TP={test_subject_metrics['TP']}"
    )

    # Save subject scores
    test_subject[
        "prediction"
    ] = pred_test_subject

    test_subject[
        "true_label_text"
    ] = test_subject[
        "true_label"
    ].map({
        0: "HC",
        1: "PD"
    })

    test_subject[
        "prediction_text"
    ] = test_subject[
        "prediction"
    ].map({
        0: "HC",
        1: "PD"
    })

    test_subject.to_csv(
        output
        / "test_subject_scores.csv",
        index=False
    )

    # ========================================================
    # GRAPHS
    # ========================================================

    plot_metrics(
        validation_metrics,

        "Augmented TCN-F3 Validation Performance (Window Level)",

        output
        / "fig_validation_window_metrics.png"
    )

    plot_metrics(
        test_window_metrics,

        "Augmented TCN-F3 Test Performance (Window Level)",

        output
        / "fig_test_window_metrics.png"
    )

    plot_metrics(
        test_subject_metrics,

        "Augmented TCN-F3 Test Performance (Subject Level)",

        output
        / "fig_test_subject_metrics.png"
    )

    plot_confusion_matrix(
        y_test,
        pred_test_window,

        (
            "Augmented TCN-F3 Test Confusion Matrix "
            f"(Window Level, Threshold = "
            f"{args.window_threshold:.4f})"
        ),

        output
        / "fig_test_window_confusion_matrix.png"
    )

    plot_confusion_matrix(
        y_test_subject,
        pred_test_subject,

        (
            "Augmented TCN-F3 Test Confusion Matrix "
            f"(Subject Level, Threshold = "
            f"{args.subject_threshold:.4f})"
        ),

        output
        / "fig_test_subject_confusion_matrix.png"
    )

    plot_roc(
        y_test,
        score_test,

        "Augmented TCN-F3 Test ROC Curve (Window Level)",

        output
        / "fig_test_window_roc.png"
    )

    plot_roc(
        y_test_subject,
        test_subject["score"],

        "Augmented TCN-F3 Test ROC Curve (Subject Level)",

        output
        / "fig_test_subject_roc.png"
    )

    plot_thresholds(
        args.window_threshold,
        args.subject_threshold,

        output
        / "fig_validation_threshold_comparison.png"
    )

    print("\n================================")
    print("DONE")
    print("================================")

    print(
        f"Figures and CSV files saved to:\n"
        f"{output}"
    )


if __name__ == "__main__":
    main()