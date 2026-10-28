#!/usr/bin/env python3

import argparse
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    roc_auc_score,
    roc_curve,
)


def evaluate_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, scores):
    candidates = np.unique(scores)

    # Also test boundaries.
    candidates = np.concatenate([
        [0.0],
        candidates,
        [0.5],
        [1.0],
    ])

    candidates = np.unique(
        np.clip(candidates, 0.0, 1.0)
    )

    results = []

    for threshold in candidates:
        results.append(
            evaluate_threshold(
                y_true,
                scores,
                threshold,
            )
        )

    df = pd.DataFrame(results)

    # Primary criterion:
    # maximize balanced accuracy.
    #
    # Tie breakers:
    # 1. higher sensitivity
    # 2. higher specificity
    df = df.sort_values(
        [
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def save_threshold_plot(
    results,
    y_true,
    scores,
    best,
    auc,
    output_path,
    title,
):
    """Left: metrics vs threshold. Right: ROC curve with the best point."""

    r = results.sort_values("threshold").reset_index(drop=True)
    best_t = float(best["threshold"])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    for col, label in [
        ("balanced_accuracy", "Balanced accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
    ]:
        ax1.plot(
            r["threshold"],
            r[col],
            linewidth=2 if col == "balanced_accuracy" else 1.4,
            label=label,
        )

    ax1.axvline(
        best_t,
        color="red",
        linestyle="--",
        label=f"Best threshold = {best_t:.4f}",
    )

    ax1.axvline(
        0.5,
        color="gray",
        linestyle=":",
        label="0.5",
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title(f"{title}: metrics vs threshold")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="lower center", fontsize=9)

    fpr, tpr, _ = roc_curve(y_true, scores)

    ax2.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC (AUC = {auc:.3f})",
    )

    ax2.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Chance",
    )

    ax2.scatter(
        [1 - float(best["specificity"])],
        [float(best["sensitivity"])],
        color="red",
        zorder=5,
        label=(
            f"Best threshold {best_t:.4f}\n"
            f"Sens {best['sensitivity']:.3f}, "
            f"Spec {best['specificity']:.3f}"
        ),
    )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("False Positive Rate (1 - Specificity)")
    ax2.set_ylabel("True Positive Rate (Sensitivity)")
    ax2.set_title(f"{title}: ROC curve")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_metrics_bar(best, auc, title, output_path):
    """Bar chart of the metrics at the best threshold."""

    names = [
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
        ("F1", "f1"),
    ]

    labels = [n for n, _ in names]
    values = [float(best[k]) for _, k in names]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(labels, values)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_title(
        f"{title}\n"
        f"threshold = {float(best['threshold']):.4f}, "
        f"ROC-AUC = {auc:.3f}"
    )
    ax.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


class Tee:
    """Print to the console and keep a copy of everything printed."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, text):
        self.stream.write(text)
        self.buffer.write(text)

    def flush(self):
        self.stream.flush()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="validation_predictions.csv",
    )

    ap.add_argument(
        "--output",
        default=None,
        help="Optional CSV output for all thresholds",
    )

    args = ap.parse_args()

    tee = Tee(sys.stdout)
    sys.stdout = tee

    path = Path(args.input)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "group_id",
        "label",
        "score",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["group_id", "label", "score"]
    ).copy()

    df["label"] = df["label"].astype(int)
    df["score"] = df["score"].astype(float)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "label must contain only 0/1"
        )

    print("=" * 60)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 60)

    print(f"Rows/windows: {len(df)}")
    print(
        f"Subjects/groups: {df['group_id'].nunique()}"
    )

    print("\nLabels:")
    print(
        df["label"]
        .map({0: "HC", 1: "PD"})
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # 1) WINDOW LEVEL
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("1) WINDOW-LEVEL THRESHOLD")
    print("=" * 60)

    y_window = df["label"].to_numpy()
    s_window = df["score"].to_numpy()

    if len(np.unique(y_window)) < 2:
        raise ValueError(
            "Validation data must contain both HC and PD."
        )

    auc_window = roc_auc_score(
        y_window,
        s_window,
    )

    window_results = find_best_threshold(
        y_window,
        s_window,
    )

    best_window = window_results.iloc[0]

    print(f"ROC-AUC: {auc_window:.4f}")

    print("\nBest window threshold:")
    print(
        f"Threshold          : "
        f"{best_window['threshold']:.4f}"
    )
    print(
        f"Balanced accuracy  : "
        f"{best_window['balanced_accuracy']:.4f}"
    )
    print(
        f"Accuracy            : "
        f"{best_window['accuracy']:.4f}"
    )
    print(
        f"Sensitivity         : "
        f"{best_window['sensitivity']:.4f}"
    )
    print(
        f"Specificity         : "
        f"{best_window['specificity']:.4f}"
    )
    print(
        f"Precision           : "
        f"{best_window['precision']:.4f}"
    )
    print(
        f"F1                  : "
        f"{best_window['f1']:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        f"TN={int(best_window['tn'])} "
        f"FP={int(best_window['fp'])} "
        f"FN={int(best_window['fn'])} "
        f"TP={int(best_window['tp'])}"
    )

    window_plot_out = path.with_name(
        "validation_window_threshold.png"
    )

    save_threshold_plot(
        window_results,
        y_window,
        s_window,
        best_window,
        auc_window,
        window_plot_out,
        "Window level",
    )

    # ---------------------------------------------------------
    # 2) SUBJECT LEVEL
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("2) SUBJECT-LEVEL THRESHOLD")
    print("=" * 60)

    # Each subject gets ONE score:
    # mean PD score across all validation windows.
    subject_df = (
        df.groupby("group_id", as_index=False)
        .agg(
            label=("label", "first"),
            score=("score", "mean"),
            n_windows=("score", "size"),
        )
    )

    y_subject = subject_df["label"].to_numpy()
    s_subject = subject_df["score"].to_numpy()

    if len(np.unique(y_subject)) < 2:
        raise ValueError(
            "Subject-level validation data must contain "
            "both HC and PD."
        )

    auc_subject = roc_auc_score(
        y_subject,
        s_subject,
    )

    subject_results = find_best_threshold(
        y_subject,
        s_subject,
    )

    best_subject = subject_results.iloc[0]

    print(
        f"Subjects: {len(subject_df)}"
    )

    print(
        f"ROC-AUC: {auc_subject:.4f}"
    )

    print("\nBest SUBJECT threshold:")
    print(
        f"Threshold          : "
        f"{best_subject['threshold']:.4f}"
    )
    print(
        f"Balanced accuracy  : "
        f"{best_subject['balanced_accuracy']:.4f}"
    )
    print(
        f"Accuracy            : "
        f"{best_subject['accuracy']:.4f}"
    )
    print(
        f"Sensitivity         : "
        f"{best_subject['sensitivity']:.4f}"
    )
    print(
        f"Specificity         : "
        f"{best_subject['specificity']:.4f}"
    )
    print(
        f"Precision           : "
        f"{best_subject['precision']:.4f}"
    )
    print(
        f"F1                  : "
        f"{best_subject['f1']:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        f"TN={int(best_subject['tn'])} "
        f"FP={int(best_subject['fp'])} "
        f"FN={int(best_subject['fn'])} "
        f"TP={int(best_subject['tp'])}"
    )

    # ---------------------------------------------------------
    # 3) SHOW SUBJECT SCORES
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION SUBJECT SCORES")
    print("=" * 60)

    display_df = subject_df.copy()

    display_df["prediction"] = (
        display_df["score"]
        >= float(best_subject["threshold"])
    ).astype(int)

    display_df["true_label"] = (
        display_df["label"]
        .map({0: "HC", 1: "PD"})
    )

    display_df["prediction"] = (
        display_df["prediction"]
        .map({0: "HC", 1: "PD"})
    )

    print(
        display_df[
            [
                "group_id",
                "true_label",
                "prediction",
                "score",
                "n_windows",
            ]
        ]
        .sort_values("group_id")
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    if args.output:
        out = Path(args.output)
    else:
        out = path.with_name(
            "validation_threshold_results.csv"
        )

    combined = []

    window_results = window_results.copy()
    window_results["level"] = "window"

    subject_results = subject_results.copy()
    subject_results["level"] = "subject"

    combined_df = pd.concat(
        [
            window_results,
            subject_results,
        ],
        ignore_index=True,
    )

    combined_df.to_csv(
        out,
        index=False,
    )

    subject_out = path.with_name(
        "validation_subject_scores.csv"
    )

    display_df.to_csv(
        subject_out,
        index=False,
    )

    print("\n" + "=" * 60)
    print("RECOMMENDED THRESHOLD")
    print("=" * 60)

    print(
        f"{float(best_subject['threshold']):.4f}"
    )

    print(
        "\nUse this SUBJECT-level threshold for the "
        "application after TEST evaluation."
    )

    # ---------------------------------------------------------
    # SAVE SUMMARY (best thresholds, AUC, accuracy, ...)
    # ---------------------------------------------------------

    summary_rows = []

    for level, n, auc, best in [
        ("window", len(df), auc_window, best_window),
        ("subject", len(subject_df), auc_subject, best_subject),
    ]:
        row = {"level": level, "n": int(n), "roc_auc": float(auc)}
        row.update(best.to_dict())
        for k in ("tn", "fp", "fn", "tp"):
            row[k] = int(row[k])
        summary_rows.append(row)

    summary_out = path.with_name(
        "validation_threshold_summary.csv"
    )

    pd.DataFrame(summary_rows).to_csv(
        summary_out,
        index=False,
    )

    window_plot_out = path.with_name(
        "validation_window_metrics.png"
    )

    save_metrics_bar(
        best_window,
        auc_window,
        "VALIDATION window-level performance",
        window_plot_out,
    )

    report_out = path.with_name(
        "validation_threshold_report.txt"
    )

    print(f"\nSaved:")
    print(out)
    print(subject_out)
    print(summary_out)
    print(window_plot_out)
    print(report_out)
    print(window_plot_out)

    sys.stdout = tee.stream
    report_out.write_text(
        tee.buffer.getvalue(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()