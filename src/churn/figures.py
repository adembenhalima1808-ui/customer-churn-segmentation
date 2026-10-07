"""Report figures: elbow/silhouette, segment profiles, churn model results.

Color is assigned by the job it does, not by taste:
- categorical identity (segments) uses a fixed hue order, never reused per-run
- magnitude (feature importance, confusion counts) uses one hue, light to dark
- every bar carries a direct label, since several of these hues sit under
  3:1 contrast against a white page and cannot be read by color alone
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order (first 4 slots of the validated 8-hue palette).
SEGMENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQUENTIAL_BLUE = "#2a78d6"
SEQUENTIAL_BLUE_LIGHT = "#9ec5f4"

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "font.size": 10,
    }
)


def elbow_silhouette_figure(sweep, chosen_k: int, out_path: Path) -> Path:
    """Two single-measure panels, never a dual-axis chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax1.plot(sweep.k_values, sweep.inertia, color=SEQUENTIAL_BLUE, marker="o", markersize=6)
    ax1.set_title("Inertia by k")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (within-cluster SSE)")

    ax2.plot(sweep.k_values, sweep.silhouette, color=SEQUENTIAL_BLUE, marker="o", markersize=6)
    chosen_idx = sweep.k_values.index(chosen_k)
    ax2.scatter(
        [chosen_k], [sweep.silhouette[chosen_idx]],
        color="#eb6834", s=70, zorder=5, label=f"chosen k={chosen_k}",
    )
    ax2.annotate(
        f"{sweep.silhouette[chosen_idx]:.2f}",
        (chosen_k, sweep.silhouette[chosen_idx]),
        textcoords="offset points", xytext=(8, 6),
    )
    ax2.set_title("Silhouette score by k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def segment_profile_figure(profile: pd.DataFrame, labels: dict, out_path: Path) -> Path:
    """One bar per segment for tenure and monthly spend, fixed categorical colors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    segments = list(profile.index)
    names = [labels.get(s, str(s)) for s in segments]
    colors = [SEGMENT_COLORS[i % len(SEGMENT_COLORS)] for i in range(len(segments))]

    bars1 = ax1.bar(names, profile["tenure"], color=colors)
    ax1.set_title("Average tenure (months)")
    ax1.bar_label(bars1, fmt="%.0f", padding=3)
    ax1.tick_params(axis="x", rotation=20)

    bars2 = ax2.bar(names, profile["MonthlyCharges"], color=colors)
    ax2.set_title("Average monthly charges ($)")
    ax2.bar_label(bars2, fmt="%.0f", padding=3)
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def feature_importance_figure(importances: pd.Series, out_path: Path, top_n: int = 8) -> Path:
    """Horizontal bars, single hue: one series, so color carries no identity."""
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(top.index, top.values, color=SEQUENTIAL_BLUE)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Top churn drivers — random forest")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def confusion_matrix_figure(y_test, y_pred, out_path: Path) -> Path:
    """2x2 sequential heatmap with every cell labeled, never color-only."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actual stay", "Actual churn"])
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            text_color = "white" if value > cm.max() / 2 else "#0b0b0b"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=12)
    ax.set_title("Random forest confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
