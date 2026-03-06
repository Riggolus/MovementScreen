"""Report generation: text summary and matplotlib visualisations."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

from movementscreen.analysis.aggregator import TrialResult, TrialStats
from movementscreen.analysis.compensation import Severity


# Severity colour mapping for plots
_SEV_COLORS = {
    Severity.NONE: "#2ecc71",
    Severity.MILD: "#f1c40f",
    Severity.MODERATE: "#e67e22",
    Severity.SEVERE: "#e74c3c",
}


def print_report(result: TrialResult) -> None:
    """Print a formatted text report to stdout."""
    print(result.summary())


def save_text_report(result: TrialResult, output_path: str | Path) -> None:
    """Write the text report to a file."""
    Path(output_path).write_text(result.summary(), encoding="utf-8")


def plot_angle_ranges(result: TrialResult, output_path: str | Path | None = None) -> None:
    """Bar chart of joint angle ranges (min/mean/max) across the trial.

    If *output_path* is given the figure is saved to disk; otherwise it is shown.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        raise RuntimeError("matplotlib is not installed. Run: pip install matplotlib")

    stats: list[TrialStats] = [s for s in result.stats.values() if s.mean is not None]
    if not stats:
        print("No angle data to plot.")
        return

    labels = [s.name for s in stats]
    means = [s.mean for s in stats]
    mins = [s.min for s in stats]
    maxs = [s.max for s in stats]
    errors_low = [m - mn for m, mn in zip(means, mins)]
    errors_high = [mx - m for mx, m in zip(maxs, means)]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(x, means, color="#3498db", alpha=0.8, label="Mean")
    ax.errorbar(x, means, yerr=[errors_low, errors_high], fmt="none", color="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Angle (degrees)")
    ax.set_title(f"{result.screen_name} — Joint Angle Ranges")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_compensation_summary(result: TrialResult, output_path: str | Path | None = None) -> None:
    """Horizontal bar chart of compensation findings coloured by severity."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is not installed.")

    findings = result.compensation_report.findings
    if not findings:
        print("No compensations to plot.")
        return

    labels = [f.name for f in findings]
    severities = [f.severity for f in findings]
    colors = [_SEV_COLORS[s] for s in severities]
    # Use metric value as bar length when available, else use severity rank
    _rank = {Severity.NONE: 0, Severity.MILD: 1, Severity.MODERATE: 2, Severity.SEVERE: 3}
    values = [_rank[s] for s in severities]

    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5 + 1)))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 3.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Mild", "Moderate", "Severe"])
    ax.set_title(f"{result.screen_name} — Compensation Findings")

    legend_handles = [
        mpatches.Patch(color=_SEV_COLORS[s], label=s.value.capitalize())
        for s in (Severity.MILD, Severity.MODERATE, Severity.SEVERE)
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close(fig)
