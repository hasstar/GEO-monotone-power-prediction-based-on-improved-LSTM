#!/usr/bin/env python3
"""
AWPL experiment aggregation + paper-ready visualization pipeline.

What it does
------------
1. Traverse folders like:
   AWPL_pruning_logs_<Yhis>_<Phis>_<rate>_<model>_<loss>
2. Read history_round_03.csv ... history_round_09.csv if they exist.
3. Merge them into a single long-format results table.
4. Create publication-oriented summary tables and figures.
5. Save everything under an output directory.

Usage
-----
python awpl_paper_pipeline.py \
    --root /path/to/experiment_root \
    --outdir /path/to/awpl_paper_outputs

Typical usage when the script is placed next to the AWPL_pruning_logs_* folders:
python awpl_paper_pipeline.py --root . --outdir ./AWPL_paper_outputs
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Matplotlib global style (paper friendly) ----------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (7.5, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------- Experiment schema ----------
FOLDER_RE = re.compile(
    r"^AWPL_pruning_logs_(?P<Yhis>\d+)_(?P<Phis>\d+)_(?P<rate>\d+)_(?P<model>.+)_(?P<loss>.+)$"
)
HISTORY_RE = re.compile(r"^history_round_(?P<round>\d{2})\.csv$")

# Lower is better
MINIMIZE_METRICS = [
    "val_RMSE", "val_NRMSE", "val_MAE", "val_W_Dist", "val_Tail_W_Dist", "val_KS_Dist",
    "val_Fade_RMSE", "val_Slope_RMSE",
    "tes_RMSE", "tes_NRMSE", "tes_MAE", "tes_W_Dist", "tes_Tail_W_Dist", "tes_KS_Dist",
    "tes_Fade_RMSE", "tes_Slope_RMSE",
]

# Higher is better
MAXIMIZE_METRICS = [
    "val_R2", "val_Corr", "val_Fade_Recall", "val_Fade_Precision", "val_Fade_F1",
    "tes_R2", "tes_Corr", "tes_Fade_Recall", "tes_Fade_Precision", "tes_Fade_F1",
]

CORE_TEST_METRICS = [
    "tes_RMSE", "tes_MAE", "tes_R2", "tes_Corr",
    "tes_W_Dist", "tes_Tail_W_Dist", "tes_KS_Dist",
    "tes_Fade_RMSE", "tes_Fade_Recall", "tes_Fade_Precision", "tes_Fade_F1",
    "tes_Slope_RMSE",
]

MAIN_STORY_METRICS = ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist", "tes_Slope_RMSE"]

METRIC_LABELS = {
    "tes_RMSE": "Test RMSE",
    "tes_MAE": "Test MAE",
    "tes_R2": "Test R2",
    "tes_Corr": "Test Corr",
    "tes_W_Dist": "Test Wasserstein",
    "tes_Tail_W_Dist": "Test Tail Wasserstein",
    "tes_KS_Dist": "Test KS Distance",
    "tes_Fade_RMSE": "Test Fade RMSE",
    "tes_Fade_Recall": "Test Fade Recall",
    "tes_Fade_Precision": "Test Fade Precision",
    "tes_Fade_F1": "Test Fade F1",
    "tes_Slope_RMSE": "Test Slope RMSE",
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remaining_features_from_round(round_id: int, total_initial_features: int = 10) -> int:
    return total_initial_features - round_id


def metric_direction(metric: str) -> str:
    if metric in MINIMIZE_METRICS:
        return "min"
    if metric in MAXIMIZE_METRICS:
        return "max"
    raise KeyError(f"Unknown metric direction for: {metric}")


def find_experiment_folders(root: Path) -> List[Path]:
    folders = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and FOLDER_RE.match(p.name):
            folders.append(p)
    return folders


def read_history_csvs(folder: Path, total_initial_features: int = 10) -> List[pd.DataFrame]:
    match = FOLDER_RE.match(folder.name)
    if not match:
        return []
    meta = match.groupdict()
    out = []
    for csv_path in sorted(folder.glob("history_round_*.csv")):
        m = HISTORY_RE.match(csv_path.name)
        if not m:
            continue
        round_id = int(m.group("round"))
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed reading {csv_path}: {exc}")
            continue

        df.columns = [str(c).strip() for c in df.columns]
        df["source_folder"] = folder.name
        df["source_file"] = csv_path.name
        df["Yhis"] = int(meta["Yhis"])
        df["Phis"] = int(meta["Phis"])
        df["rate"] = int(meta["rate"])
        df["model"] = meta["model"]
        df["loss"] = meta["loss"]
        df["history_round"] = round_id
        df["remaining_features"] = remaining_features_from_round(round_id, total_initial_features)
        out.append(df)
    return out


def aggregate_results(root: Path, total_initial_features: int = 10) -> pd.DataFrame:
    folders = find_experiment_folders(root)
    if not folders:
        raise FileNotFoundError(
            f"No folders matched AWPL_pruning_logs_* under: {root.resolve()}"
        )

    frames: List[pd.DataFrame] = []
    for folder in folders:
        frames.extend(read_history_csvs(folder, total_initial_features=total_initial_features))

    if not frames:
        raise FileNotFoundError("Matched experiment folders, but no history_round_*.csv files were readable.")

    df = pd.concat(frames, ignore_index=True)

    numeric_candidates = [
        c for c in df.columns
        if c not in {"source_folder", "source_file", "model", "loss"}
    ]
    for c in numeric_candidates:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Stable sort for reproducibility
    sort_cols = [c for c in ["Yhis", "Phis", "rate", "model", "loss", "remaining_features", "history_round"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def choose_best_round_per_setting(df: pd.DataFrame, score_metric: str = "tes_RMSE") -> pd.DataFrame:
    direction = metric_direction(score_metric)
    group_cols = ["Yhis", "Phis", "rate", "model", "loss"]

    work = df.dropna(subset=[score_metric]).copy()
    if direction == "min":
        idx = work.groupby(group_cols)[score_metric].idxmin()
    else:
        idx = work.groupby(group_cols)[score_metric].idxmax()
    return work.loc[idx].sort_values(group_cols).reset_index(drop=True)


def rank_within_group(series: pd.Series, direction: str) -> pd.Series:
    asc = direction == "min"
    return series.rank(method="min", ascending=asc)


def add_ranks(df: pd.DataFrame, metrics: Sequence[str], group_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        direction = metric_direction(metric)
        rank_col = f"rank_{metric}"
        out[rank_col] = out.groupby(list(group_cols))[metric].transform(lambda s: rank_within_group(s, direction))
    return out


def save_csvs(df_all: pd.DataFrame, outdir: Path) -> Dict[str, Path]:
    csv_dir = outdir / "tables" / "csv"
    safe_mkdir(csv_dir)

    paths: Dict[str, Path] = {}

    p = csv_dir / "all_results_long.csv"
    df_all.to_csv(p, index=False)
    paths["all_results_long"] = p

    # Remaining 1 feature = final pruning stage
    df_f1 = df_all[df_all["remaining_features"] == 1].copy()
    p = csv_dir / "results_remaining_1_feature.csv"
    df_f1.to_csv(p, index=False)
    paths["results_remaining_1_feature"] = p

    df_best_rmse = choose_best_round_per_setting(df_all, score_metric="tes_RMSE")
    p = csv_dir / "best_round_per_setting_by_tes_RMSE.csv"
    df_best_rmse.to_csv(p, index=False)
    paths["best_round_by_rmse"] = p

    df_best_fade = choose_best_round_per_setting(df_all, score_metric="tes_Fade_F1")
    p = csv_dir / "best_round_per_setting_by_tes_Fade_F1.csv"
    df_best_fade.to_csv(p, index=False)
    paths["best_round_by_fade_f1"] = p

    return paths


def write_markdown_table(df: pd.DataFrame, path: Path, float_digits: int = 4) -> None:
    tmp = df.copy()
    for c in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[c]):
            tmp[c] = tmp[c].map(lambda x: f"{x:.{float_digits}f}" if pd.notna(x) else "")
    path.write_text(tmp.to_markdown(index=False), encoding="utf-8")


def table_main_best_models(df: pd.DataFrame, outdir: Path) -> None:
    """
    Main paper table:
    for each (Yhis, rate, Phis), compare the best model/loss combinations at final pruning stage.
    We keep remaining_features == 1 because that matches your final pruning target.
    """
    table_dir = outdir / "tables" / "paper_tables"
    safe_mkdir(table_dir)

    work = df[df["remaining_features"] == 1].copy()
    if work.empty:
        return

    metrics = ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist", "tes_Slope_RMSE"]
    group_cols = ["Yhis", "rate", "Phis"]
    ranked = add_ranks(work, metrics, group_cols)

    cols = ["Yhis", "rate", "Phis", "model", "loss"] + metrics + [f"rank_{m}" for m in metrics]
    out = ranked[cols].sort_values(group_cols + ["rank_tes_RMSE", "rank_tes_Fade_F1"]).reset_index(drop=True)

    out_csv = table_dir / "table_main_final_feature1_all_rows.csv"
    out.to_csv(out_csv, index=False)

    # Compact version: top 3 rows per (Yhis, rate, Phis) using combined rank.
    compact = out.copy()
    compact["rank_sum"] = compact[[f"rank_{m}" for m in metrics]].sum(axis=1)
    compact = compact.sort_values(group_cols + ["rank_sum", "rank_tes_RMSE", "rank_tes_Fade_F1"])
    compact = compact.groupby(group_cols).head(3).reset_index(drop=True)

    compact_csv = table_dir / "table_main_final_feature1_top3_per_phis.csv"
    compact.to_csv(compact_csv, index=False)
    write_markdown_table(compact, table_dir / "table_main_final_feature1_top3_per_phis.md")


def table_loss_comparison(df: pd.DataFrame, outdir: Path) -> None:
    table_dir = outdir / "tables" / "paper_tables"
    safe_mkdir(table_dir)

    # Use best round (by RMSE) for each model/loss/setting to avoid bias from one pruning stage only.
    work = choose_best_round_per_setting(df, score_metric="tes_RMSE")
    if work.empty:
        return

    group_cols = ["Yhis", "Phis", "rate", "model"]
    metrics = ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist"]
    ranked = add_ranks(work, metrics, group_cols)

    cols = ["Yhis", "Phis", "rate", "model", "loss", "remaining_features"] + metrics + [f"rank_{m}" for m in metrics]
    out = ranked[cols].sort_values(group_cols + ["rank_tes_RMSE", "rank_tes_Fade_F1"])
    out.to_csv(table_dir / "table_loss_comparison_best_round.csv", index=False)


def table_ablation_pruning(df: pd.DataFrame, outdir: Path) -> None:
    table_dir = outdir / "tables" / "paper_tables"
    safe_mkdir(table_dir)

    metrics = ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist", "tes_Slope_RMSE"]
    cols = ["Yhis", "Phis", "rate", "model", "loss", "remaining_features"] + metrics
    work = df[cols].copy().sort_values(["Yhis", "Phis", "rate", "model", "loss", "remaining_features"])
    work.to_csv(table_dir / "table_pruning_curve_values.csv", index=False)


def plot_line_by_phis(
    data: pd.DataFrame,
    metric: str,
    hue_col: str,
    title: str,
    save_path: Path,
    style_col: Optional[str] = None,
) -> None:
    if data.empty or metric not in data.columns:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    data = data.sort_values([hue_col, "Phis"])

    hue_values = list(dict.fromkeys(data[hue_col].tolist()))
    style_values = list(dict.fromkeys(data[style_col].tolist())) if style_col else [None]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":"]

    color_map = plt.cm.get_cmap("tab10", max(3, len(hue_values)))

    for i, hv in enumerate(hue_values):
        sub_h = data[data[hue_col] == hv]
        if style_col is None:
            ax.plot(sub_h["Phis"], sub_h[metric], marker=markers[i % len(markers)], color=color_map(i), label=str(hv))
        else:
            for j, sv in enumerate(style_values):
                sub = sub_h[sub_h[style_col] == sv]
                if sub.empty:
                    continue
                label = f"{hv} | {style_col}={sv}"
                ax.plot(
                    sub["Phis"], sub[metric],
                    marker=markers[i % len(markers)],
                    linestyle=linestyles[j % len(linestyles)],
                    color=color_map(i),
                    label=label,
                )

    ax.set_xlabel("Prediction Length Phis")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_pruning_curve(data: pd.DataFrame, metric: str, title: str, save_path: Path) -> None:
    if data.empty or metric not in data.columns:
        return

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    data = data.sort_values(["model", "loss", "remaining_features"])
    groups = data.groupby(["model", "loss"])
    color_map = plt.cm.get_cmap("tab20", max(3, len(groups)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, ((model, loss), sub) in enumerate(groups):
        ax.plot(
            sub["remaining_features"], sub[metric],
            marker=markers[i % len(markers)],
            color=color_map(i),
            label=f"{model} | {loss}",
        )

    ax.set_xlabel("Remaining atmospheric features")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title)
    ax.invert_xaxis()  # from many features to few features, visually intuitive for pruning
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, title: str, save_path: Path) -> None:
    if df.empty or values not in df.columns:
        return
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(pivot.columns) + 2), max(4.2, 0.5 * len(pivot.index) + 2)))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(METRIC_LABELS.get(values, values), rotation=90)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_rank_bar(df: pd.DataFrame, group_cols: Sequence[str], score_cols: Sequence[str], save_path: Path, title: str) -> None:
    if df.empty:
        return
    tmp = df.copy()
    tmp["composite_rank_score"] = tmp[list(score_cols)].sum(axis=1)
    summary = tmp.groupby("model")["composite_rank_score"].mean().sort_values()
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(summary.index.astype(str), summary.values)
    ax.set_ylabel("Average composite rank (lower is better)")
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(df_all: pd.DataFrame, outdir: Path) -> None:
    fig_dir = outdir / "figures"
    safe_mkdir(fig_dir)

    # ---------- A. Main-story line plots at final pruning stage ----------
    final_df = df_all[df_all["remaining_features"] == 1].copy()
    if not final_df.empty:
        for (yhis, rate, loss), sub in final_df.groupby(["Yhis", "rate", "loss"]):
            for metric in ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist"]:
                save_path = fig_dir / f"main_line_models_Y{yhis}_R{rate}_L{loss}_{metric}.png"
                title = f"Models vs Phis | Yhis={yhis}, rate={rate}, loss={loss}, remaining=1"
                plot_line_by_phis(sub, metric=metric, hue_col="model", title=title, save_path=save_path)

        for (yhis, rate, model), sub in final_df.groupby(["Yhis", "rate", "model"]):
            for metric in ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist"]:
                save_path = fig_dir / f"main_line_losses_Y{yhis}_R{rate}_M{model}_{metric}.png"
                title = f"Losses vs Phis | Yhis={yhis}, rate={rate}, model={model}, remaining=1"
                plot_line_by_phis(sub, metric=metric, hue_col="loss", title=title, save_path=save_path)

    # ---------- B. Pruning curves per setting ----------
    for (yhis, phis, rate), sub in df_all.groupby(["Yhis", "Phis", "rate"]):
        for metric in ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist"]:
            save_path = fig_dir / f"pruning_curve_Y{yhis}_P{phis}_R{rate}_{metric}.png"
            title = f"Pruning curves | Yhis={yhis}, Phis={phis}, rate={rate}"
            plot_pruning_curve(sub, metric=metric, title=title, save_path=save_path)

    # ---------- C. Heatmaps for model-loss combinations ----------
    # Use best round by RMSE per setting/model/loss, then heatmap model x loss.
    best_rmse_df = choose_best_round_per_setting(df_all, score_metric="tes_RMSE")
    for (yhis, phis, rate), sub in best_rmse_df.groupby(["Yhis", "Phis", "rate"]):
        for metric in ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist"]:
            save_path = fig_dir / f"heatmap_model_loss_Y{yhis}_P{phis}_R{rate}_{metric}.png"
            title = f"Model-Loss heatmap (best pruning round by RMSE) | Yhis={yhis}, Phis={phis}, rate={rate}"
            plot_heatmap(sub, index="model", columns="loss", values=metric, title=title, save_path=save_path)

    # ---------- D. Yhis-Phis heatmaps after selecting the best model/loss per grid ----------
    for metric in ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist"]:
        direction = metric_direction(metric)
        if direction == "min":
            idx = best_rmse_df.groupby(["Yhis", "Phis", "rate"])[metric].idxmin()
        else:
            idx = best_rmse_df.groupby(["Yhis", "Phis", "rate"])[metric].idxmax()
        best_grid = best_rmse_df.loc[idx].copy()
        for rate, sub in best_grid.groupby("rate"):
            save_path = fig_dir / f"heatmap_Yhis_Phis_best_overall_R{rate}_{metric}.png"
            title = f"Best overall over (model, loss) | rate={rate}"
            plot_heatmap(sub, index="Yhis", columns="Phis", values=metric, title=title, save_path=save_path)

    # ---------- E. Composite ranking bar plot ----------
    if not final_df.empty:
        ranked = add_ranks(final_df, ["tes_RMSE", "tes_Fade_F1", "tes_W_Dist"], ["Yhis", "Phis", "rate", "loss"])
        plot_rank_bar(
            ranked,
            group_cols=["Yhis", "Phis", "rate", "loss"],
            score_cols=["rank_tes_RMSE", "rank_tes_Fade_F1", "rank_tes_W_Dist"],
            save_path=fig_dir / "bar_model_average_composite_rank_final_feature1.png",
            title="Average model rank across settings (remaining feature = 1)",
        )


def build_report(df_all: pd.DataFrame, outdir: Path) -> None:
    report_path = outdir / "REPORT.md"
    safe_mkdir(outdir)

    n_folders = df_all["source_folder"].nunique()
    n_rows = len(df_all)
    models = sorted(df_all["model"].dropna().unique().tolist())
    losses = sorted(df_all["loss"].dropna().unique().tolist())
    yhis_vals = sorted(df_all["Yhis"].dropna().unique().tolist())
    phis_vals = sorted(df_all["Phis"].dropna().unique().tolist())
    rates = sorted(df_all["rate"].dropna().unique().tolist())
    rem = sorted(df_all["remaining_features"].dropna().unique().tolist())

    best_rmse = choose_best_round_per_setting(df_all, score_metric="tes_RMSE")
    if best_rmse.empty:
        top_lines = ["No best-round summary available."]
    else:
        global_best = best_rmse.nsmallest(10, "tes_RMSE")[
            ["Yhis", "Phis", "rate", "model", "loss", "remaining_features", "tes_RMSE", "tes_Fade_F1", "tes_W_Dist"]
        ]
        top_lines = [global_best.to_markdown(index=False)]

    text = f"""# AWPL Paper Output Report

## Data coverage
- Experiment folders found: {n_folders}
- Total history rows merged: {n_rows}
- Yhis values: {yhis_vals}
- Phis values: {phis_vals}
- rate values: {rates}
- models: {models}
- losses: {losses}
- remaining atmospheric features: {rem}

## Recommended paper usage
- Main text: use figures under `figures/` beginning with `main_line_`, `pruning_curve_`, and `heatmap_model_loss_`.
- Main tables: use files under `tables/paper_tables/`.
- Appendix: use `tables/csv/all_results_long.csv` and all remaining heatmaps.

## Top settings by test RMSE (best pruning round per setting)
{top_lines[0]}
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate AWPL pruning logs and generate paper-ready tables/figures.")
    parser.add_argument("--root", type=str, required=True, help="Directory containing AWPL_pruning_logs_* folders.")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save generated tables/figures/report.")
    parser.add_argument("--total-initial-features", type=int, default=10, help="Initial number of atmospheric features before pruning. Default: 10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    safe_mkdir(outdir)

    df_all = aggregate_results(root, total_initial_features=args.total_initial_features)
    save_csvs(df_all, outdir)
    table_main_best_models(df_all, outdir)
    table_loss_comparison(df_all, outdir)
    table_ablation_pruning(df_all, outdir)
    generate_figures(df_all, outdir)
    build_report(df_all, outdir)

    print(f"[OK] Outputs written to: {outdir}")
    print(f"[OK] Aggregated rows: {len(df_all)}")
    print(f"[OK] Unique experiment folders: {df_all['source_folder'].nunique()}")


if __name__ == "__main__":
    main()
# python Main_AWPL_PlotFig.py \
#     --root . \
#     --outdir ./AWPL_paper_outputs