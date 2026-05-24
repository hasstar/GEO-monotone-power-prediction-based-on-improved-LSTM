# -*- coding: utf-8 -*-
"""
Optimized Publication-ready Plotting Script
- Refined Color Schemes (Nature-style palettes)
- Enhanced Geometry visualization (Grid masks)
- Standardized Font hierarchies
- Professional Figure Spacing
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
# =========================
# 0. Global config & RC Params
# =========================
CSV_PATH = "all_results_long_filtered.csv"
OUT_DIR = Path("./paper_figures_optimized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOCUS_LOSS = "W2"
METRIC = "tes_KS_Dist"
VALID_RATES = [10, 30, 60]
USE_FULL_FEATURE_ONLY = True

ABLATION_MODELS = ["Fusion", "NoPosEnc", "NoTempAttn", "SingleCross", "NoVertical"]
COMPARISON_MODELS = ["Fusion", "AGLSTM", "SimpleLSTM", "FiLMLSTM"]

Y_ORDER = [100, 300, 900]
P_ORDER = [100, 300, 900, 1500, 2100, 2700]

# Standard Journal Font Sizes
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 10,
    "axes.titlesize": 18,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "axes.linewidth": 0.75,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Color Palettes
MODEL_COLORS = {
    "Fusion": "#2C3E50",  # Dark Navy
    "AGLSTM": "#2980B9",  # Blue
    "SimpleLSTM": "#E67E22",  # Orange
    "FiLMLSTM": "#27AE60",  # Green
    "NoPosEnc": "#95A5A6",
    "NoTempAttn": "#8E44AD",
    "SingleCross": "#C0392B",
    "NoVertical": "#16A085",
}

RATE_COLORS = {10: "#4E79A7", 30: "#F28E2B", 60: "#59A14F"}


# =========================
# 1. Helpers & Utils
# =========================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df[METRIC].notna()]
    df = df[df["Phis"] >= df["Yhis"]]
    df = df[df["rate"].isin(VALID_RATES)]
    if USE_FULL_FEATURE_ONLY:
        max_feat = df["remaining_features"].max()
        df = df[df["remaining_features"] == max_feat]
    df["Yhis"] = pd.Categorical(df["Yhis"], categories=Y_ORDER, ordered=True)
    df["Phis"] = pd.Categorical(df["Phis"], categories=P_ORDER, ordered=True)
    return df


def style_axis(ax):
    ax.tick_params(axis="both", which="major", length=3, width=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_masked_heatmap(ax, mat, vmin, vmax, cmap="RdBu_r", label_fmt=".3f"):
    display = np.ma.masked_invalid(mat)
    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Drawing the "Invalid" geometry mask with refined hatch
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F5F5F5",
                                       edgecolor="#DCDCDC", linewidth=0.5, hatch='////'))
            else:
                color = "white" if abs(mat[i, j]) > (vmax * 0.7) else "black"
                ax.text(j, i, f"{mat[i, j]:{label_fmt}}", ha="center", va="center",
                        fontsize=7.5, color=color, fontweight='medium')

    ax.set_xticks(np.arange(len(P_ORDER)))
    ax.set_xticklabels(P_ORDER)
    ax.set_yticks(np.arange(len(Y_ORDER)))
    ax.set_yticklabels(Y_ORDER)
    ax.set_xlabel("$Prediction / s$")
    ax.set_ylabel("$Input / s$")
    style_axis(ax)
    return im


def add_panel_caption(ax, text, y=-0.22):
    ax.set_title(text, y=y, pad=-14, fontsize=14, fontweight='bold')


# =========================
# 2. Figure 3: Ablation
# =========================
def build_ablation_figure(df: pd.DataFrame, out_dir: Path):
    sub_df = df[df["model"].isin(ABLATION_MODELS) & (df["loss"] == FOCUS_LOSS)].copy()
    fusion = sub_df[sub_df["model"] == "Fusion"].rename(columns={METRIC: "fusion_metric"})
    merged = sub_df.merge(fusion[["Yhis", "Phis", "rate", "fusion_metric"]], on=["Yhis", "Phis", "rate"])
    merged["delta"] = merged[METRIC] - merged["fusion_metric"]

    ablations = ["NoVertical", "SingleCross", "NoTempAttn", "NoPosEnc"]
    vmax = np.nanpercentile(np.abs(merged[merged["model"].isin(ablations)]["delta"]), 98)
    vmin = -vmax

    fig = plt.figure(figsize=(16, 9.5))
    outer = gridspec.GridSpec(1, 2, width_ratios=[4.2, 1.2], wspace=0.18)
    left_gs = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=outer[0], wspace=0.25, hspace=0.35)

    panel_labels = list("abcdefghijklmn")
    last_im = None

    for r_i, rate in enumerate(VALID_RATES):
        for c_i, model in enumerate(ablations):
            ax = fig.add_subplot(left_gs[r_i, c_i])
            sub = merged[(merged["rate"] == rate) & (merged["model"] == model)]
            mat = np.full((len(Y_ORDER), len(P_ORDER)), np.nan)
            for yi, y in enumerate(Y_ORDER):
                for pi, p in enumerate(P_ORDER):
                    val = sub[(sub["Yhis"] == y) & (sub["Phis"] == p)]["delta"]
                    if not val.empty: mat[yi, pi] = val.mean()

            last_im = draw_masked_heatmap(ax, mat, vmin, vmax, cmap="RdBu_r")
            add_panel_caption(ax, f"({panel_labels[r_i * 4 + c_i]}) {model} (rate={rate})")

    # Right distribution panel
    ax_sum = fig.add_subplot(outer[1])
    positions = np.arange(1, len(ablations) + 1)

    bp = ax_sum.boxplot([merged[merged["model"] == m]["delta"].dropna() for m in ablations],
                        positions=positions, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5),
                        boxprops=dict(facecolor='#FFFFFF', edgecolor='#333333', alpha=0.7, linewidth=0.8),
                        zorder=2)
    for i, model in enumerate(ablations, start=1):
        for rate in VALID_RATES:
            vals = merged[(merged["model"] == model) & (merged["rate"] == rate)]["delta"].dropna()
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax_sum.scatter(np.full(len(vals), i) + jitter, vals, s=50, alpha=0.98,
                           c=RATE_COLORS[rate], edgecolors='none', zorder=3)


    ax_sum.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax_sum.set_xticks(positions)
    ax_sum.set_xticklabels(ablations, rotation=25, ha="right")
    ax_sum.set_ylabel(r"$\Delta$KS (Ablation - Fusion)")
    style_axis(ax_sum)
    add_panel_caption(ax_sum, f"({panel_labels[12]}) Total Delta Distribution", y=-0.06)

    # Colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.02, 0.5, 0.015])
    cb = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
    cb.set_label(r"Performance Gap ($\Delta$KS)", labelpad=-1, fontsize=14)

    # --- 添加图例标注的代码开始 ---
    # 1. 创建图例手柄 (基于你定义的 RATE_COLORS)
    # marker='o' 表示圆点，color='w' 是为了隐藏连线，markerfacecolor 是实心颜色
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'rate = {r}',
               markerfacecolor=color, markersize=16)
        for r, color in RATE_COLORS.items()
    ]
    # 2. 将图例添加到右侧子图 ax_sum
    # loc='upper left' 将其放在左上角。你也可以微调 bbox_to_anchor 来精确控制位置
    ax_sum.legend(handles=legend_elements, loc='upper right',fontsize=16,
                  frameon=False, handletextpad=0.5, borderpad=1)
    # --- 添加图例标注的代码结束 ---

    fig.savefig(out_dir / "Fig3_Ablation_Standardized.png", dpi=400, bbox_inches='tight')
    fig.savefig(out_dir / "Fig3_Ablation_Standardized.pdf", dpi=400, bbox_inches='tight')
    plt.close(fig)


# =========================
# 3. Figure 4: Comparison
# =========================
def build_comparison_figure(df: pd.DataFrame, out_dir: Path):
    comp_df = df[df["model"].isin(COMPARISON_MODELS) & (df["loss"] == FOCUS_LOSS)].copy()
    fig = plt.figure(figsize=(15, 9.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.9], hspace=0.35, wspace=0.3)

    winner_order = ["Fusion", "AGLSTM", "SimpleLSTM", "FiLMLSTM"]
    cmap_win = ListedColormap([MODEL_COLORS[m] for m in winner_order])
    norm_win = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap_win.N)

    # Top: Winner Maps
    for i, rate in enumerate(VALID_RATES):
        ax = fig.add_subplot(gs[0, i])
        pivot = comp_df[comp_df["rate"] == rate].pivot_table(index=["Yhis", "Phis"], columns="model", values=METRIC)

        mat = np.full((len(Y_ORDER), len(P_ORDER)), np.nan)
        for yi, y in enumerate(Y_ORDER):
            for pi, p in enumerate(P_ORDER):
                if p >= y and (y, p) in pivot.index:
                    mat[yi, pi] = winner_order.index(pivot.loc[(y, p)].idxmin())

        display = np.ma.masked_invalid(mat)
        ax.imshow(display, cmap=cmap_win, norm=norm_win, aspect="auto")

        # Overlay short names
        for yi in range(len(Y_ORDER)):
            for pi in range(len(P_ORDER)):
                if not np.isnan(mat[yi, pi]):
                    m_name = winner_order[int(mat[yi, pi])]
                    ax.text(pi, yi, m_name[:2], ha="center", va="center", fontsize=16,
                            color="white" if m_name == "Fusion" else "black", fontweight='bold')

        ax.set_xticks(range(len(P_ORDER)));
        ax.set_xticklabels(P_ORDER)
        ax.set_yticks(range(len(Y_ORDER)));
        ax.set_yticklabels(Y_ORDER)
        ax.set_xlabel("$Prediction / s$")
        ax.set_ylabel("$Input / s$")
        style_axis(ax)
        add_panel_caption(ax, f"({list('abcdef')[i]}) Winner Map (rate={rate})")

    # Bottom Left: Pairwise Win-rate
    ax_win = fig.add_subplot(gs[1, 0])
    pivot_all = comp_df.pivot_table(index=["Yhis", "Phis", "rate"], columns="model", values=METRIC)
    pair_mat = np.array([[(pivot_all[mi] < pivot_all[mj]).mean() if mi != mj else 0.5
                          for mj in COMPARISON_MODELS] for mi in COMPARISON_MODELS])

    im_pair = ax_win.imshow(pair_mat, cmap="GnBu", vmin=0.3, vmax=0.9)
    ax_win.set_xticks(range(4));
    ax_win.set_xticklabels(COMPARISON_MODELS, rotation=30, ha='right')
    ax_win.set_yticks(range(4));
    ax_win.set_yticklabels(COMPARISON_MODELS)
    for (i, j), z in np.ndenumerate(pair_mat):
        ax_win.text(j, i, f'{z:.2f}', ha='center', va='center', fontsize=16, color="white" if z > 0.75 else "black")
    add_panel_caption(ax_win, "(d) Pairwise Win-rate Matrix")

    # Bottom Middle: Delta Boxplot
    ax_delta = fig.add_subplot(gs[1, 1])
    fusion_vals = comp_df[comp_df["model"] == "Fusion"].set_index(["Yhis", "Phis", "rate"])[METRIC]
    for i, m in enumerate(["AGLSTM", "SimpleLSTM", "FiLMLSTM"], 1):
        delta = comp_df[comp_df["model"] == m].set_index(["Yhis", "Phis", "rate"])[METRIC] - fusion_vals
        ax_delta.boxplot(delta.dropna(), positions=[i], widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor=MODEL_COLORS[m], alpha=0.3), showfliers=False)
        for r in VALID_RATES:
            r_delta = delta.xs(r, level='rate').dropna()
            ax_delta.scatter(np.full(len(r_delta), i) + np.random.normal(0, 0.03, len(r_delta)), r_delta,
                             s=10, color=RATE_COLORS[r], alpha=0.5)
    ax_delta.set_xticks([1, 2, 3]);
    ax_delta.set_xticklabels(["AG", "SL", "Fi"])
    ax_delta.axhline(0, color='black', lw=0.8, ls='--')
    ax_delta.set_ylabel(r"$\Delta$KS to Fusion")
    add_panel_caption(ax_delta, "(e) Penalty vs. Fusion")

    # --- 为 Fig 4 (e) 添加图例的代码 ---
    # 1. 创建基于 RATE_COLORS 的图例手柄
    from matplotlib.lines import Line2D
    rate_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'rate = {r}',
               markerfacecolor=color, markersize=10)
        for r, color in RATE_COLORS.items()
    ]

    # 2. 将图例添加到 ax_delta
    # loc='upper left' 通常比较合适，因为 penalty 通常是正值，左上角相对较空
    ax_delta.legend(handles=rate_legend_elements, loc='upper left',
                    frameon=False, handletextpad=0.3, borderpad=0.5)

    # Bottom Right: Horizon Curves
    ax_curve = fig.add_subplot(gs[1, 2])
    curve_data = comp_df.groupby(["model", "Phis"])[METRIC].agg(["mean", "std"]).reset_index()
    for m in COMPARISON_MODELS:
        sub = curve_data[curve_data["model"] == m]
        ax_curve.plot(sub["Phis"], sub["mean"], marker='o', markersize=4, label=m, color=MODEL_COLORS[m])
        ax_curve.fill_between(sub["Phis"].astype(float), sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                              color=MODEL_COLORS[m], alpha=0.1)
    ax_curve.set_xlabel("$Prediction / s$");
    ax_curve.set_ylabel("Mean KS")
    ax_curve.legend(loc='upper left', ncol=1, fontsize=8)
    add_panel_caption(ax_curve, "(f) Horizon Sensitivity")

    fig.savefig(out_dir / "Fig4_Comparison_Standardized.png", dpi=400, bbox_inches='tight')
    fig.savefig(out_dir / "Fig4_Comparison_Standardized.pdf", dpi=400, bbox_inches='tight')
    plt.close(fig)


# =========================
# 4. Entry Point
# =========================
if __name__ == "__main__":
    data = load_data(CSV_PATH)
    build_ablation_figure(data, OUT_DIR)
    build_comparison_figure(data, OUT_DIR)
    print(f"Optimized figures saved to {OUT_DIR.resolve()}")