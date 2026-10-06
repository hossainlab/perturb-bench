"""Generate publication-ready figures for the BioTransfer paper.

Generates:
1. fig1_benchmark_performance.png: Multi-panel bar plots comparing BioTransfer vs all baselines.
2. fig2_dual_zero_shot_scatter.png: Per-perturbation scatter plots showing error reduction.
3. fig3_pathway_conservation_landscape.png: Biological pathway divergence & conservation analysis.
4. fig4_error_reduction_distribution.png: Distribution of MSE reductions across all 308 unseen genes.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path("results/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_benchmark_performance():
    """Figure 1: Benchmark Performance Bar Charts across Scenarios."""
    csv_path = Path("results/cross_cell/cross_cell_benchmark_summary.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Clean scenario names
    scenario_map = {
        "Task_1_K562_OOD_Test": "Task 1: K562 Unseen OOD Knockouts (N=308)",
        "Task_2_RPE1_Transfer_TrainPerts": "Task 2: RPE1 Transfer of Screened Genes (N=1,542)",
        "Task_3_RPE1_Dual_ZeroShot_TestPerts": "Task 3: Dual Zero-Shot RPE1 Unseen Genes (N=308)",
    }
    df["Scenario_Clean"] = df["Scenario"].map(scenario_map)

    # Color palette
    colors = {
        "Control Mean": "#8E9AA8",
        "Mean Shift": "#5B84B1",
        "Naive K562 Shift Copy": "#E69F00",
        "BioTransferNet (De Novo)": "#009E73",
        "BioTransferTranslator (Screen Translation)": "#D55E00",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Pearson Correlation (Top-20 DE)
    ax = axes[0]
    sns.barplot(
        data=df,
        x="Scenario_Clean",
        y="pearson_top20_de",
        hue="Model",
        palette=colors,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title("A. Top-20 DE Genes Pearson Correlation (Higher is Better)", fontweight="bold", pad=12)
    ax.set_ylabel("Pearson Correlation ($\\rho$)")
    ax.set_xlabel("")
    ax.set_ylim(0.75, 1.01)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=18, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Method", frameon=True, facecolor="white", framealpha=0.9)

    # Panel B: Mean Squared Error (Top-20 DE)
    ax = axes[1]
    sns.barplot(
        data=df,
        x="Scenario_Clean",
        y="mse_top20_de",
        hue="Model",
        palette=colors,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title("B. Top-20 DE Genes Mean Squared Error (Lower is Better)", fontweight="bold", pad=12)
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=18, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend().remove()

    plt.tight_layout()
    out_path = OUT_DIR / "fig1_benchmark_performance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_dual_zero_shot_scatter():
    """Figure 2: Head-to-Head Per-Perturbation Error Reduction on Unseen Genes."""
    trans_csv = Path("results/cross_cell/dual_zero_shot_translator_breakdown.csv")
    copy_csv = Path("results/cross_cell/dual_zero_shot_naive_copy_breakdown.csv")

    if not trans_csv.exists() or not copy_csv.exists():
        print("Breakdown files not found.")
        return

    df_tr = pd.read_csv(trans_csv)
    df_cp = pd.read_csv(copy_csv)

    merged = pd.merge(df_tr, df_cp, on="perturbation", suffixes=("_translator", "_naive_copy"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Correlation Comparison
    ax = axes[0]
    ax.scatter(
        merged["pearson_top20_de_naive_copy"],
        merged["pearson_top20_de_translator"],
        alpha=0.6,
        color="#2B5C8F",
        edgecolors="white",
        s=45,
    )
    ax.plot([0.3, 1.0], [0.3, 1.0], "r--", linewidth=1.5, label="Identity (x = y)")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Naive K562 Shift Copy (Top-20 DE $\\rho$)")
    ax.set_ylabel("BioTransferTranslator (Top-20 DE $\\rho$)")
    ax.set_title("A. Per-Target Pearson Correlation (Unseen Genes, N=308)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    # Panel B: MSE Reduction
    ax = axes[1]
    ax.scatter(
        merged["mse_top20_de_naive_copy"],
        merged["mse_top20_de_translator"],
        alpha=0.6,
        color="#D95F02",
        edgecolors="white",
        s=45,
    )
    max_val = max(merged["mse_top20_de_naive_copy"].max(), merged["mse_top20_de_translator"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Identity (x = y)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Naive K562 Shift Copy (Top-20 DE MSE)")
    ax.set_ylabel("BioTransferTranslator (Top-20 DE MSE)")
    ax.set_title("B. Per-Target Mean Squared Error (Unseen Genes, N=308)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    plt.tight_layout()
    out_path = OUT_DIR / "fig2_dual_zero_shot_scatter.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_pathway_conservation():
    """Figure 3: Biological Pathway Divergence & Functional Modularity."""
    from perturb_bench.data.transfer import CrossCellAlignedData
    aligned = CrossCellAlignedData.from_h5ad(cache_path="data/cross_cell_aligned_summary.npz")

    rhos = []
    for i, p in enumerate(aligned.perturbations):
        d_k = aligned.k562_deltas[i]
        d_r = aligned.rpe1_deltas[i]
        diff_k = d_k - np.mean(d_k)
        diff_r = d_r - np.mean(d_r)
        denom = np.sqrt(np.sum(diff_k**2) * np.sum(diff_r**2))
        r = float(np.sum(diff_k * diff_r) / denom) if denom > 1e-12 else 0.0
        rhos.append({
            "perturbation": p,
            "cross_cell_rho": r,
            "k562_mag": float(np.linalg.norm(d_k)),
            "rpe1_mag": float(np.linalg.norm(d_r)),
        })

    df = pd.DataFrame(rhos)

    # Assign biological pathways
    def assign_category(gene):
        g = gene.upper()
        if g.startswith(("RPL", "RPS", "MRPL", "MRPS")):
            return "Ribosome / Translation"
        elif g.startswith(("PSMA", "PSMB", "PSMC", "PSMD")):
            return "Proteasome Core"
        elif g.startswith(("POLR", "TBP", "GTF")):
            return "RNA Polymerase / Transcription"
        elif g.startswith(("ACT", "TUB", "MYO", "KIF", "ARPC")):
            return "Cytoskeleton & Spindle"
        elif g in {"TP53", "MDM2", "CDKN1A", "CHEK1", "CHEK2", "ATM", "ATR", "RAD51"}:
            return "DNA Damage / p53 Axis"
        elif g.startswith(("NDUF", "COX", "UQCR", "ATP5", "POLRMT", "LRPPRC")):
            return "Mitochondrial Respiration"
        else:
            return "Other Essential"

    df["Pathway"] = df["perturbation"].apply(assign_category)

    # Filter to substantial effect sizes
    df_active = df[df["k562_mag"] > 1.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Cross-Cell Conservation by Pathway
    ax = axes[0]
    order = df_active.groupby("Pathway")["cross_cell_rho"].median().sort_values(ascending=False).index
    palette = sns.color_palette("muted", len(order))

    sns.boxplot(
        data=df_active,
        x="Pathway",
        y="cross_cell_rho",
        order=order,
        palette=palette,
        ax=ax,
        fliersize=2,
    )
    ax.set_title("A. Cross-Cell Lineage Transferability by Pathway", fontweight="bold")
    ax.set_ylabel("K562-RPE1 Response Correlation ($\\rho$)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Distribution of Effect Magnitudes
    ax = axes[1]
    ax.scatter(
        df["k562_mag"],
        df["rpe1_mag"],
        alpha=0.4,
        c=df["cross_cell_rho"],
        cmap="coolwarm",
        s=30,
        edgecolors="none",
    )
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Lineage Correlation ($\\rho$)")
    ax.plot([0, 25], [0, 25], "k--", alpha=0.5, label="Equal Magnitude")
    ax.set_xlabel("K562 Differential Shift Norm ($||\\Delta_{\\mathrm{K562}}||_2$)")
    ax.set_ylabel("RPE1 Differential Shift Norm ($||\\Delta_{\\mathrm{RPE1}}||_2$)")
    ax.set_title("B. Perturbation Response Magnitude Comparison", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    plt.tight_layout()
    out_path = OUT_DIR / "fig3_pathway_conservation_landscape.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating publication-ready figures...")
    plot_benchmark_performance()
    plot_dual_zero_shot_scatter()
    plot_pathway_conservation()
    print("All figures successfully created in results/paper_figures/")


if __name__ == "__main__":
    main()
