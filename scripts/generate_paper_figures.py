"""Generate high-quality, publication-ready figures for the BioTransfer paper.

Adheres strictly to Nature / Cell / Science figure formatting standards:
- Subplot panel labels are bold lowercase: 'a', 'b', 'c' (not uppercase).
- Clean tick locators and no overlapping text or matplotlib warnings.
- Okabe-Ito accessible color palettes and consistent 300 DPI rendering.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Configure publication typography and aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
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


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.08):
    """Add bold lowercase panel label (e.g., 'a', 'b') in Nature journal style."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_fig1_benchmark_performance():
    """Figure 1: Cross-Cell Transfer Benchmark Performance."""
    csv_path = Path("results/cross_cell/cross_cell_benchmark_summary.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    scenarios = [
        "Task_1_K562_OOD_Test",
        "Task_2_RPE1_Transfer_TrainPerts",
        "Task_3_RPE1_Dual_ZeroShot_TestPerts",
    ]
    scenario_labels = [
        "Task 1: K562 Unseen OOD\nKnockouts (N=308)",
        "Task 2: RPE1 Transfer of\nScreened Genes (N=1,542)",
        "Task 3: Dual Zero-Shot\nRPE1 Unseen Genes (N=308)",
    ]

    model_display_map = {
        "Control Mean": "Control Mean",
        "Mean Shift": "Mean Shift",
        "BioTransferNet (De Novo)": "BioTransferNet (Ours)",
        "Naive K562 Shift Copy": "Naive Shift Copy",
        "BioTransferTranslator (Screen Translation)": "BioTransferTranslator (Ours)",
    }
    df["Model_Clean"] = df["Model"].map(model_display_map)

    all_models = [
        "Control Mean",
        "Mean Shift",
        "BioTransferNet (Ours)",
        "Naive Shift Copy",
        "BioTransferTranslator (Ours)",
    ]

    palette = {
        "Control Mean": "#8E9AA8",
        "Mean Shift": "#5B84B1",
        "BioTransferNet (Ours)": "#009E73",
        "Naive Shift Copy": "#E69F00",
        "BioTransferTranslator (Ours)": "#D55E00",
    }

    models_per_task = {
        "Task_1_K562_OOD_Test": ["Control Mean", "Mean Shift", "BioTransferNet (Ours)"],
        "Task_2_RPE1_Transfer_TrainPerts": [
            "Control Mean",
            "Mean Shift",
            "BioTransferNet (Ours)",
            "Naive Shift Copy",
            "BioTransferTranslator (Ours)",
        ],
        "Task_3_RPE1_Dual_ZeroShot_TestPerts": [
            "Control Mean",
            "Mean Shift",
            "BioTransferNet (Ours)",
            "Naive Shift Copy",
            "BioTransferTranslator (Ours)",
        ],
    }

    bar_width = 0.135
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))

    metrics = [
        ("pearson_top20_de", "Top-20 DE genes Pearson correlation", "Pearson correlation ($\\rho$)", (0.75, 1.01)),
        ("mse_top20_de", "Top-20 DE genes mean squared error", "Mean squared error (MSE)", (0.0, 0.58)),
    ]

    for ax_idx, (col, title, ylabel, ylim) in enumerate(metrics):
        ax = axes[ax_idx]
        add_panel_label(ax, "a" if ax_idx == 0 else "b", x=-0.08, y=1.06)

        for s_idx, s_key in enumerate(scenarios):
            m_list = models_per_task[s_key]
            n_bars = len(m_list)
            offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * (bar_width + 0.01)

            sub_df = df[df["Scenario"] == s_key].set_index("Model_Clean")
            for m_i, m_name in enumerate(m_list):
                if m_name in sub_df.index:
                    val = float(sub_df.loc[m_name, col])
                    ax.bar(
                        s_idx + offsets[m_i],
                        val,
                        width=bar_width,
                        color=palette[m_name],
                        edgecolor="black",
                        linewidth=0.7,
                    )

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenario_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12, fontweight="medium")
        ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[m], edgecolor="black", linewidth=0.7, label=m)
        for m in all_models
    ]

    fig.legend(
        handles=legend_elements,
        labels=all_models,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#C0C0C0",
        framealpha=1.0,
        fontsize=11,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.6,
        borderpad=0.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = OUT_DIR / "fig1_benchmark_performance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig2_dual_zero_shot_scatter():
    """Figure 2: Per-Target Head-to-Head Error Reduction on Unseen Genes."""
    trans_csv = Path("results/cross_cell/dual_zero_shot_translator_breakdown.csv")
    copy_csv = Path("results/cross_cell/dual_zero_shot_naive_copy_breakdown.csv")

    if not trans_csv.exists() or not copy_csv.exists():
        print("Breakdown files not found.")
        return

    df_tr = pd.read_csv(trans_csv)
    df_cp = pd.read_csv(copy_csv)
    merged = pd.merge(df_tr, df_cp, on="perturbation", suffixes=("_translator", "_naive_copy"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel a: Pearson Correlation
    ax = axes[0]
    add_panel_label(ax, "a")
    ax.scatter(
        merged["pearson_top20_de_naive_copy"],
        merged["pearson_top20_de_translator"],
        alpha=0.65,
        color="#2B5C8F",
        edgecolors="white",
        linewidths=0.5,
        s=48,
    )
    ax.plot([0.3, 1.0], [0.3, 1.0], "r--", linewidth=1.5, label="Identity line ($x = y$)")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Naive K562 shift copy (Top-20 DE $\\rho$)")
    ax.set_ylabel("BioTransferTranslator (Top-20 DE $\\rho$)")
    ax.set_title("Per-target Pearson correlation (unseen genes, N=308)", pad=12, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="lower right")

    # Panel b: MSE Reduction
    ax = axes[1]
    add_panel_label(ax, "b")
    ax.scatter(
        merged["mse_top20_de_naive_copy"],
        merged["mse_top20_de_translator"],
        alpha=0.65,
        color="#D95F02",
        edgecolors="white",
        linewidths=0.5,
        s=48,
    )
    max_val = max(merged["mse_top20_de_naive_copy"].max(), merged["mse_top20_de_translator"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Identity line ($x = y$)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Naive K562 shift copy (Top-20 DE MSE)")
    ax.set_ylabel("BioTransferTranslator (Top-20 DE MSE)")
    ax.set_title("Per-target mean squared error (unseen genes, N=308)", pad=12, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    out_path = OUT_DIR / "fig2_dual_zero_shot_scatter.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig3_pathway_conservation():
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

    def assign_category(gene):
        g = gene.upper()
        if g.startswith(("RPL", "RPS", "MRPL", "MRPS")):
            return "Ribosome / Translation"
        elif g.startswith(("PSMA", "PSMB", "PSMC", "PSMD")):
            return "Proteasome Core"
        elif g.startswith(("POLR", "TBP", "GTF")):
            return "RNA Pol / Transcription"
        elif g.startswith(("ACT", "TUB", "MYO", "KIF", "ARPC")):
            return "Cytoskeleton & Spindle"
        elif g in {"TP53", "MDM2", "CDKN1A", "CHEK1", "CHEK2", "ATM", "ATR", "RAD51"}:
            return "DNA Damage / p53 Axis"
        elif g.startswith(("NDUF", "COX", "UQCR", "ATP5", "POLRMT", "LRPPRC")):
            return "Mitochondrial Respiration"
        else:
            return "Other Essential"

    df["Pathway"] = df["perturbation"].apply(assign_category)
    df_active = df[df["k562_mag"] > 1.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Panel a: Boxplot by Pathway
    ax = axes[0]
    add_panel_label(ax, "a")
    order = df_active.groupby("Pathway")["cross_cell_rho"].median().sort_values(ascending=False).index
    palette = sns.color_palette("muted", len(order))

    sns.boxplot(
        data=df_active,
        x="Pathway",
        y="cross_cell_rho",
        order=order,
        hue="Pathway",
        palette=palette,
        legend=False,
        ax=ax,
        fliersize=2,
    )
    ax.set_title("Cross-cell lineage transferability by pathway", pad=12, fontweight="medium")
    ax.set_ylabel("K562-RPE1 response correlation ($\\rho$)")
    ax.set_xlabel("")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=22, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel b: Scatter Magnitude
    ax = axes[1]
    add_panel_label(ax, "b")
    sc = ax.scatter(
        df["k562_mag"],
        df["rpe1_mag"],
        alpha=0.45,
        c=df["cross_cell_rho"],
        cmap="coolwarm",
        s=32,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Lineage correlation ($\\rho$)")
    ax.plot([0, 25], [0, 25], "k--", alpha=0.5, label="Equal magnitude ($x = y$)")
    ax.set_xlabel("K562 differential shift norm ($||\\Delta_{\\mathrm{K562}}||_2$)")
    ax.set_ylabel("RPE1 differential shift norm ($||\\Delta_{\\mathrm{RPE1}}||_2$)")
    ax.set_title("Perturbation response magnitude comparison", pad=12, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    out_path = OUT_DIR / "fig3_pathway_conservation_landscape.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig4_multi_dataset_benchmark():
    """Figure 4: Generalization Across Dixit, Adamson, Norman, K562, and RPE1."""
    ind_csv = Path("results/all_individual_datasets_summary.csv")
    cross_csv = Path("results/cross_cell/cross_cell_benchmark_summary.csv")

    if not ind_csv.exists() or not cross_csv.exists():
        print("Data files for Figure 4 not found.")
        return

    df_ind = pd.read_csv(ind_csv)
    df_cross = pd.read_csv(cross_csv)

    # Filter K562 OOD and RPE1 Dual Zero Shot from cross-cell
    k562_ood = df_cross[df_cross["Scenario"] == "Task_1_K562_OOD_Test"].copy()
    k562_ood["Dataset"] = "k562_essential"
    k562_ood["Model"] = k562_ood["Model"].replace({"BioTransferNet (De Novo)": "BioTransferNet (Ours)"})

    # Harmonize columns
    records = []
    for _, r in df_ind.iterrows():
        records.append({
            "Dataset": r["Dataset"],
            "Model": r["Model"],
            "pearson_top20_de": float(r["Pearson ρ (Top-20 DE)"]),
            "mse_top20_de": float(r["MSE (Top-20 DE)"]),
        })

    for _, r in k562_ood.iterrows():
        if r["Model"] in ["Control Mean", "Mean Shift", "BioTransferNet (Ours)"]:
            records.append({
                "Dataset": "k562_essential",
                "Model": r["Model"],
                "pearson_top20_de": float(r["pearson_top20_de"]),
                "mse_top20_de": float(r["mse_top20_de"]),
            })

    df_all = pd.DataFrame(records)

    dataset_order = ["dixit", "adamson", "norman", "k562_essential"]
    dataset_labels = ["Dixit (TF)\nN=4", "Adamson (UPR)\nN=11", "Norman (Dual)\nN=25", "Replogle K562\nN=308"]
    models = ["Control Mean", "Mean Shift", "BioTransferNet (Ours)"]

    palette = {
        "Control Mean": "#8E9AA8",
        "Mean Shift": "#5B84B1",
        "BioTransferNet (Ours)": "#009E73",
    }

    bar_width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))

    metrics = [
        ("pearson_top20_de", "Hold-out unseen gene prediction correlation across datasets", "Top-20 DE Pearson correlation ($\\rho$)", (0.78, 1.01)),
        ("mse_top20_de", "Hold-out unseen gene prediction MSE across datasets", "Top-20 DE mean squared error (MSE)", (0.0, 0.52)),
    ]

    for ax_idx, (col, title, ylabel, ylim) in enumerate(metrics):
        ax = axes[ax_idx]
        add_panel_label(ax, "a" if ax_idx == 0 else "b", x=-0.08, y=1.06)

        offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * (bar_width + 0.02)

        for d_idx, ds in enumerate(dataset_order):
            sub_df = df_all[df_all["Dataset"] == ds].set_index("Model")
            for m_i, m_name in enumerate(models):
                if m_name in sub_df.index:
                    val = float(sub_df.loc[m_name, col])
                    ax.bar(
                        d_idx + offsets[m_i],
                        val,
                        width=bar_width,
                        color=palette[m_name],
                        edgecolor="black",
                        linewidth=0.7,
                    )

        ax.set_xticks(range(len(dataset_order)))
        ax.set_xticklabels(dataset_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12, fontweight="medium")
        ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[m], edgecolor="black", linewidth=0.7, label=m)
        for m in models
    ]

    fig.legend(
        handles=legend_elements,
        labels=models,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#C0C0C0",
        framealpha=1.0,
        fontsize=11.5,
        handlelength=1.5,
        handleheight=0.9,
        columnspacing=2.5,
        borderpad=0.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = OUT_DIR / "fig4_multi_dataset_benchmark.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating all publication-ready figures with strict Nature styling ('a', 'b', 'c')...")
    plot_fig1_benchmark_performance()
    plot_fig2_dual_zero_shot_scatter()
    plot_fig3_pathway_conservation()
    plot_fig4_multi_dataset_benchmark()
    print("All figures successfully created in results/paper_figures/!")


if __name__ == "__main__":
    main()
