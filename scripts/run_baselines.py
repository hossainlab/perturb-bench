#!/usr/bin/env python
"""CLI tool to evaluate benchmark baselines across Perturb-seq datasets."""
import argparse
from pathlib import Path

import pandas as pd

from perturb_bench.data.loader import list_available_datasets, load_dataset
from perturb_bench.evaluation.benchmark import BenchmarkHarness
from perturb_bench.models.baselines import ControlMeanPredictor, MeanShiftPredictor
from perturb_bench.utils.seed import set_seed


def run_benchmark_on_dataset(dataset_name: str, out_dir: Path) -> pd.DataFrame:
    print("\n========================================================")
    print(f"Running Baselines on: {dataset_name}")
    print("========================================================")

    # 1. Load dataset with standardized audit fixes
    adata = load_dataset(dataset_name)
    print(f"Loaded {dataset_name}: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # 2. Fit baselines
    print("Fitting Control Mean baseline...")
    b_ctrl = ControlMeanPredictor().fit(adata)

    print("Fitting Mean Shift baseline...")
    b_shift = MeanShiftPredictor().fit(adata)

    # 3. Benchmark on hold-out test set
    harness = BenchmarkHarness(adata)
    print(f"Evaluating on {len(harness.test_perts)} hold-out test perturbations...")

    models = {
        "Control Mean": b_ctrl,
        "Mean Shift": b_shift,
    }

    results_df = harness.compare_models(models)
    results_df.insert(0, "Dataset", dataset_name)

    print("\nRESULTS:")
    print(results_df.to_string(index=False))

    # 4. Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"baselines_{dataset_name}.json"
    csv_path = out_dir / f"baselines_{dataset_name}.csv"

    results_df.to_json(json_path, orient="records", indent=2)
    results_df.to_csv(csv_path, index=False)
    print("\nSaved metrics to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate standard baselines on Perturb-seq benchmark datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dixit",
        help="Dataset name ('dixit', 'adamson', 'norman', 'replogle_k562_essential', or 'all').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory to save benchmark reports.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)

    if args.dataset.lower() == "all":
        datasets = list_available_datasets()
        all_results = []
        for d in datasets:
            try:
                res = run_benchmark_on_dataset(d, out_dir)
                all_results.append(res)
            except Exception as e:
                print(f"Error evaluating {d}: {e}")
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(out_dir / "baselines_all_summary.csv", index=False)
            print("\n================ ALL DATASETS SUMMARY ================")
            print(combined.to_string(index=False))
    else:
        run_benchmark_on_dataset(args.dataset, out_dir)


if __name__ == "__main__":
    main()
