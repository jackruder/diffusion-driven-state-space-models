import os
import csv
import math
import argparse

import matplotlib.pyplot as plt


def plot_metrics(csv_path, save_path, keys=None):
    """Plots specific metrics from CSV log on separate subplots."""
    if keys is None:
        keys = ["loss/total"]

    if not os.path.exists(csv_path):
        print(f"[Warning] No CSV log found at {csv_path}, skipping metric plot.")
        return

    data = {k: [] for k in keys}
    steps = []

    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                valid_keys = [k for k in keys if k in reader.fieldnames]
                if len(valid_keys) != len(keys):
                    print(f"[Warning] Some keys not found in CSV. Found: {valid_keys}")
                keys = valid_keys

            for row in reader:
                if "step" not in row:
                    continue
                try:
                    s = int(row["step"])
                    current_values = {}
                    ok = True
                    for k in keys:
                        try:
                            current_values[k] = float(row[k])
                        except ValueError:
                            print("Found error in table at step: ", s)
                            ok = False
                            continue

                    if ok:
                        steps.append(s)
                        for k in keys:
                            data[k].append(current_values[k])
                except ValueError:
                    print("Error in table")
                    continue

    except Exception as e:
        print(f"[Error] Failed to read metrics: {e}")
        return

    if not steps:
        print("[Warning] No valid data found in CSV log.")
        return

    # Determine grid size
    n = len(keys)
    cols = 1 if n == 1 else 2
    rows = math.ceil(n / cols)

    plt.figure(figsize=(6 * cols, 4 * rows))

    for i, k in enumerate(keys):
        plt.subplot(rows, cols, i + 1)
        plt.plot(steps, data[k], label=k)
        plt.xlabel("Steps")
        plt.ylabel(k)
        plt.title(k)
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved metric plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from a CSV log file.")

    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Base directory for logs (e.g., runs/unit_recon_iid).",
    )
    parser.add_argument(
        "--mode", type=str, required=True, help="Synthetic data mode (e.g., iid)."
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        default=["loss/total"],
        help="List of metrics column names to plot.",
    )

    args = parser.parse_args()

    # Automatically resolve the 'latest' symlink path
    target_dir = os.path.join(args.work_dir, args.mode, "latest")
    csv_file = os.path.join(target_dir, "metrics.csv")
    out_file = os.path.join(target_dir, "metrics_plot.png")

    plot_metrics(csv_file, out_file, keys=args.keys)
