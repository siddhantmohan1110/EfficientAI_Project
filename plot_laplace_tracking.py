import os
import numpy as np
import matplotlib.pyplot as plt

def plot_laplace_logs(log_dir):
    for fname in os.listdir(log_dir):
        if fname.endswith("_laplace.npy"):
            data = np.load(os.path.join(log_dir, fname), allow_pickle=True)
            if not data:
                continue
            steps, locs, scales = zip(*data)

            fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            axs[0].plot(steps, locs, marker='o', label="Loc (μ)")
            axs[0].set_ylabel("Loc (μ)")
            axs[0].grid(True)
            axs[0].legend()

            axs[1].plot(steps, scales, marker='o', color='orange', label="Scale (b)")
            axs[1].set_xlabel("Training Step")
            axs[1].set_ylabel("Scale (b)")
            axs[1].grid(True)
            axs[1].legend()

            fig.suptitle(fname.replace("_laplace.npy", "").replace("_", "."))
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)

            plot_path = os.path.join(log_dir, fname.replace(".npy", "_params.png"))
            plt.savefig(plot_path)
            plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Path to logs/ folder inside BASE_DIR")
    args = parser.parse_args()
    plot_laplace_logs(args.log_dir)
