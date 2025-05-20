import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm, laplace, t, lognorm, cauchy, logistic, pareto, invgamma

from config import SAVE_DIR


# --- Q-Q Plot Utility --- #
def compute_qq(data, dist, absval=False):
    data = data.cpu().numpy()
    if absval:
        data = np.abs(data)
    data = data[np.isfinite(data) & (data != 0)]
    probs = np.linspace(0.01, 0.99, 100)
    empirical = np.quantile(data, probs)

    if dist == 'normal':
        loc, scale = norm.fit(data)
        theoretical = norm.ppf(probs, loc=loc, scale=scale)
    elif dist == 'laplace':
        loc, scale = laplace.fit(data)
        theoretical = laplace.ppf(probs, loc=loc, scale=scale)
    elif dist == 't':
        shape, loc, scale = t.fit(data)
        theoretical = t.ppf(probs, df=3, loc=loc, scale=scale)
    elif dist == 'lognorm':
        shape, loc, scale = lognorm.fit(data, floc=0)
        theoretical = lognorm.ppf(probs, shape, loc=loc, scale=scale)
    elif dist == 'cauchy':
        loc, scale = cauchy.fit(data)
        theoretical = cauchy.ppf(probs, loc=loc, scale=scale)
    elif dist == 'logistic':
        loc, scale = logistic.fit(data)
        theoretical = logistic.ppf(probs, loc=loc, scale=scale)
    elif dist == 'pareto':
        b, loc, scale = pareto.fit(data)
        theoretical = pareto.ppf(probs, b, loc=loc, scale=scale)
    elif dist == 'invgamma':
        b, loc, scale = invgamma.fit(data)
        theoretical = invgamma.ppf(probs, b, loc=loc, scale=scale)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    return theoretical, empirical


def plot_gradients(storage, bins=50):
    grouped = defaultdict(dict)
    for name in storage:
        match = re.search(r'encoder\.layer\.(\d+)\.(.*?)\.(.*)', name)
        if match:
            lid, sub, param = match.groups()
            grouped[f"layer_{lid}_{sub}"][param] = name
        else:
            grouped["classifier_head"][name] = name

    ch=0
    for gname, parts in grouped.items():
        ch+=1
        fig, axs = plt.subplots(4, len(parts), figsize=(6 * len(parts), 15))
        if len(parts) == 1:
            axs = np.array([[axs[0]], [axs[1]], [axs[2]], [axs[3]]])

        for col, (param, name) in enumerate(parts.items()):
            grads = torch.cat(storage[name])
            nonzero = grads[grads != 0]
            if nonzero.numel() == 0:
                continue

            abs_grads = nonzero.abs()
            log_grads = torch.log(abs_grads)
            m, s = grads.mean().item(), grads.std().item()
            mlog, slog = log_grads.mean().item(), log_grads.std().item()

            f1, b1 = np.histogram(grads.numpy(), bins=bins, range=(m - 3*s, m + 3*s))
            f2, b2 = np.histogram(log_grads.numpy(), bins=bins, range=(mlog - 3*slog, mlog + 3*slog))

            axs[0][col].bar(b1[:-1], f1, width=np.diff(b1), edgecolor="black")
            axs[0][col].set_title(f"{gname}: {param}\nMean={m:.2e}, Std={s:.2e}")
            axs[0][col].set_xlabel("Grad")
            axs[0][col].set_ylabel("Count")

            axs[1][col].bar(b2[:-1], f2, width=np.diff(b2), edgecolor="black")
            axs[1][col].set_title(f"log(|Grad|): Mean={mlog:.2f}, Std={slog:.2f}")
            axs[1][col].set_xlabel("log(|Grad|)")
            axs[1][col].set_ylabel("Count")

            centred = ['normal', 'laplace', 'logistic']
            heavy_tailed = ['lognorm', 'pareto', 'invgamma']

            for d in centred:
                tq, eq = compute_qq(grads, d)
                axs[2][col].plot(tq, eq, label=d)
            axs[2][col].plot(tq, tq, 'k--')
            axs[2][col].set_title("Q-Q Plot (Centered)")
            axs[2][col].legend()
            axs[2][col].set_xlabel("Theoretical")
            axs[2][col].set_ylabel("Actual")

            for d in heavy_tailed:
                tq, eq = compute_qq(abs_grads, d)
                axs[3][col].plot(tq, eq, label=d)
            axs[3][col].plot(tq, tq, 'k--')
            axs[3][col].set_title("Q-Q Plot (Heavy-Tailed)")
            axs[3][col].legend()
            axs[3][col].set_xlabel("Theoretical")
            axs[3][col].set_ylabel("Actual")

        fig.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{gname.replace('.', '_')}_grouped.png"))
        plt.close(fig)
        # if ch==4:
        #     break

def plot_and_save_quantization_stats(
    grad_storage,
    quantized_grads,
    save_dir="quant_plots",
    num_bins=50
):
    """
    For each module in grad_storage, compute and save:
      • A histogram of quantization levels (no zeros, no ±inf/NaN)
      • A histogram of quantization error (no zeros, no ±inf/NaN)
      • A bar‐chart of the quantization levels themselves
    into PNG files under `save_dir`.
    """
    os.makedirs(save_dir, exist_ok=True)

    for name in grad_storage:
        # --- stack & mask as before --- #
        g   = np.concatenate([g.cpu().numpy() for g in grad_storage[name]])
        g_q = quantized_grads[name].cpu().numpy()

        mask = (g != 0)
        g, g_q = g[mask], g_q[mask]
        finite_mask = np.isfinite(g) & np.isfinite(g_q)
        g, g_q = g[finite_mask], g_q[finite_mask]

        if g.size == 0:
            print(f"Skipping '{name}' (all grads were zero/non‐finite)")
            continue

        levels = np.unique(g_q)
        error  = g - g_q
        mse    = np.mean(error**2)

        # --- expand to 4 subplots: dist(levels), error, linear bars, log‐scale bars --- #
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20, 4))

        # 1) distribution of quantization levels
        ax0.hist(levels, bins=min(len(levels), num_bins), edgecolor='white')
        ax0.set_title(f"{name}\n# levels = {len(levels)}")
        ax0.set_xlabel("Quantized value")
        ax0.set_ylabel("Count")

        # 2) distribution of quantization error
        ax1.hist(error, bins=num_bins, edgecolor='white')
        ax1.set_title(f"{name}\nMSE = {mse:.2e}")
        ax1.set_xlabel("Quantization error")
        ax1.set_ylabel("Count")

        # 3) bar‐chart of levels (linear scale)
        if len(levels) > 1:
            width = np.min(np.diff(levels))
        else:
            width = 1.0
        ax2.bar(levels,
                np.ones_like(levels),
                width=width,
                align='center',
                edgecolor='red')
        ax2.set_title(f"{name}\nLinear Quantization Levels")   # New: title change
        ax2.set_xlabel("Value")
        ax2.set_yticks([])
        ax2.set_ylim(0, 1.1)

        # --- Newly added snippet: bar‐chart on log(abs(x)+1) scale --- #
        # transform levels by log(abs(x)+1)
        levels_trans = np.log(np.abs(levels) + 1)                         # New
        # compute bar width in transformed space
        if len(levels_trans) > 1:
            width_t = np.min(np.diff(levels_trans))  * 0.1                    # New
        else:
            width_t = 1.0                                                # New
        ax3.bar(levels_trans,
                np.ones_like(levels_trans),
                width=width_t,
                align='center',
                edgecolor='white')                                      # New
        ax3.set_title(f"{name}\nlog(abs(x)+1) Quant Levels")             # New
        ax3.set_xlabel("log(|value| + 1)")                               # New
        ax3.set_yticks([])                                               # New
        ax3.set_ylim(0, 1.1)                                             # New
                                 # New: set y‐axis limits

        plt.tight_layout()

        safe_name = name.replace("/", "_").replace(".", "_")
        out_path   = os.path.join(save_dir, f"{safe_name}_quant.png")
        plt.savefig(out_path, dpi=150)
        plt.show()
        plt.close(fig)
        # break
        print(f"Saved quant stats for '{name}' → {out_path}")