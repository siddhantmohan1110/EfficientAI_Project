from collections import defaultdict
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, laplace, t, lognorm, cauchy, pareto, logistic, invgamma
from scipy.special import lambertw
import torch

def track_laplacian_params(step, storage, cfg):
    """
    Stores Laplace (loc, scale) params of attention layer gradients to .npy files.

    Args:
        step (int): Current training step
        storage (dict): {name: [tensor list]} of gradients
        cfg (dict): config with BASE_DIR
    """
    log_dir = os.path.join(cfg["BASE_DIR"], "logs")
    os.makedirs(log_dir, exist_ok=True)

    result = defaultdict(list)

    for name in storage:
        if "attention" not in name.lower():
            continue

        grads = torch.cat(storage[name])
        grads = grads[torch.isfinite(grads) & (grads != 0)]
        if grads.numel() == 0:
            continue

        data = grads.cpu().numpy()
        loc, scale = laplace.fit(data)
        result[name].append((step, loc, scale))

    for layer_name, records in result.items():
        sanitized_name = layer_name.replace(".", "_")
        path = os.path.join(log_dir, f"{sanitized_name}_laplace.npy")
        if os.path.exists(path):
            existing = np.load(path, allow_pickle=True).tolist()
        else:
            existing = []
        existing.extend(records)
        np.save(path, existing)


def log_training(steps, losses, train_acc, test_acc, prefix="baseline", save_dir="."):
    # Plot Loss
    plt.figure()
    plt.plot(steps, losses, marker='o', label='Loss')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title(f'{prefix.capitalize()}: Loss vs Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_{prefix}.png'))
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(steps, train_acc, marker='s', label='Train Acc')
    plt.plot(steps, test_acc, marker='^', label='Test Acc')
    plt.xlabel('Global Step')
    plt.ylabel('Accuracy')
    plt.title(f'{prefix.capitalize()}: Accuracy vs Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'acc_{prefix}.png'))
    plt.close()

    df = pd.DataFrame({
        "step": steps,
        "loss": losses,
        "train_acc": train_acc,
        "test_acc": test_acc
    })
    df.to_csv(os.path.join(save_dir, f"{prefix}_training_log.csv"), index=False)


def register_hooks(model, storage):
    def save_grad(name):
        def hook(_, __, grad_output):
            if grad_output[0] is not None:
                storage[name].append(grad_output[0].detach().cpu().flatten())
        return hook
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
            module.register_full_backward_hook(save_grad(name))


def plot_gradients(storage, cfg, bins=50):
    grouped = defaultdict(dict)
    for name in storage:
        match = re.search(r'encoder\.layer\.(\d+)\.(.*?)\.(.*)', name)
        if match:
            lid, sub, param = match.groups()
            grouped[f"layer_{lid}_{sub}"][param] = name
        else:
            grouped["classifier_head"][name] = name

    for gname, parts in grouped.items():
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

            # Centered distributions
            for d in ['normal', 'laplace', 'logistic']:
                tq, eq = compute_qq(grads, d)
                axs[2][col].plot(tq, eq, label=d)
            axs[2][col].plot(tq, tq, 'k--')
            axs[2][col].set_title("Q-Q Plot (Centered)")
            axs[2][col].legend()
            axs[2][col].set_xlabel("Theoretical")
            axs[2][col].set_ylabel("Actual")

            # Heavy-tailed distributions
            for d in ['lognorm', 'pareto', 'invgamma']:
                tq, eq = compute_qq(abs_grads, d)
                axs[3][col].plot(tq, eq, label=d)
            axs[3][col].plot(tq, tq, 'k--')
            axs[3][col].set_title("Q-Q Plot (Heavy-Tailed)")
            axs[3][col].legend()
            axs[3][col].set_xlabel("Theoretical")
            axs[3][col].set_ylabel("Actual")

        fig.tight_layout()
        save_path = os.path.join(cfg["BASE_DIR"], "histograms", f"{gname.replace('.', '_')}_grouped.png")
        plt.savefig(save_path)
        plt.close(fig)


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


def prune_gradients_three_way(model, sparsity_ratio, epsilon):
    grad_list = [param.grad.data.abs().flatten() for param in model.parameters() if param.grad is not None]
    all_gradients = torch.cat(grad_list).cpu().numpy()
    valid_gradients = all_gradients[np.isfinite(all_gradients)]
    if valid_gradients.size == 0:
        return 0.0, 0.0
    _, scale_b = laplace.fit(valid_gradients, floc=0)
    delta = 1.0 - sparsity_ratio
    argument = -1.0 / delta * np.exp(-1.0 / delta)
    lambert_term = lambertw(argument).real
    alpha = scale_b * ((1.0 / delta) + lambert_term)
    for param in model.parameters():
        if param.grad is None: continue
        grad_tensor = param.grad.data
        abs_grad = grad_tensor.abs()
        low_mask = abs_grad < alpha * epsilon
        mid_mask = (abs_grad >= alpha * epsilon) & (abs_grad <= alpha)
        grad_tensor[low_mask] = 0.0
        grad_tensor[mid_mask] = grad_tensor.sign()[mid_mask] * alpha
    return float(alpha), float(scale_b)