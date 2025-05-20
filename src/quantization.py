import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import laplace


# --- 1) Fit Laplace params to all stored grads --- #
def laplacian_fit_params(grad_storage):
    """
    Flatten all gradients in grad_storage and fit a Laplace(loc, scale).
    grad_storage: dict[name] -> list of 1D torch tensors
    Returns: loc (float), scale (float)
    """
    # concatenate everything into one big 1D tensor
    all_grads = torch.cat([g for grads in grad_storage.values() for g in grads])
    all_np = all_grads.numpy()
    loc, scale = laplace.fit(all_np)  # SciPy’s MLE fit
    return loc, scale


# --- 2) Laplace CDF and PPF (inverse CDF) in torch --- #
def laplace_cdf(x, loc, scale):
    """
    Torch implementation of the Laplace CDF:
      F(x) = 0.5 exp((x-mu)/b)           if x < mu
           = 1 - 0.5 exp(-(x-mu)/b)      if x >= mu
    """
    return torch.where(
        x < loc,
        0.5 * torch.exp((x - loc) / scale),
        1 - 0.5 * torch.exp(-(x - loc) / scale)
    )


def laplace_ppf(u, loc, scale):
    """
    Torch implementation of the Laplace PPF (inverse CDF):
      F⁻¹(u) = mu + b · ln(2u)           if u < 0.5
             = mu - b · ln(2(1-u))       if u >= 0.5
    """
    return torch.where(
        u < 0.5,
        loc + scale * torch.log(2 * u),
        loc - scale * torch.log(2 * (1 - u))
    )


# --- 3) Quantization routine --- #
def quantize_gradients(grad_storage, bits=7):
    """
    For each named-gradient in grad_storage, compand→quantize→expand.
    Returns a dict[name] -> tensor of quantized values.
    """
    num_levels = 2 ** bits
    loc, scale = laplacian_fit_params(grad_storage)

    quantized = {}
    for name, grads_list in grad_storage.items():
        # stack all the saved grads for this module
        g = torch.cat(grads_list)  # 1D tensor of floats

        # after computing the raw CDF
        eps = 1e-6  # or 1/(num_levels*10), something << 1/L
        u = laplace_cdf(g, loc, scale)
        u = u.clamp(min=eps, max=1 - eps)  # ← NEW: avoid exact 0/1

        # quantize
        u_q = torch.round(u * (num_levels - 1)) / (num_levels - 1)
        u_q = u_q.clamp(min=eps, max=1 - eps)  # ← NEW: also clamp post‐quant

        # then invert
        g_q = laplace_ppf(u_q, loc, scale)

        quantized[name] = g_q

    return quantized


def compute_laplace_mse(grad_storage, quantized_grads):
    laplace_mse = {}
    for name in grad_storage:
        # original grads
        g = np.concatenate([g.cpu().numpy() for g in grad_storage[name]])
        # filter zeros & non‐finite
        mask = (g != 0) & np.isfinite(g)
        g = g[mask]
        g_q = quantized_grads[name].cpu().numpy()[mask]
        laplace_mse[name] = np.mean((g - g_q)**2)
    return laplace_mse

def uniform_quantize(g, num_levels):
    g_min, g_max = g.min(), g.max()
    if g_max == g_min:
        return np.full_like(g, g_min)
    # map to [0,1]
    u   = (g - g_min) / (g_max - g_min)
    u_q = np.round(u * (num_levels - 1)) / (num_levels - 1)
    # back to original range
    return g_min + u_q * (g_max - g_min)

def compute_uniform_mse(grad_storage, num_levels=256):
    uniform_mse = {}
    for name in grad_storage:
        g = np.concatenate([g.cpu().numpy() for g in grad_storage[name]])
        mask = (g != 0) & np.isfinite(g)
        g = g[mask]
        g_q = uniform_quantize(g, num_levels)
        uniform_mse[name] = np.mean((g - g_q)**2)
    return uniform_mse

# --- assume these exist in your notebook: grad_storage, quantized_grads ---

def plot_mse_difference(laplace_mse, uniform_mse, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # --- assume you already have laplace_mse and uniform_mse dicts --- #
    layers = [idx for idx, x in enumerate(laplace_mse.keys())]
    diff_mse = np.array([uniform_mse[l] - laplace_mse[l] for l in laplace_mse.keys()])
    total = diff_mse.sum()

    plt.figure(figsize=(12, 5))
    plt.plot(layers, diff_mse, marker='x', linestyle='-')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    # --- NEW: use symmetric log scale on y to zoom in around zero --- #
    linthresh = diff_mse.max() * 0.1 if diff_mse.max() > 0 else 1e-8  # New: threshold for linear region
    plt.yscale('symlog', linthresh=linthresh)  # New: apply symlog

    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Transformer Layer")
    plt.ylabel("Δ MSE (uniform − laplace)")
    plt.title(f"MSE Difference per Layer (Total Δ = {total:.2e})\n(symlog y‐axis, linthresh={linthresh:.1e})")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_dir + '/mse_error.png', dpi=250)
    plt.show()



# --- 4) Usage example, after train() has filled grad_storage: --- #
# num_bits = 8 → 2⁸ = 256 levels


# Now quantized_grads[name] is a 1D tensor of the same length as
# the flattened grads for each module.  You can reshape or re-assign
# them back into your model as needed.
