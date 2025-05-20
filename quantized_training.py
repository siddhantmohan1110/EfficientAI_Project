from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTConfig, ViTForImageClassification
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.stats import laplace

from config import SAVE_DIR

# --- Hyperparameters --- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 1
BITS_LIST = [6, 4]  # run once for 6-bit, once for 4-bit
NUM_LEVELS = {b: 2 ** b for b in BITS_LIST}

global LOC_SCALE
LOC_SCALE = defaultdict(dict)

# --- Data Loader --- #
def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)




# --- Laplace companding functions --- #
def laplace_cdf_np(x, loc, scale):
    return np.where(
        x < loc,
        0.5 * np.exp((x - loc) / scale),
        1 - 0.5 * np.exp(-(x - loc) / scale)
    )


def laplace_ppf_np(u, loc, scale):
    return np.where(
        u < 0.5,
        loc + scale * np.log(2 * u),
        loc - scale * np.log(2 * (1 - u))
    )




def laplace_fit(w, layer):
    global LOC_SCALE
    if w in LOC_SCALE and layer in LOC_SCALE[w]:
        return LOC_SCALE[w][layer]

    LOC_SCALE[w][layer] = laplace.fit(w)
    return LOC_SCALE[w][layer]



def laplace_quantize_tensor(w, num_levels, layer, eps=1e-6, min_scale=1):
    # w = tensor.detach().cpu().numpy().ravel()

    # 2) Filter out any non‐finite values
    w_finite = w[np.isfinite(w)]
    if w_finite.size == 0:
        # nothing to fit on, leave unchanged
        return w

    loc, scale = laplace_fit(w, layer)
    scale = max(scale, min_scale)

    # 2) compand
    u = laplace_cdf_np(w, loc, scale)
    u = np.clip(u, eps, 1 - eps)

    # 3) uniform quant
    u_q = np.round(u * (num_levels - 1)) / (num_levels - 1)

    # 4) de‐compand
    w_q = laplace_ppf_np(u_q, loc, scale)

    # 5) clamp back to original support
    w_q = np.clip(w_q, w_finite.min(), w_finite.max())
    return w_q


def quantize_model_gradients(model, num_bits=6):
    num_levels = 2 ** num_bits
    for p in model.parameters():
        if p.grad is None:
            continue
        # pull out gradient, flatten to numpy
        g = p.grad.detach().cpu().numpy().ravel()
        # quantize
        g_q = laplace_quantize_tensor(g, num_levels, p.grad)
        # write back into p.grad
        p.grad.data.copy_(torch.from_numpy(g_q.reshape(p.grad.shape)).to(p.grad.device))


# --- Single‐run trainer with Laplace quantization --- #
def train_with_laplacian(bits):
    model = ViTForImageClassification(ViTConfig(num_labels=10))
    model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    losses = []
    loader = get_data_loader()
    for _ in tqdm(range(EPOCHS)):
        for idx, (x, y) in tqdm(enumerate(loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x).logits
            loss = crit(out, y)
            loss.backward()
            opt.step()
            # quantize weights in‐place
            with torch.no_grad():
                for p in model.parameters():
                    quantize_model_gradients(model, NUM_LEVELS[bits])
            losses.append(loss.item())

            if idx > 100:
                break
    return losses


# --- Plot both curves --- #
def plot_training(save_dir, loss_6_bits, loss_4_bits):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10,5))
    iters = list(range(1, len(loss_6_bits) + 1))
    plt.plot(iters, loss_6_bits, marker='o', label='6-bit Laplace')
    plt.plot(iters, loss_4_bits, marker='s', label='4-bit Laplace')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Loss Convergence: ViT trained with Laplace Quantized Weights")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + '/train_loss.png', dpi=250)
    plt.show()


def main():
    loss_6 = train_with_laplacian(6)
    loss_4 = train_with_laplacian(4)
    plot_training(SAVE_DIR, loss_6, loss_4)

if __name__ == "__main__":
    main()



