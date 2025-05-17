import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTConfig
from collections import defaultdict
import numpy as np
import os
import re
import time
from scipy.stats import norm, laplace, t, lognorm, cauchy, pareto, logistic, invgamma

# --- Configuration --- #
MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 8
NUM_CLASSES = 10
LR = 2e-5
NUM_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "gradient_histograms_combined_raw_init"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Data Loader --- #
def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model and Optimizer --- #
def setup_model():
    config = ViTConfig.from_pretrained(MODEL_NAME)
    config.num_labels = NUM_CLASSES
    model = ViTForImageClassification(config)
    #model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    #model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    return model.to(DEVICE), optim.AdamW(model.parameters(), lr=LR), nn.CrossEntropyLoss()

# --- Gradient Hooking --- #
def register_hooks(model, storage):
    def save_grad(name):
        def hook(_, __, grad_output):
            if grad_output[0] is not None:
                storage[name].append(grad_output[0].detach().cpu().flatten())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            module.register_full_backward_hook(save_grad(name))

# --- Training --- #
def train(model, loader, optimizer, criterion, gradient_storage=None):
    model.train()
    for epoch in range(NUM_EPOCHS):
        for idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Batch {idx+1}, Loss: {loss.item():.4f}")
            if idx == 10:
                return

    
    #return

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

# --- Plotting --- #
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

# --- Execution --- #
if __name__=='__main__':

    data_loader = get_data_loader()
    model, optimizer, criterion = setup_model()
    grad_storage = defaultdict(list)
    register_hooks(model, grad_storage)
    train(model, data_loader, optimizer, criterion, grad_storage)

    # t1 = time.time()
    # train(model, data_loader, optimizer, criterion)
    # print('Time taken :', time.time() - t1)

    plot_gradients(grad_storage)