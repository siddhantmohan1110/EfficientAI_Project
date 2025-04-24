import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTFeatureExtractor
from collections import defaultdict
import numpy as np
import os
import re

# ------------- CONFIGURATION ------------- #

MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 8
NUM_CLASSES = 10  # Assume CIFAR-10 for simplicity
LR = 2e-5
NUM_EPOCHS = 1  # Keep it low for demonstration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMBINED_SAVE_DIR = "gradient_histograms_combined_wabs"

# Create directory to save combined plots
os.makedirs(COMBINED_SAVE_DIR, exist_ok=True)

# ------------- DATA PREPARATION ------------- #

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------- MODEL SETUP ------------- #

model = ViTForImageClassification.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)

in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ------------- GRADIENT COLLECTION SETUP ------------- #

gradient_storage = defaultdict(list)

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradient_storage[name].append(grad_output[0].detach().cpu().flatten())
    return hook

for name, module in model.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.LayerNorm):
        module.register_full_backward_hook(save_gradient(name))

# ------------- TRAINING LOOP ------------- #

model.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        if batch_idx == 10:
            break

# ------------- GRADIENT HISTOGRAM PLOTTING ------------- #

def plot_gradients(gradient_storage, bins=50):
    grouped_layers = defaultdict(dict)

    for name, grads_list in gradient_storage.items():
        match = re.search(r'encoder\.layer\.(\d+)\.(.*?)\.(.*)', name)
        if match:
            layer_id, subblock, param = match.groups()
            key = f"layer_{layer_id}_{subblock}"
            grouped_layers[key][param] = name
        else:
            if name.startswith("classifier") or "pooler" in name:
                grouped_layers["classifier_head"][name] = name
            else:
                grouped_layers[name]["single"] = name

    for group_name, subparts in grouped_layers.items():
        fig, axes = plt.subplots(2, len(subparts), figsize=(6 * len(subparts), 10))
        if len(subparts) == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for col, (param, name) in enumerate(subparts.items()):
            #grads = torch.abs(torch.cat(gradient_storage[name]))

            grads = torch.cat(gradient_storage[name])

            nonzero_grads = grads[grads != 0]
            if nonzero_grads.numel() == 0:
                continue

            mean, std = grads.mean().item(), grads.std().item()
            log_grads = torch.log(nonzero_grads.abs())
            mean_log, std_log = log_grads.mean().item(), log_grads.std().item()

            freq, bins_ = np.histogram(grads.numpy(), bins=bins, range=(mean - 3 * std, mean + 3 * std), density=False)
            freq_log, bins_log = np.histogram(log_grads.numpy(), bins=bins, range=(mean_log - 3 * std_log, mean_log + 3 * std_log), density=False)

            ax1 = axes[0][col]
            ax1.bar(bins_[:-1], freq, width=np.diff(bins_), edgecolor="black", align="edge", alpha=0.7)
            ax1.set_title(f"{group_name}: {param}\nMean={mean:.2e}, Std={std:.2e}")
            ax1.set_xlabel("Grad Value")
            ax1.set_ylabel("Count")
            ax1.grid(True)

            ax2 = axes[1][col]
            ax2.bar(bins_log[:-1], freq_log, width=np.diff(bins_log), edgecolor="black", align="edge", alpha=0.7)
            ax2.set_title(f"log(|Grad|): Mean={mean_log:.2f}, Std={std_log:.2f}")
            ax2.set_xlabel("log(|Grad|)")
            ax2.set_ylabel("Count")
            ax2.grid(True)

        fig.tight_layout()
        plt.savefig(os.path.join(COMBINED_SAVE_DIR, f"{group_name.replace('.', '_')}_grouped.png"))
        plt.close()

plot_gradients(gradient_storage)
