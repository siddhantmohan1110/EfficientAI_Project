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

# ------------- CONFIGURATION ------------- #

MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 8
NUM_CLASSES = 10  # Assume CIFAR-10 for simplicity
LR = 2e-5
NUM_EPOCHS = 1  # Keep it low for demonstration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "gradient_histograms"
LOG_SAVE_DIR = "gradient_histograms_log"
COMBINED_SAVE_DIR = "gradient_histograms_combined"

# Create directories to save plots
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_SAVE_DIR, exist_ok=True)
os.makedirs(COMBINED_SAVE_DIR, exist_ok=True)

# ------------- DATA PREPARATION ------------- #

# Transformation: Resize CIFAR-10 (32x32) images to 224x224 for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------- MODEL SETUP ------------- #

# Load Vision Transformer model with classification head
model = ViTForImageClassification.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)

# Replace classifier for CIFAR-10
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ------------- GRADIENT COLLECTION SETUP ------------- #

# Dictionary to store gradients per layer
# Structure: {layer_name: [list of gradients per step]}
gradient_storage = defaultdict(list)

# Hook function to capture gradients
def save_gradient(name):
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradient_storage[name].append(grad_output[0].detach().cpu().flatten())
    return hook

# Register hooks on each layer
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

        # For demonstration, only process a few batches
        if batch_idx == 10:
            break

# ------------- GRADIENT HISTOGRAM PLOTTING ------------- #

# Function to plot histogram for each layer
def plot_gradients(gradient_storage, bins=50):
    for layer_name, grads_list in gradient_storage.items():
        # Concatenate all gradients collected across steps
        all_grads = torch.cat(grads_list)

        # Take absolute value
        all_grads_abs = all_grads.abs()

        # Compute statistics for centering and scaling
        mean_abs = all_grads_abs.mean().item()
        std_abs = all_grads_abs.std().item()

        # Plot normal gradient histogram
        frequencies, bin_edges = np.histogram(all_grads_abs.numpy(), bins=bins, range=(mean_abs - 3*std_abs, mean_abs + 3*std_abs))

        plt.figure(figsize=(8, 5))
        plt.bar(bin_edges[:-1], frequencies, width=np.diff(bin_edges), edgecolor="black", align="edge")
        plt.xlabel("|Gradient Value|")
        plt.ylabel("Frequency")
        plt.title(f"Gradient Histogram for Layer: {layer_name}")
        plt.xlim(mean_abs - 3*std_abs, mean_abs + 3*std_abs)
        plt.grid(True)
        plt.tight_layout()

        safe_layer_name = layer_name.replace("/", "_").replace(".", "_")
        plt.savefig(os.path.join(SAVE_DIR, f"{safe_layer_name}.png"))
        plt.close()

        # Plot log of gradient histogram
        all_grads_log = torch.log1p(all_grads_abs)  # log(1 + x) to avoid log(0)
        mean_log = all_grads_log.mean().item()
        std_log = all_grads_log.std().item()
        frequencies_log, bin_edges_log = np.histogram(all_grads_log.numpy(), bins=bins, range=(mean_log - 3*std_log, mean_log + 3*std_log))

        plt.figure(figsize=(8, 5))
        plt.bar(bin_edges_log[:-1], frequencies_log, width=np.diff(bin_edges_log), edgecolor="black", align="edge")
        plt.xlabel("log(1 + |Gradient Value|)")
        plt.ylabel("Frequency")
        plt.title(f"Log Gradient Histogram for Layer: {layer_name}")
        plt.xlim(mean_log - 3*std_log, mean_log + 3*std_log)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(LOG_SAVE_DIR, f"{safe_layer_name}_log.png"))
        plt.close()

        # Combined plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].bar(bin_edges[:-1], frequencies, width=np.diff(bin_edges), edgecolor="black", align="edge")
        axes[0].set_xlabel("|Gradient Value|")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Normal Gradient")
        axes[0].set_xlim(mean_abs - 3*std_abs, mean_abs + 3*std_abs)
        axes[0].grid(True)

        axes[1].bar(bin_edges_log[:-1], frequencies_log, width=np.diff(bin_edges_log), edgecolor="black", align="edge")
        axes[1].set_xlabel("log(1 + |Gradient Value|)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Log Gradient")
        axes[1].set_xlim(mean_log - 3*std_log, mean_log + 3*std_log)
        axes[1].grid(True)

        fig.suptitle(f"Gradient Histograms for Layer: {layer_name}")
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        plt.savefig(os.path.join(COMBINED_SAVE_DIR, f"{safe_layer_name}_combined.png"))
        plt.close()

# Call plotting function
plot_gradients(gradient_storage)