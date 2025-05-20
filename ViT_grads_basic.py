import os
from collections import defaultdict

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig

from config import (
    MODEL_NAME, BATCH_SIZE, NUM_CLASSES, LR,
    NUM_EPOCHS, DEVICE, SAVE_DIR, QUANTIZATION_BITS
)
from src.utils import plot_gradients, plot_and_save_quantization_stats
from src.quantization import quantize_gradients, compute_laplace_mse, compute_uniform_mse, plot_mse_difference

# --- Configuration --- #
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
    # model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    # model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
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
    quantized_grads = quantize_gradients(grad_storage, bits=QUANTIZATION_BITS)
    plot_and_save_quantization_stats(grad_storage, quantized_grads)
    laplace_mse = compute_laplace_mse(grad_storage, quantized_grads)
    uniform_mse = compute_uniform_mse(grad_storage, num_levels=256)
    plot_mse_difference(laplace_mse, uniform_mse, SAVE_DIR)
    