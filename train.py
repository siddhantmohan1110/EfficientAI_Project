from data import get_data_loaders
from model import setup_model
from utils import prune_gradients_three_way
from config import get_config
from utils import log_training, plot_gradients, track_laplacian_params

import os
import time
import torch
from torch import amp
from torch.amp import GradScaler

from contextlib import contextmanager

@contextmanager
def identity_context():
    yield

def collect_gradients(model):
    storage = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in storage:
                storage[name] = []
            storage[name].append(param.grad.detach().clone().flatten().cpu())
    return storage

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct = total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total

def train(cfg):
    device = cfg["DEVICE"]
    base_dir = cfg["BASE_DIR"]
    logs_dir = os.path.join(base_dir, "logs")

    

    train_loader, test_loader = get_data_loaders(cfg)
    model, optimizer, loss_fn = setup_model(cfg)

    steps, losses, train_acc, test_acc = [], [], [], []
    step = 0

    is_mps = device.type == "mps"
    scaler = GradScaler() if not is_mps else None

    start = time.time()
    for _ in range(cfg["EPOCHS"]):
        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device)
            optimizer.zero_grad()
            with amp.autocast(device_type=device.type) if not is_mps else identity_context():
                logits = model(x).logits
                loss = loss_fn(logits, y)
            if is_mps:
                loss.backward()
                if cfg["PRUNE"]:
                    eps = torch.rand(1).item()
                    prune_gradients_three_way(model, cfg["SPARSITY"], eps)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                if cfg["PRUNE"]:
                    eps = torch.rand(1).item()
                    prune_gradients_three_way(model, cfg["SPARSITY"], eps)
                scaler.step(optimizer)
                scaler.update()
            step += 1
            if step % 500 == 0:
                acc = logits.argmax(1).eq(y).sum().item() / y.size(0)
                test = evaluate(model, test_loader, device)
                steps.append(step)
                losses.append(loss.item())
                train_acc.append(acc)
                test_acc.append(test)
                if cfg["VERBOSE"]:
                    print(f"Step {step}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}, Test Acc: {test:.4f}")
            

    if cfg["VERBOSE"]:
        print("Time taken:", time.time() - start)

    prefix = "prune" if cfg["PRUNE"] else "baseline"
    log_training(
    steps, losses, train_acc, test_acc,
    prefix="prune" if cfg["PRUNE"] else "baseline",
    save_dir=os.path.join(cfg["BASE_DIR"], "logs")
)

if __name__ == '__main__':
    cfg = get_config()
    train(cfg)