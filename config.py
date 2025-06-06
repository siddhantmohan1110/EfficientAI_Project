import argparse
import torch
import random
import numpy as np
import os
from datetime import datetime

def get_config():
    parser = argparse.ArgumentParser(description="ViT Gradient Analysis Configuration")

    # Training and dataset
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="CIFAR10")

    # System and logging
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--save_root", type=str, default="plots", help="Root directory for saving experiment runs")
    parser.add_argument("--tag", type=str, default=None, help="Optional run name prefix")
    parser.add_argument("--laplace_param_tracking", action="store_true", help="Track Laplace params")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args, _ = parser.parse_known_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Run name and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip().replace(" ", "_") if args.tag else None
    run_name = f"{tag}_{timestamp}" if tag else f"{'prune' if args.prune else 'baseline'}_{timestamp}"
    base_dir = os.path.join(args.save_root, run_name)

    os.makedirs(os.path.join(base_dir, "histograms"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = {
        "RUN_NAME": run_name,
        "BASE_DIR": base_dir,
        "DEVICE": device,
        "PIN_MEMORY": device.type != "cpu",
        "MODEL_NAME": args.model_name,
        "BATCH_SIZE": args.batch_size,
        "NUM_CLASSES": args.num_classes,
        "LEARNING_RATE": args.learning_rate,
        "EPOCHS": args.epochs,
        "NUM_WORKERS": args.num_workers,
        "SPARSITY": args.sparsity,
        "USE_PRETRAINED": args.use_pretrained,
        "VERBOSE": args.verbose,
        "PRUNE": args.prune,
        "DATASET": args.dataset,
        "SEED": args.seed,
        "LAPLACE_PARAM_TRACKING": args.laplace_param_tracking,
    }

    return config
