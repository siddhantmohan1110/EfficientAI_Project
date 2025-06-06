from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(cfg):
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    if cfg["DATASET"].lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root="data", train=True, transform=transform_pipeline, download=True)
        test_dataset = datasets.CIFAR10(root="data", train=False, transform=transform_pipeline, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {cfg['DATASET']}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=cfg["PIN_MEMORY"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=cfg["PIN_MEMORY"]
    )
    
    return train_loader, test_loader
