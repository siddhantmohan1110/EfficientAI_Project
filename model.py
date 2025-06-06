from transformers import ViTConfig, ViTForImageClassification
from torch import nn, optim

def setup_model(cfg):
    if cfg["USE_PRETRAINED"]:
        model = ViTForImageClassification.from_pretrained(cfg["MODEL_NAME"])
        model.classifier = nn.Linear(model.classifier.in_features, cfg["NUM_CLASSES"])
    else:
        config = ViTConfig.from_pretrained(cfg["MODEL_NAME"])
        config.num_labels = cfg["NUM_CLASSES"]
        model = ViTForImageClassification(config)
    
    model = model.to(cfg["DEVICE"]).float()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LEARNING_RATE"])
    loss_fn = nn.CrossEntropyLoss()

    return model, optimizer, loss_fn
