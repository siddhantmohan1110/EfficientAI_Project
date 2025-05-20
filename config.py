MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 8
NUM_CLASSES = 10
LR = 2e-5
NUM_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "gradient_histograms_combined_raw_init"
QUANTIZATION_BITS=6
