# config.py

# Dataset and Emotion Mapping
NUM_CLASSES = 6  # HAP, SAD, ANG, FEA, DIS, NEU
EMOTION_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 5}
DATA_PATH = "../dataset/"
CSV_PATH = "../dataset/cremad/cremad_paths.csv"

# Image settings
IMAGE_SIZE = 224
SEQ_LEN = 3  # Number of frames per sequence

# Audio settings
SAMPLE_RATE = 16000
N_FFT = 1024
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
WAVE_SECONDS = 4.0
WAVE_TARGET_LEN = int(SAMPLE_RATE * WAVE_SECONDS)

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LR_HEAD = 1e-3
LR_BACKBONE = 3e-4
WEIGHT_DECAY_HEAD = 1e-4
WEIGHT_DECAY_BACKBONE = 1e-5
SEED = 42

# Model
IMAGE_BACKBONE = "resnet18"
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
FUSION_TYPE = "crossattn"
FREEZE_BACKBONES = True
IMG_UNFREEZE_LAST_BLOCKS = 1
AUDIO_UNFREEZE_LAST_BLOCKS = 1

# Paths
MODEL_SAVE_PATH = "./checkpoints/best_model.pth"
