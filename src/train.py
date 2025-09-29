# train.py

import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger

from src import config
from src.dataset import EmotionDataset
from src.model import MultimodalEmotionRecognizer
from src.utils import setup_finetune, build_optimizer
from src.engine import train


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and split data
    df = pd.read_csv(config.CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=config.SEED, stratify=df["emotion"]
    )

    # Create datasets and dataloaders
    train_dataset = EmotionDataset(config.DATA_PATH, train_df, is_train=True)
    val_dataset = EmotionDataset(config.DATA_PATH, val_df, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    model = MultimodalEmotionRecognizer(
        num_classes=config.NUM_CLASSES,
        fusion=config.FUSION_TYPE,
        image_backbone=config.IMAGE_BACKBONE,
        ast_model_id=config.AST_MODEL_ID,
        T=config.SEQ_LEN,
    ).to(device)

    # Setup fine-tuning and optimizer
    if config.FREEZE_BACKBONES:
        setup_finetune(
            model,
            img_unfreeze_last_blocks=config.IMG_UNFREEZE_LAST_BLOCKS,
            audio_unfreeze_last_blocks=config.AUDIO_UNFREEZE_LAST_BLOCKS,
        )

    optimizer = build_optimizer(
        model,
        lr_head=config.LR_HEAD,
        lr_backbone=config.LR_BACKBONE,
        wd_head=config.WEIGHT_DECAY_HEAD,
        wd_backbone=config.WEIGHT_DECAY_BACKBONE,
    )

    # Handle multi-GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # Loss function and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    # Uncommend to turn on scheduler
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
    scheduler = None

    # Create directory for checkpoints
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    # Start training
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=config.EPOCHS,
        save_path=config.MODEL_SAVE_PATH,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
