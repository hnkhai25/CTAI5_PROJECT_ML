# engine.py

import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, audio, labels in val_loader:
        images, audio, labels = images.to(device), audio.to(device), labels.to(device)

        outputs = model(images, audio)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    save_path,
    scheduler=None,
):
    """Main training loop."""
    best_val_acc = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, audio, labels in pbar:
            images, audio, labels = (
                images.to(device),
                audio.to(device),
                labels.to(device),
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images, audio)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        logger.debug(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                "epoch": epoch + 1,
                "model_state": (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                ),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }
            torch.save(state, save_path)
            logger.info(
                f"✅ Saved best model to {save_path} (val_acc={best_val_acc:.4f})"
            )

    logger.info(f"🎯 Training finished. Best val_acc: {best_val_acc:.4f}")
