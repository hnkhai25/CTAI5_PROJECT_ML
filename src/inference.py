import torch
import torch.nn as nn
import cv2
import numpy as np
import soundfile as sf
import torchaudio
import os
import subprocess
from torchvision import transforms
from loguru import logger

from src import config
from src.model import MultimodalEmotionRecognizer

# --- Configuration ---
MODEL_PATH = config.MODEL_SAVE_PATH
# Create an inverse mapping from index to emotion label
IDX_TO_EMOTION = {v: k for k, v in config.EMOTION_MAP.items()}

# --- Preprocessing Functions ---


def extract_audio(video_path, audio_output_path):
    """Extracts audio from a video file using ffmpeg."""
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:a",
        "0",
        "-map",
        "a",
        "-ar",
        str(config.SAMPLE_RATE),
        "-ac",
        "1",
        "-y",
        audio_output_path,
    ]
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.debug(f"Audio extracted successfully to {audio_output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error extracting audio: {e.stderr.decode()}")
        return False


def extract_frames(video_path, num_frames=config.SEQ_LEN):
    """Extracts a specified number of frames, evenly spaced, from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        logger.error("Error: Video file could not be opened or is empty.")
        return []

    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    logger.debug(f"Extracted {len(frames)} frames from video.")
    return frames


def preprocess_input(video_path):
    """
    Takes a video file path, extracts frames and audio, and preprocesses them
    into tensors ready for the model.
    """
    # 1. Extract frames
    frames = extract_frames(video_path, num_frames=config.SEQ_LEN)
    if not frames:
        raise ValueError("Could not extract frames from the video.")

    # 2. Extract audio
    temp_audio_path = "temp_audio.wav"
    if not extract_audio(video_path, temp_audio_path):
        raise ValueError("Could not extract audio from the video.")

    # 3. Process frames
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensors = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensors.append(image_transform(frame_rgb))

    # If fewer frames were extracted than needed, duplicate the last one
    while len(image_tensors) < config.SEQ_LEN:
        image_tensors.append(image_tensors[-1])

    images_tensor = torch.stack(image_tensors, dim=0).unsqueeze(
        0
    )  # Add batch dimension

    # 4. Process audio
    waveform, sr = sf.read(temp_audio_path, dtype="float32")
    waveform = torch.from_numpy(waveform).unsqueeze(0)  # Add channel dimension

    if sr != config.SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)

    # Normalize and pad/truncate
    maxv = waveform.abs().max().clamp_min(1e-8)
    waveform = (waveform / maxv).clamp(-1.0, 1.0)

    L = waveform.size(-1)
    tgt = config.WAVE_TARGET_LEN
    if L > tgt:
        start = (L - tgt) // 2
        waveform = waveform[:, start : start + tgt]
    elif L < tgt:
        pad = tgt - L
        waveform = torch.nn.functional.pad(waveform, (0, pad), "constant", 0)

    audio_tensor = waveform.squeeze(0).unsqueeze(
        0
    )  # Remove channel, add batch dimension

    # Clean up temporary audio file
    os.remove(temp_audio_path)

    return images_tensor, audio_tensor


def predict(model, image_tensor, audio_tensor, device):
    """Runs a forward pass and returns the predicted emotion label."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        audio_tensor = audio_tensor.to(device)

        logits = model(image_tensor, audio_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_emotion = IDX_TO_EMOTION[predicted_idx.item()]

    return predicted_emotion, confidence.item()


def main(video_path):
    """Main function to load the model and perform inference."""
    if not os.path.exists(video_path):
        logger.error(f"Error: Video file not found at '{video_path}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Model Structure
    model = MultimodalEmotionRecognizer(
        num_classes=config.NUM_CLASSES,
        fusion=config.FUSION_TYPE,
        image_backbone=config.IMAGE_BACKBONE,
        ast_model_id=config.AST_MODEL_ID,
        T=config.SEQ_LEN,
    ).to(device)

    # 2. Load Trained Weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # Handle models saved with DataParallel
        state_dict = checkpoint["model_state"]
        if next(iter(state_dict)).startswith("module."):
            # create a new OrderedDict that does not contain `module.`
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(
            f"Error: Model checkpoint not found at {MODEL_PATH}. Please train the model first."
        )
        return
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        return

    # 3. Preprocess Input
    try:
        image_tensor, audio_tensor = preprocess_input(video_path)
    except ValueError as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    # 4. Predict
    emotion, confidence = predict(model, image_tensor, audio_tensor, device)

    logger.info("\n--- Inference Result ---")
    logger.info(f"Predicted Emotion: {emotion}")
    logger.info(f"Confidence: {confidence:.2%}")
    logger.info("------------------------")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run emotion recognition inference on a video file."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    args = parser.parse_args()

    main(args.video_path)
