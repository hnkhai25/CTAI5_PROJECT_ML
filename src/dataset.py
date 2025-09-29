# dataset.py

import os
import torch
import cv2
import numpy as np
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

from src import config


def load_wav(path: str, target_sr: int, mix_to_mono: bool = True):
    """Loads a waveform from a file."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    if audio.ndim == 1:
        wav = torch.from_numpy(audio).unsqueeze(0)
    else:
        wav = torch.from_numpy(audio).T
        if mix_to_mono:
            wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr


class EmotionDataset(Dataset):
    """Custom dataset for loading video frames and audio."""

    def __init__(self, data_path: str, df: pd.DataFrame, is_train: bool = True) -> None:
        self.data_path = data_path
        self.df = df
        self.is_train = is_train
        self.wave_target_len = config.WAVE_TARGET_LEN

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_frame_dir = self.data_path + row["video_frame_dir"]
        audio_path = self.data_path + row["audio_path"]
        emotion_label = config.EMOTION_MAP[row["emotion"]]

        # Load and process images
        images = []
        frame_files = sorted(os.listdir(video_frame_dir))
        for path in frame_files:
            image_path = os.path.join(video_frame_dir, path)
            frame = cv2.imread(image_path)
            if frame is None:
                frame = np.zeros(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.image_transform(frame)
            images.append(frame)
            if len(images) >= config.SEQ_LEN:
                break

        while len(images) < config.SEQ_LEN:
            images.append(
                images[-1]
                if images
                else torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            )

        images = torch.stack(images, dim=0)

        # Load and process audio
        waveform, _ = load_wav(audio_path, config.SAMPLE_RATE, mix_to_mono=True)
        maxv = waveform.abs().max().clamp_min(1e-8)
        waveform = (waveform / maxv).clamp(-1.0, 1.0)
        waveform = self._pad_or_truncate_waveform(waveform)

        return images, waveform, torch.tensor(emotion_label, dtype=torch.long)

    def _pad_or_truncate_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """Pads or truncates waveform to a target length."""
        L = wav.size(-1)
        tgt = self.wave_target_len

        if L > tgt:
            start = (
                np.random.randint(0, L - tgt + 1) if self.is_train else (L - tgt) // 2
            )
            wav = wav[:, start : start + tgt]
        elif L < tgt:
            pad = tgt - L
            wav = torch.nn.functional.pad(wav, (0, pad), mode="constant", value=0.0)

        return wav.squeeze(0)
