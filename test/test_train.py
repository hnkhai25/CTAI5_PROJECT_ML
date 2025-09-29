import unittest
from unittest.mock import patch, MagicMock
from loguru import logger
import torch
import torch.nn as nn
import pandas as pd
import os
import shutil
import sys

project_path = os.path.dirname(os.path.dirname(__file__))
logger.info(project_path)
sys.path.append(project_path)

# --- Mock Modules ---
# Mock the core modules before they are imported by the training script.
# This ensures that when `train.py` runs, it imports our fake versions.
sys.modules["config"] = MagicMock()
sys.modules["model"] = MagicMock()
sys.modules["dataset"] = MagicMock()
sys.modules["utils"] = MagicMock()

import config
from src.model import MultimodalEmotionRecognizer
from src.dataset import EmotionDataset
from src.utils import setup_finetune, build_optimizer

# --- Define Dummy Components ---

# 1. Dummy Configuration
# We define a minimal set of configurations needed for the test to run quickly.
config.MODEL_SAVE_PATH = "./dummy_checkpoints/dummy_model.pth"
config.CSV_PATH = "dummy.csv"
config.SEED = 42
config.BATCH_SIZE = 2  # Small batch size for speed
config.EPOCHS = 2  # Only a couple of epochs are needed to test the loop
config.NUM_CLASSES = 6
config.SEQ_LEN = 3
config.IMAGE_SIZE = 32  # Smaller images
config.SAMPLE_RATE = 16000
config.WAVE_TARGET_LEN = 16000  # 1 second of audio
config.FREEZE_BACKBONES = False  # No backbones to freeze in the dummy model
config.LR_HEAD = 1e-3
config.LR_BACKBONE = 1e-4
config.WEIGHT_DECAY_HEAD = 1e-4
config.WEIGHT_DECAY_BACKBONE = 1e-5
config.FUSION_TYPE = "crossattn"
config.IMAGE_BACKBONE = "resnet18"
config.AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


# 2. Dummy Model
# A simple model that can be instantiated and trained on random data.
class DummyTrainingModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # A single layer is enough to test backpropagation
        self.fc = nn.Linear(config.IMAGE_SIZE, config.NUM_CLASSES)

    def forward(self, img, wave):
        # A minimal forward pass that uses one of the inputs to create a loss
        # The input shape is (B, T, C, H, W)
        # We reduce it to a single value per batch item to feed the linear layer.
        img_reduced = img.mean(dim=[1, 2, 3, 4])  # -> (B)
        # Unsqueeze to create a fake feature dimension
        return self.fc(img_reduced.unsqueeze(1))


# 3. Dummy Dataset
# Generates random tensors on the fly, avoiding any file access.
class DummyTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, df, is_train=True):
        self._len = len(df)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        image = torch.randn(config.SEQ_LEN, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        audio = torch.randn(config.WAVE_TARGET_LEN)
        label = torch.randint(0, config.NUM_CLASSES, (1,)).long().squeeze()
        return image, audio, label


# 4. Dummy Optimizer Builder
# A simple function to create an optimizer for our dummy model.
def dummy_build_optimizer(model, **kwargs):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


# --- Link Dummies to Mocks ---
# Replace the actual classes/functions in the mocked modules with our dummies.
MultimodalEmotionRecognizer = DummyTrainingModel
EmotionDataset = DummyTrainingDataset
setup_finetune.return_value = None  # Mock does nothing
build_optimizer = dummy_build_optimizer


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        """Set up the directory for saving test artifacts."""
        self.checkpoint_dir = os.path.dirname(config.MODEL_SAVE_PATH)
        # The test will run from a clean state
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

    def tearDown(self):
        """Clean up test artifacts."""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

    @patch("pandas.read_csv")
    @patch("builtins.print")
    def test_train_end_to_end(self, mock_print, mock_read_csv):
        """
        Tests the entire training script by using mocks for data, models, and file I/O.
        """
        # Configure the mock for pandas.read_csv to return a fake dataframe
        dummy_df = pd.DataFrame(
            {
                "emotion": ["HAP"] * 10
                + ["SAD"] * 10,  # Dummy labels for stratification
                # Other columns are not used by the dummy dataset but are needed for splitting
                "video_frame_dir": ["-"] * 20,
                "audio_path": ["-"] * 20,
            }
        )
        mock_read_csv.return_value = dummy_df

        # Import the main function from the training script *after* mocks are set up
        from src.train import main

        # --- Run the training process ---
        main()

        # --- Assertions ---
        # 1. Check that the model checkpoint file was created successfully.
        self.assertTrue(
            os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
        )

        # 2. Check the console output to verify that key training stages were logged.
        printed_output = "".join(
            [str(call.args[0]) for call in mock_print.call_args_list]
        )
        self.assertIn("Using device", printed_output)
        self.assertIn("Epoch 1/2", printed_output)
        self.assertIn("Epoch 2/2", printed_output)
        self.assertIn("Train Loss", printed_output)
        self.assertIn("Val Loss", printed_output)
        self.assertIn("✅ Saved best model", printed_output)
        self.assertIn("🎯 Training finished", printed_output)


if __name__ == "__main__":
    # We must remove the script's own name from sys.argv for unittest to run correctly
    sys.argv = [sys.argv[0]]
    unittest.main()
