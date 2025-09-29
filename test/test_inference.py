import unittest
from unittest.mock import patch, MagicMock
from loguru import logger
import torch
import torch.nn as nn
import numpy as np
import os
import sys

project_path = os.path.dirname(os.path.dirname(__file__))
logger.info(project_path)
sys.path.append(project_path)
# Mock the modules that will be imported by the inference script.
# This must be done BEFORE the inference script is imported.
sys.modules["config"] = MagicMock()
sys.modules["model"] = MagicMock()

from src import config

# --- Define Dummy Configuration and Model ---

# Set dummy config values that the inference script will use
config.MODEL_SAVE_PATH = "./dummy_model.pth"
config.EMOTION_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 5}
config.IMAGE_SIZE = 64
config.SEQ_LEN = 3
config.SAMPLE_RATE = 16000
config.WAVE_TARGET_LEN = int(config.SAMPLE_RATE * 4.0)


# A dummy model that matches the expected class name and returns a predictable output shape
class DummyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # The input size (1) is arbitrary as the forward pass will ignore inputs
        self.fc = nn.Linear(1, len(config.EMOTION_MAP))

    def forward(self, img, wave):
        # A simple forward pass that ignores the actual input tensors
        # and returns fixed-shape logits for consistency.
        batch_size = img.shape[0] if img is not None else 1
        return self.fc(torch.randn(batch_size, 1))


# Replace the real model class in the mocked 'model' module with our dummy one
MultimodalEmotionRecognizer = DummyModel
config.NUM_CLASSES = len(config.EMOTION_MAP)
config.FUSION_TYPE = "crossattn"
config.IMAGE_BACKBONE = "resnet18"
config.AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


class TestInferenceSimple(unittest.TestCase):

    def setUp(self):
        """Create a dummy model checkpoint file before the test."""
        self.model_path = config.MODEL_SAVE_PATH
        dummy_model = DummyModel()
        # This checkpoint is the "fake text" part of the test
        torch.save({"model_state": dummy_model.state_dict()}, self.model_path)
        self.fake_video_path = "path/to/any/video.mp4"

    def tearDown(self):
        """Remove the dummy model file after the test."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    @patch("builtins.print")
    @patch("soundfile.read")
    @patch("src.inference.extract_audio")
    @patch("src.inference.extract_frames")
    @patch(
        "os.remove"
    )  # Mock os.remove to prevent errors when it tries to delete temp audio
    def test_inference_with_mocks(
        self,
        mock_os_remove,
        mock_extract_frames,
        mock_extract_audio,
        mock_sf_read,
        mock_print,
    ):
        """
        Tests the inference pipeline by mocking all file I/O and subprocess calls,
        using only a fake model checkpoint on disk.
        """
        # --- Configure Mocks ---
        # 1. Mock extract_frames: returns a list of fake numpy image arrays
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame] * config.SEQ_LEN

        # 2. Mock extract_audio: returns True to simulate successful extraction
        mock_extract_audio.return_value = True

        # 3. Mock soundfile.read: returns a fake audio waveform and sample rate
        dummy_waveform = np.random.randn(config.SAMPLE_RATE * 2).astype("float32")
        mock_sf_read.return_value = (dummy_waveform, config.SAMPLE_RATE)

        # We must import the main function *after* all the mocks are in place
        from src.inference import main

        # --- Run Test ---
        main(self.fake_video_path)

        # --- Assertions ---
        # Verify that our mocked (file-handling) functions were called as expected
        mock_extract_frames.assert_called_once_with(
            self.fake_video_path, num_frames=config.SEQ_LEN
        )
        mock_extract_audio.assert_called_once()
        mock_sf_read.assert_called_once()
        mock_os_remove.assert_called_once()  # Check that it tried to clean up the temp audio file

        # Check the console output to ensure a prediction was printed
        output_found = False
        for call in mock_print.call_args_list:
            line = str(call[0][0])
            if "Predicted Emotion:" in line:
                output_found = True
                predicted_emotion = line.split(":")[1].strip()
                self.assertIn(predicted_emotion, config.EMOTION_MAP.keys())

        self.assertTrue(
            output_found, "The inference script did not print a final prediction."
        )


if __name__ == "__main__":
    # We must remove the script's own name from sys.argv for unittest to run correctly
    sys.argv = [sys.argv[0]]
    unittest.main()
