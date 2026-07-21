"""
==========================================================
File : model_loader.py

Description
-----------
Loads the trained PyTorch model for inference.

Responsibilities
----------------
- Select inference device
- Build model
- Load checkpoint
- Restore model weights
- Move model to device
- Switch model to evaluation mode

Author:
    Varsha Shukla
==========================================================
"""

from pathlib import Path
import logging

import torch
import torch.nn as nn

from models.cnn import SimpleCNN


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads a trained model for inference.
    """

    def __init__(self, config: dict):

        self.config = config

        self.device = self._select_device()

    # =======================================================
    # Device Selection
    # =======================================================

    def _select_device(self) -> torch.device:
        """
        Select CPU/GPU device.
        """

        device_name = self.config["training"]["device"]

        if device_name == "auto":

            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(device_name)

    # =======================================================
    # Build Model
    # =======================================================

    def _build_model(self) -> nn.Module:
        """
        Build CNN architecture.
        """

        num_classes = self.config["model"]["num_classes"]

        model = SimpleCNN(
            num_classes=num_classes
        )

        return model

    # =======================================================
    # Checkpoint Path
    # =======================================================

    def _checkpoint_path(self) -> Path:
        """
        Return checkpoint path.
        """

        checkpoint_cfg = self.config["checkpoint"]

        return (
            Path(checkpoint_cfg["save_dir"])
            / checkpoint_cfg["filename"]
        )

    # =======================================================
    # Load Model
    # =======================================================

    def load(self) -> nn.Module:
        """
        Load trained model.

        Returns
        -------
        nn.Module
            Ready-to-use inference model.
        """

        logger.info("Building model...")

        model = self._build_model()

        checkpoint_path = self._checkpoint_path()

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found:\n{checkpoint_path}"
            )

        logger.info(
            "Loading checkpoint: %s",
            checkpoint_path,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        if "model_state_dict" not in checkpoint:

            raise KeyError(
                "model_state_dict missing from checkpoint."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(self.device)

        model.eval()

        logger.info(
            "Model loaded successfully."
        )

        logger.info(
            "Inference Device : %s",
            self.device,
        )

        logger.info(
            "Validation Accuracy : %.2f%%",
            checkpoint.get(
                "validation_accuracy",
                0.0,
            ),
        )

        return model

    # =======================================================
    # Device Property
    # =======================================================

    @property
    def inference_device(self):

        return self.device