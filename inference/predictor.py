"""
==========================================================
File : predictor.py

Description
-----------
Production inference pipeline.

Responsibilities
----------------
- Receive uploaded image
- Preprocess image
- Run model inference
- Postprocess predictions
- Measure inference latency
- Return prediction response

Author:
    Varsha Shukla
==========================================================
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from api.schemas import PredictionResponse

from inference.preprocess import ImagePreprocessor
from inference.postprocess import PostProcessor


class Predictor:
    """
    Production inference pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
    ):

        self.model = model

        self.device = device

        self.config = config

        self.preprocessor = ImagePreprocessor(
            image_size=config["dataset"]["image_size"]
        )

        self.postprocessor = PostProcessor(
            class_names=config["classes"]
        )

    # ======================================================
    # Predict
    # ======================================================

    @torch.inference_mode()
    def predict(
        self,
        image_bytes: bytes,
    ) -> PredictionResponse:
        """
        Predict image class.

        Parameters
        ----------
        image_bytes : bytes

        Returns
        -------
        PredictionResponse
        """

        start_time = time.perf_counter()

        # -----------------------------------------------
        # Image preprocessing
        # -----------------------------------------------

        image_tensor = self.preprocessor.preprocess(
            image_bytes=image_bytes,
            device=self.device,
        )

        # -----------------------------------------------
        # Forward pass
        # -----------------------------------------------

        outputs = self.model(image_tensor)

        # -----------------------------------------------
        # Postprocess
        # -----------------------------------------------

        result = self.postprocessor.process(
            outputs
        )

        # -----------------------------------------------
        # Inference latency
        # -----------------------------------------------

        inference_time = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return PredictionResponse(

            predicted_class=result.predicted_class,

            confidence=result.confidence,

            inference_time_ms=round(
                inference_time,
                2,
            ),

            model_version="1.0.0",
        )