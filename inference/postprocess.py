"""
==========================================================
File : postprocess.py

Description
-----------
Converts raw model outputs into user-friendly predictions.

Responsibilities
----------------
- Apply Softmax
- Compute confidence
- Find predicted class
- Convert class index to class label
- Return prediction dictionary

Author:
    Varsha Shukla
==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ==========================================================
# Prediction Result
# ==========================================================


@dataclass(frozen=True)
class PredictionResult:
    """
    Prediction returned by the postprocessor.
    """

    predicted_class: str

    predicted_index: int

    confidence: float


# ==========================================================
# Postprocessor
# ==========================================================


class PostProcessor:
    """
    Converts model logits into prediction results.
    """

    def __init__(
        self,
        class_names: list[str],
    ):

        self.class_names = class_names

    # ======================================================
    # Validate Model Output
    # ======================================================

    def _validate_output(
        self,
        outputs: torch.Tensor,
    ):

        if outputs.ndim != 2:

            raise ValueError(
                "Model output must have shape " "(batch_size, num_classes)."
            )

    # ======================================================
    # Softmax
    # ======================================================

    def _probabilities(
        self,
        outputs: torch.Tensor,
    ) -> torch.Tensor:

        return F.softmax(
            outputs,
            dim=1,
        )

    # ======================================================
    # Prediction
    # ======================================================

    def process(
        self,
        outputs: torch.Tensor,
    ) -> PredictionResult:
        """
        Convert logits into prediction.
        """

        self._validate_output(outputs)

        probabilities = self._probabilities(outputs)

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

        predicted_index = prediction.item()

        predicted_class = self.class_names[predicted_index]

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_index=predicted_index,
            confidence=float(confidence.item()),
        )
