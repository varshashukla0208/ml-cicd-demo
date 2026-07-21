"""
==========================================================
File : preprocess.py

Description
-----------
Image preprocessing pipeline used during inference.

Responsibilities
----------------
- Decode uploaded image
- Convert to RGB
- Resize image
- Convert to tensor
- Normalize pixel values
- Add batch dimension
- Move tensor to device

Author:
    Varsha Shukla
==========================================================
"""

from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms


class ImagePreprocessor:
    """
    Preprocess uploaded images for inference.
    """

    def __init__(
        self,
        image_size: int = 128,
    ):

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        image_size,
                        image_size,
                    )
                ),

                transforms.ToTensor(),
            ]
        )

    # ======================================================
    # Read Image
    # ======================================================

    def _load_image(
        self,
        image_bytes: bytes,
    ) -> Image.Image:
        """
        Decode uploaded image.
        """

        image = Image.open(
            BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        return image

    # ======================================================
    # Transform Image
    # ======================================================

    def _transform(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Apply preprocessing transforms.
        """

        tensor = self.transform(image)

        return tensor

    # ======================================================
    # Public API
    # ======================================================

    def preprocess(
        self,
        image_bytes: bytes,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert uploaded image into
        model-ready tensor.

        Parameters
        ----------
        image_bytes : bytes

        device : torch.device

        Returns
        -------
        torch.Tensor

            Shape:
                (1, 3, 128, 128)
        """

        image = self._load_image(
            image_bytes
        )

        tensor = self._transform(
            image
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)

        return tensor