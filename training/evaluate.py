"""
Model Evaluation Script

Responsibilities
----------------
1. Load configuration
2. Select device
3. Load test dataset
4. Load trained model checkpoint
5. Evaluate on test dataset
6. Print final metrics
"""

from pathlib import Path

import yaml

import torch
import torch.nn as nn

from models.cnn import SimpleCNN
from training.dataset import DatasetManager


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load YAML configuration.
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def get_device(config: dict) -> torch.device:
    """
    Select training device.
    """

    device_name = config["training"]["device"]

    if device_name == "auto":

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def load_checkpoint(
    model: nn.Module,
    config: dict,
    device: torch.device,
) -> nn.Module:
    """
    Load best saved model.
    """

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = Path(checkpoint_cfg["save_dir"]) / checkpoint_cfg["filename"]

    if not checkpoint_path.exists():

        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 60)
    print("Checkpoint Loaded")
    print("=" * 60)
    print(f"Path               : {checkpoint_path}")
    print(f"Saved Epoch        : {checkpoint['epoch']}")
    print(f"Validation Accuracy: " f"{checkpoint['validation_accuracy']:.2f}%")

    return model


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate model on test dataset.
    """

    model.eval()

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()

            total_samples += batch_size

    test_loss = running_loss / total_samples

    test_accuracy = (correct_predictions / total_samples) * 100

    metrics = {
        "loss": test_loss,
        "accuracy": test_accuracy,
    }

    return metrics


def main():
    """
    Main evaluation pipeline.
    """

    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = load_config()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = get_device(config)

    print(f"Evaluation Device : {device}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = DatasetManager(
        dataset_root=config["dataset"]["root"],
        image_size=config["dataset"]["image_size"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    (
        _,
        _,
        test_loader,
        class_names,
    ) = dataset.create_dataloaders()

    print(f"Classes : {class_names}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = SimpleCNN(num_classes=len(class_names))

    model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    model.to(device)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("Final Test Results")
    print("=" * 70)

    print(f"Test Loss      : {metrics['loss']:.4f}")

    print(f"Test Accuracy  : {metrics['accuracy']:.2f}%")

    print("=" * 70)


if __name__ == "__main__":

    main()
