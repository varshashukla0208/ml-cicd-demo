"""
Production Training Script

Responsibilities
----------------
1. Load configuration
2. Set reproducibility
3. Create datasets
4. Build model
5. Train
6. Validate
7. Save checkpoints
8. Support CI/CD execution
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR

from models.cnn import SimpleCNN
from training.dataset import DatasetManager
from utils.mlflow_logger import MLFlowLogger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    dict
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def set_seed(seed: int) -> None:
    """
    Make experiments reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    """
    Decide training device.
    """

    device_name = config["training"]["device"]

    if device_name == "auto":

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def build_loss(config: dict) -> nn.Module:
    """
    Build loss function.
    """

    loss_name = config["loss"]["name"]

    if loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported Loss: {loss_name}")


def build_optimizer(
    model: nn.Module,
    config: dict,
) -> optim.Optimizer:
    """
    Build optimizer from configuration.

    Parameters
    ----------
    model : nn.Module
        Model to optimize.

    config : dict
        YAML configuration.

    Returns
    -------
    torch.optim.Optimizer
    """

    optimizer_cfg = config["optimizer"]

    optimizer_name = optimizer_cfg["name"]
    lr = optimizer_cfg["lr"]
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)

    if optimizer_name == "SGD":

        momentum = optimizer_cfg.get("momentum", 0.9)

        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "Adam":

        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "AdamW":

        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: dict,
):
    """
    Build learning rate scheduler.

    Returns
    -------
    Scheduler or None
    """

    scheduler_cfg = config["scheduler"]

    if not scheduler_cfg["enabled"]:
        return None

    scheduler_name = scheduler_cfg["name"]

    if scheduler_name == "StepLR":

        return StepLR(
            optimizer,
            step_size=scheduler_cfg["step_size"],
            gamma=scheduler_cfg["gamma"],
        )

    elif scheduler_name == "CosineAnnealingLR":

        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg["t_max"],
            eta_min=scheduler_cfg.get(
                "eta_min",
                1e-6,
            ),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    val_accuracy: float,
    config: dict,
) -> Path | None:
    """
    Save model checkpoint.
    """

    checkpoint_cfg = config["checkpoint"]

    if not checkpoint_cfg["enabled"]:
        return None

    save_dir = Path(checkpoint_cfg["save_dir"])

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = save_dir / checkpoint_cfg["filename"]

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "validation_accuracy": val_accuracy,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


class EarlyStopping:
    """
    Stop training when validation accuracy
    does not improve.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_score = None
        self.counter = 0
        self.stop_training = False

    def __call__(
        self,
        validation_accuracy: float,
    ):

        if self.best_score is None:

            self.best_score = validation_accuracy
            return

        improvement = validation_accuracy - self.best_score

        if improvement > self.min_delta:

            self.best_score = validation_accuracy
            self.counter = 0

        else:

            self.counter += 1

            print(f"EarlyStopping Counter: " f"{self.counter}/{self.patience}")

            if self.counter >= self.patience:

                self.stop_training = True


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config,
):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    dataloader : DataLoader
        Training dataloader.

    criterion : nn.Module
        Loss function.

    optimizer : torch.optim.Optimizer
        Optimizer.

    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    dict
        Dictionary containing epoch metrics.
    """

    # -----------------------------------------
    # Switch model to training mode
    # -----------------------------------------
    model.train()
    # -----------------------------------------
    # Smoke Training Configuration
    # -----------------------------------------

    smoke_config = config.get("smoke", {})

    smoke_enabled = smoke_config.get("enabled", False)
    train_batches = smoke_config.get("train_batches", 0)

    # -----------------------------------------
    # Running statistics
    # -----------------------------------------
    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    # -----------------------------------------
    # Iterate over training batches
    # -----------------------------------------
    for batch_idx, (images, labels) in enumerate(dataloader):

        # -----------------------------------------
        # Smoke Training
        # -----------------------------------------

        if smoke_enabled and batch_idx + 1 >= train_batches:
            print(f"Smoke mode: stopping training after " f"{train_batches} batches.")

            break

        # Move data to selected device
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # -------------------------
        # Statistics
        # -------------------------

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()

        total_samples += batch_size

    # -----------------------------------------
    # Epoch metrics
    # -----------------------------------------
    epoch_loss = running_loss / total_samples

    epoch_accuracy = (correct_predictions / total_samples) * 100.0

    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    }

    return metrics


def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    config,
):
    """
    Validate model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    dataloader
        Validation dataloader.

    criterion : nn.Module
        Loss function.

    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    dict
        Validation metrics.
    """

    # ----------------------------------
    # Evaluation mode
    # ----------------------------------

    model.eval()

    # -----------------------------------------
    # Smoke Training Configuration
    # -----------------------------------------

    smoke_config = config.get("smoke", {})

    smoke_enabled = smoke_config.get("enabled", False)
    val_batches = smoke_config.get("val_batches", 0)

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    # ----------------------------------
    # Disable gradient computation
    # ----------------------------------

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(dataloader):

            if smoke_enabled and batch_idx + 1 >= val_batches:
                print(f"Smoke mode enabled. Stopping after {val_batches} batches.")
                break

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()

            total_samples += batch_size

    epoch_loss = running_loss / total_samples

    epoch_accuracy = (correct_predictions / total_samples) * 100.0

    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    }

    return metrics


def main():
    """
    Main training pipeline.
    """

    print("=" * 70)
    print("Starting Training Pipeline")
    print("=" * 70)

    # ---------------------------------------------------
    # Load Configuration
    # ---------------------------------------------------

    args = parse_arguments()

    config = load_config(args.config)

    ml_logger = MLFlowLogger(experiment_name="Image Classification")

    ml_logger.start_run(run_name=config["model"].get("name", "SimpleCNN"))

    try:
        seed = config["training"]["seed"]
        set_seed(seed)
        print(f"Random Seed : {seed}")

        device = get_device(config)
        print(f"Training Device : {device}")

        dataset = DatasetManager(
            dataset_root=config["dataset"]["root"],
            image_size=config["dataset"]["image_size"],
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
        )

        (
            train_loader,
            validation_loader,
            _test_loader,
            class_names,
        ) = dataset.create_dataloaders()

        print(f"Classes : {class_names}")
        print(f"Training batches   : {len(train_loader)}")
        print(f"Validation batches : {len(validation_loader)}")

        model = SimpleCNN(num_classes=len(class_names))
        model.to(device)
        print("Model Created Successfully")

        criterion = build_loss(config)
        print(f"Loss : {config['loss']['name']}")

        optimizer = build_optimizer(
            model,
            config,
        )
        print(f"Optimizer : {config['optimizer']['name']}")

        scheduler = build_scheduler(
            optimizer=optimizer,
            config=config,
        )

        if scheduler is not None:
            print(f"Scheduler : {config['scheduler']['name']}")
        else:
            print("Scheduler : Disabled")

        ml_logger.log_params(
            {
                "model": model.__class__.__name__,
                "epochs": config["training"]["epochs"],
                "batch_size": config["dataset"]["batch_size"],
                "learning_rate": config["optimizer"]["lr"],
                "optimizer": config["optimizer"]["name"],
                "scheduler": (
                    config["scheduler"]["name"]
                    if config["scheduler"]["enabled"]
                    else "disabled"
                ),
                "loss": config["loss"]["name"],
                "device": str(device),
                "num_classes": len(class_names),
            }
        )

        if config["early_stopping"]["enabled"]:
            early_stopping = EarlyStopping(
                patience=config["early_stopping"]["patience"],
            )
            print("Early Stopping : Enabled")
        else:
            early_stopping = None
            print("Early Stopping : Disabled")

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "learning_rate": [],
        }

        best_validation_accuracy = 0.0
        epochs = config["training"]["epochs"]

        print("=" * 70)
        print("Training Started")
        print("=" * 70)

        for epoch in range(epochs):
            print(f"\nEpoch [{epoch + 1}/{epochs}]")

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                config=config,
            )

            validation_metrics = validate_one_epoch(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
                config=config,
            )

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["validation_loss"].append(validation_metrics["loss"])
            history["validation_accuracy"].append(validation_metrics["accuracy"])
            history["learning_rate"].append(current_lr)

            ml_logger.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )

            if validation_metrics["accuracy"] > best_validation_accuracy:
                best_validation_accuracy = validation_metrics["accuracy"]

                checkpoint_path = save_checkpoint(
                    model=model,
                    epoch=epoch + 1,
                    val_accuracy=best_validation_accuracy,
                    config=config,
                )

                if checkpoint_path is not None:
                    ml_logger.log_artifact(
                        str(checkpoint_path),
                        artifact_dir="checkpoints",
                    )

            print(f"Train Loss      : {train_metrics['loss']:.4f}")
            print(f"Train Accuracy  : {train_metrics['accuracy']:.2f}%")
            print(f"Validation Loss : {validation_metrics['loss']:.4f}")
            print(f"Validation Acc  : {validation_metrics['accuracy']:.2f}%")
            print(f"Learning Rate   : {current_lr:.8f}")

            if early_stopping is not None:
                early_stopping(validation_metrics["accuracy"])

                if early_stopping.stop_training:
                    print("\nEarly stopping triggered.")
                    break

        print("\n" + "=" * 70)
        print("Training Completed")
        print("=" * 70)

        ml_logger.log_artifact(
            args.config,
            artifact_dir="config",
        )
        print(f"Best Validation Accuracy : " f"{best_validation_accuracy:.2f}%")

        return history
    finally:
        ml_logger.end_run()


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Production Training Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    return parser.parse_args()


if __name__ == "__main__":

    main()
