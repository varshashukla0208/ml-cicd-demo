"""
Unit tests for checkpoint saving.
"""

from pathlib import Path

import torch

from training.train import save_checkpoint


# ---------------------------------------------------------
# Save Checkpoint
# ---------------------------------------------------------

def test_save_checkpoint(
    model,
    config,
):
    """
    Verify checkpoint file is created.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=95.5,
        config=config,
    )

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = (
        Path(checkpoint_cfg["save_dir"])
        / checkpoint_cfg["filename"]
    )

    assert checkpoint_path.exists()


# ---------------------------------------------------------
# Load Checkpoint
# ---------------------------------------------------------

def test_load_checkpoint(
    model,
    config,
):
    """
    Verify checkpoint loads.
    """

    save_checkpoint(
        model=model,
        epoch=5,
        val_accuracy=91.2,
        config=config,
    )

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = (
        Path(checkpoint_cfg["save_dir"])
        / checkpoint_cfg["filename"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint is not None


# ---------------------------------------------------------
# Checkpoint Keys
# ---------------------------------------------------------

def test_checkpoint_keys(
    model,
    config,
):
    """
    Verify checkpoint keys.
    """

    save_checkpoint(
        model=model,
        epoch=3,
        val_accuracy=88.6,
        config=config,
    )

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = (
        Path(checkpoint_cfg["save_dir"])
        / checkpoint_cfg["filename"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert "epoch" in checkpoint

    assert "model_state_dict" in checkpoint

    assert "validation_accuracy" in checkpoint


# ---------------------------------------------------------
# Epoch
# ---------------------------------------------------------

def test_checkpoint_epoch(
    model,
    config,
):
    """
    Verify epoch saved.
    """

    save_checkpoint(
        model=model,
        epoch=11,
        val_accuracy=90.5,
        config=config,
    )

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = (
        Path(checkpoint_cfg["save_dir"])
        / checkpoint_cfg["filename"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 11


# ---------------------------------------------------------
# Validation Accuracy
# ---------------------------------------------------------

def test_validation_accuracy(
    model,
    config,
):
    """
    Verify validation accuracy.
    """

    save_checkpoint(
        model=model,
        epoch=7,
        val_accuracy=97.25,
        config=config,
    )

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = (
        Path(checkpoint_cfg["save_dir"])
        / checkpoint_cfg["filename"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint["validation_accuracy"] == 97.25


# ---------------------------------------------------------
# Restore Model
# ---------------------------------------------------------

def test_restore_model(
    model,
    config,
):
    """
    Verify model weights restore correctly.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90,
        config=config,
    )

    original = {
        k: v.clone()
        for k, v in model.state_dict().items()
    }

    for param in model.parameters():
        param.data.zero_()

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = (
        Path(checkpoint_cfg["save_dir"])
        / checkpoint_cfg["filename"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    restored = model.state_dict()

    for key in original:

        assert torch.equal(
            original[key],
            restored[key],
        )


# ---------------------------------------------------------
# File Exists
# ---------------------------------------------------------

def test_checkpoint_exists(
    model,
    config,
):
    """
    Verify checkpoint exists.
    """

    save_checkpoint(
        model=model,
        epoch=2,
        val_accuracy=90,
        config=config,
    )

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = (
        Path(checkpoint_cfg["save_dir"])
        / checkpoint_cfg["filename"]
    )

    assert checkpoint_path.exists()


# ---------------------------------------------------------
# File Not Empty
# ---------------------------------------------------------

def test_checkpoint_not_empty(
    model,
    config,
):
    """
    Verify checkpoint isn't empty.
    """

    save_checkpoint(
        model=model,
        epoch=2,
        val_accuracy=90,
        config=config,
    )

    checkpoint_cfg = config["checkpoint"]

    checkpoint_path = (
        Path(checkpoint_cfg["save_dir"])
        / checkpoint_cfg["filename"]
    )

    assert checkpoint_path.stat().st_size > 0