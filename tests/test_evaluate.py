"""
Unit tests for evaluation pipeline.
"""

import torch
import torch.nn as nn

from training.train import save_checkpoint
from training.evaluate import (
    load_checkpoint,
    evaluate,
)

# ---------------------------------------------------------
# Load Checkpoint
# ---------------------------------------------------------


def test_load_checkpoint(
    model,
    config,
    device,
):
    """
    Verify checkpoint loads successfully.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=95.5,
        config=config,
    )

    loaded_model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    assert loaded_model is not None


# ---------------------------------------------------------
# Model State Dict
# ---------------------------------------------------------


def test_model_state_dict_loaded(
    model,
    config,
    device,
):
    """
    Verify model parameters exist after loading.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90.0,
        config=config,
    )

    loaded_model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    assert len(loaded_model.state_dict()) > 0


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------


def test_evaluate(
    model,
    test_loader,
    config,
    device,
):
    """
    Verify evaluation executes.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90,
        config=config,
    )

    model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(
        metrics,
        dict,
    )


# ---------------------------------------------------------
# Metrics Keys
# ---------------------------------------------------------


def test_metrics_keys(
    model,
    test_loader,
    config,
    device,
):
    """
    Verify metric dictionary keys.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90,
        config=config,
    )

    model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    assert "loss" in metrics

    assert "accuracy" in metrics


# ---------------------------------------------------------
# Loss
# ---------------------------------------------------------


def test_loss_positive(
    model,
    test_loader,
    config,
    device,
):
    """
    Test loss should be non-negative.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90,
        config=config,
    )

    model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    assert metrics["loss"] >= 0


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------


def test_accuracy_range(
    model,
    test_loader,
    config,
    device,
):
    """
    Accuracy should lie between
    0 and 100.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90,
        config=config,
    )

    model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    assert 0 <= metrics["accuracy"] <= 100


# ---------------------------------------------------------
# Evaluation Mode
# ---------------------------------------------------------


def test_model_eval_mode(
    model,
    test_loader,
    config,
    device,
):
    """
    Model should be in evaluation mode.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90,
        config=config,
    )

    model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    assert model.training is False


# ---------------------------------------------------------
# Finite Metrics
# ---------------------------------------------------------


def test_metrics_are_finite(
    model,
    test_loader,
    config,
    device,
):
    """
    Verify metrics are finite.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90,
        config=config,
    )

    model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    assert torch.isfinite(torch.tensor(metrics["loss"]))

    assert torch.isfinite(torch.tensor(metrics["accuracy"]))


# ---------------------------------------------------------
# Evaluation Returns Float
# ---------------------------------------------------------


def test_metrics_type(
    model,
    test_loader,
    config,
    device,
):
    """
    Verify returned metric types.
    """

    save_checkpoint(
        model=model,
        epoch=1,
        val_accuracy=90,
        config=config,
    )

    model = load_checkpoint(
        model=model,
        config=config,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(
        metrics["loss"],
        float,
    )

    assert isinstance(
        metrics["accuracy"],
        float,
    )
