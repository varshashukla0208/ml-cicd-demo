"""
Unit tests for the training pipeline.
"""

import copy

import torch

from training.train import (
    build_loss,
    build_optimizer,
    train_one_epoch,
    validate_one_epoch,
)


# ---------------------------------------------------------
# Loss Function
# ---------------------------------------------------------

def test_build_loss(config):
    """
    Verify loss function creation.
    """

    criterion = build_loss(config)

    assert criterion is not None
    assert isinstance(
        criterion,
        torch.nn.CrossEntropyLoss,
    )


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

def test_build_optimizer(
    model,
    config,
):
    """
    Verify optimizer creation.
    """

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    assert optimizer is not None

    optimizer_name = config["optimizer"]["name"]

    if optimizer_name == "Adam":
        assert isinstance(
            optimizer,
            torch.optim.Adam,
        )

    elif optimizer_name == "AdamW":
        assert isinstance(
            optimizer,
            torch.optim.AdamW,
        )

    elif optimizer_name == "SGD":
        assert isinstance(
            optimizer,
            torch.optim.SGD,
        )


# ---------------------------------------------------------
# One Training Epoch
# ---------------------------------------------------------

def test_train_one_epoch(
    model,
    train_loader,
    device,
    config,
):
    """
    Verify one epoch of training executes.
    """

    model.to(device)

    criterion = build_loss(config)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    metrics = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    assert isinstance(metrics, dict)

    assert "loss" in metrics

    assert "accuracy" in metrics


# ---------------------------------------------------------
# Train Loss
# ---------------------------------------------------------

def test_train_loss_positive(
    model,
    train_loader,
    device,
    config,
):
    """
    Training loss should be valid.
    """

    model.to(device)

    criterion = build_loss(config)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    metrics = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    assert metrics["loss"] >= 0


# ---------------------------------------------------------
# Train Accuracy
# ---------------------------------------------------------

def test_train_accuracy_range(
    model,
    train_loader,
    device,
    config,
):
    """
    Accuracy should lie between
    0 and 100.
    """

    model.to(device)

    criterion = build_loss(config)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    metrics = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    assert 0 <= metrics["accuracy"] <= 100


# ---------------------------------------------------------
# Validation Epoch
# ---------------------------------------------------------

def test_validate_one_epoch(
    model,
    validation_loader,
    device,
    config,
):
    """
    Verify validation executes.
    """

    model.to(device)

    criterion = build_loss(config)

    metrics = validate_one_epoch(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(metrics, dict)

    assert "loss" in metrics

    assert "accuracy" in metrics


# ---------------------------------------------------------
# Validation Loss
# ---------------------------------------------------------

def test_validation_loss_positive(
    model,
    validation_loader,
    device,
    config,
):
    """
    Validation loss should be valid.
    """

    model.to(device)

    criterion = build_loss(config)

    metrics = validate_one_epoch(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
    )

    assert metrics["loss"] >= 0


# ---------------------------------------------------------
# Validation Accuracy
# ---------------------------------------------------------

def test_validation_accuracy_range(
    model,
    validation_loader,
    device,
    config,
):
    """
    Validation accuracy should
    lie between 0 and 100.
    """

    model.to(device)

    criterion = build_loss(config)

    metrics = validate_one_epoch(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
    )

    assert 0 <= metrics["accuracy"] <= 100


# ---------------------------------------------------------
# Parameters Updated
# ---------------------------------------------------------

def test_parameters_updated(
    model,
    train_loader,
    device,
    config,
):
    """
    Verify model weights change
    after training.
    """

    model.to(device)

    criterion = build_loss(config)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    before = copy.deepcopy(
        model.state_dict()
    )

    train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    after = model.state_dict()

    updated = any(
        not torch.equal(before[key], after[key])
        for key in before
    )

    assert updated


# ---------------------------------------------------------
# Optimizer State
# ---------------------------------------------------------

def test_optimizer_has_param_groups(
    model,
    config,
):
    """
    Optimizer should contain
    parameter groups.
    """

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    assert len(
        optimizer.param_groups
    ) > 0


# ---------------------------------------------------------
# Model Train Mode
# ---------------------------------------------------------

def test_model_train_mode(
    model,
    train_loader,
    device,
    config,
):
    """
    Model should remain
    in training mode.
    """

    model.to(device)

    criterion = build_loss(config)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    assert model.training


# ---------------------------------------------------------
# Model Eval Mode
# ---------------------------------------------------------

def test_model_eval_mode(
    model,
    validation_loader,
    device,
    config,
):
    """
    Validation should switch
    model into evaluation mode.
    """

    model.to(device)

    criterion = build_loss(config)

    validate_one_epoch(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
    )

    assert model.training is False