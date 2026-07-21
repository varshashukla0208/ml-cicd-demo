"""
Unit tests for dataset pipeline.
"""

import torch

# ---------------------------------------------------------
# Dataset Manager
# ---------------------------------------------------------


def test_dataset_manager_created(dataset_manager):
    """
    Verify DatasetManager is created successfully.
    """

    assert dataset_manager is not None


# ---------------------------------------------------------
# Train Loader
# ---------------------------------------------------------


def test_train_loader_exists(train_loader):
    """
    Verify training dataloader exists.
    """

    assert train_loader is not None

    assert len(train_loader) > 0


# ---------------------------------------------------------
# Validation Loader
# ---------------------------------------------------------


def test_validation_loader_exists(validation_loader):
    """
    Verify validation dataloader exists.
    """

    assert validation_loader is not None

    assert len(validation_loader) > 0


# ---------------------------------------------------------
# Test Loader
# ---------------------------------------------------------


def test_test_loader_exists(test_loader):
    """
    Verify test dataloader exists.
    """

    assert test_loader is not None

    assert len(test_loader) > 0


# ---------------------------------------------------------
# Class Names
# ---------------------------------------------------------


def test_class_names(class_names):
    """
    Verify class names.
    """

    assert isinstance(class_names, list)

    assert len(class_names) == 2

    assert "cats" in class_names

    assert "dogs" in class_names


# ---------------------------------------------------------
# Train Batch Shape
# ---------------------------------------------------------


def test_train_batch_shape(train_loader):
    """
    Verify train batch dimensions.
    """

    images, labels = next(iter(train_loader))

    assert images.ndim == 4

    assert images.shape[1] == 3

    assert images.shape[2] == 128

    assert images.shape[3] == 128

    assert labels.ndim == 1

    assert images.size(0) == labels.size(0)


# ---------------------------------------------------------
# Validation Batch Shape
# ---------------------------------------------------------


def test_validation_batch_shape(validation_loader):
    """
    Verify validation batch dimensions.
    """

    images, labels = next(iter(validation_loader))

    assert images.ndim == 4

    assert images.shape[1] == 3

    assert images.shape[2] == 128

    assert images.shape[3] == 128

    assert labels.ndim == 1

    assert images.size(0) == labels.size(0)


# ---------------------------------------------------------
# Test Batch Shape
# ---------------------------------------------------------


def test_test_batch_shape(test_loader):
    """
    Verify test batch dimensions.
    """

    images, labels = next(iter(test_loader))

    assert images.ndim == 4

    assert images.shape[1] == 3

    assert images.shape[2] == 128

    assert images.shape[3] == 128

    assert labels.ndim == 1

    assert images.size(0) == labels.size(0)


# ---------------------------------------------------------
# Image Data Type
# ---------------------------------------------------------


def test_image_dtype(train_loader):
    """
    Verify image datatype.
    """

    images, _ = next(iter(train_loader))

    assert images.dtype == torch.float32


# ---------------------------------------------------------
# Label Data Type
# ---------------------------------------------------------


def test_label_dtype(train_loader):
    """
    Verify label datatype.
    """

    _, labels = next(iter(train_loader))

    assert labels.dtype == torch.int64


# ---------------------------------------------------------
# Batch Size
# ---------------------------------------------------------


def test_batch_size(train_loader, config):
    """
    Verify batch size.
    """

    images, labels = next(iter(train_loader))

    expected_batch = config["dataset"]["batch_size"]

    assert images.shape[0] <= expected_batch

    assert labels.shape[0] <= expected_batch


# ---------------------------------------------------------
# Image Range
# ---------------------------------------------------------


def test_image_tensor(train_loader):
    """
    Verify image tensor.
    """

    images, _ = next(iter(train_loader))

    assert torch.is_tensor(images)


# ---------------------------------------------------------
# Label Tensor
# ---------------------------------------------------------


def test_label_tensor(train_loader):
    """
    Verify label tensor.
    """

    _, labels = next(iter(train_loader))

    assert torch.is_tensor(labels)


# ---------------------------------------------------------
# Number of Classes
# ---------------------------------------------------------


def test_number_of_classes(class_names):
    """
    Verify dataset contains exactly
    two classes.
    """

    assert len(class_names) == 2


# ---------------------------------------------------------
# Dataset Is Not Empty
# ---------------------------------------------------------


def test_dataset_not_empty(train_loader):
    """
    Verify dataset contains samples.
    """

    assert len(train_loader.dataset) > 0


# ---------------------------------------------------------
# Dataset Length
# ---------------------------------------------------------


def test_dataset_lengths(
    train_loader,
    validation_loader,
    test_loader,
):
    """
    Verify dataset splits.
    """

    assert len(train_loader.dataset) == 200

    assert len(validation_loader.dataset) == 40

    assert len(test_loader.dataset) == 40
