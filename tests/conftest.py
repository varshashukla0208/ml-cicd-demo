"""
Shared pytest fixtures.

These fixtures are automatically available
to every test file inside the tests package.
"""

from pathlib import Path

import pytest
import yaml
import torch

from models.cnn import SimpleCNN
from training.dataset import DatasetManager


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def config():
    """
    Load configuration once for the
    entire test session.
    """

    config_path = Path("configs/config.yaml")

    with open(config_path, "r") as file:

        config = yaml.safe_load(file)

    return config


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def device(config):
    """
    Testing device.
    """

    device_name = config["training"]["device"]

    if device_name == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(device_name)


# ---------------------------------------------------------
# Dataset Manager
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def dataset_manager(config):
    """
    Dataset manager.
    """

    return DatasetManager(

        dataset_root=config["dataset"]["root"],

        image_size=config["dataset"]["image_size"],

        batch_size=config["dataset"]["batch_size"],

        num_workers=config["dataset"]["num_workers"],

    )


# ---------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def dataloaders(dataset_manager):
    """
    Return train, validation,
    and test dataloaders.
    """

    return dataset_manager.create_dataloaders()


# ---------------------------------------------------------
# Train Loader
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def train_loader(dataloaders):

    train_loader, _, _, _ = dataloaders

    return train_loader


# ---------------------------------------------------------
# Validation Loader
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def validation_loader(dataloaders):

    _, validation_loader, _, _ = dataloaders

    return validation_loader


# ---------------------------------------------------------
# Test Loader
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def test_loader(dataloaders):

    _, _, test_loader, _ = dataloaders

    return test_loader


# ---------------------------------------------------------
# Class Names
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def class_names(dataloaders):

    _, _, _, classes = dataloaders

    return classes


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

@pytest.fixture
def model(class_names):
    """
    Fresh CNN model.

    Function scope ensures every test
    receives a new model.
    """

    return SimpleCNN(
        num_classes=len(class_names)
    )


# ---------------------------------------------------------
# Loss Function
# ---------------------------------------------------------

@pytest.fixture
def criterion():

    return torch.nn.CrossEntropyLoss()


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

@pytest.fixture
def optimizer(model):

    return torch.optim.Adam(

        model.parameters(),

        lr=0.001,

    )


# ---------------------------------------------------------
# Dummy Batch
# ---------------------------------------------------------

@pytest.fixture
def dummy_batch():

    images = torch.randn(
        8,
        3,
        128,
        128,
    )

    labels = torch.randint(
        0,
        2,
        (8,),
    )

    return images, labels