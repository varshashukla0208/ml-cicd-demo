"""
dataset.py

Production-ready dataset module for image classification.

Responsibilities:
1. Load datasets using ImageFolder
2. Apply preprocessing transforms
3. Create DataLoaders
4. Return datasets, dataloaders, and class names
"""

from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DatasetManager:
    """
    Dataset manager for image classification projects.
    """

    def __init__(
        self,
        dataset_root: str,
        image_size: int = 128,
        batch_size: int = 16,
        num_workers: int = 0,
    ):
        self.dataset_root = Path(dataset_root)

        self.train_dir = self.dataset_root / "train"
        self.val_dir = self.dataset_root / "validation"
        self.test_dir = self.dataset_root / "test"

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load_datasets(self):
        """Load train, validation and test datasets."""

        train_dataset = datasets.ImageFolder(
            root=self.train_dir,
            transform=self.train_transform,
        )

        val_dataset = datasets.ImageFolder(
            root=self.val_dir,
            transform=self.val_transform,
        )

        test_dataset = datasets.ImageFolder(
            root=self.test_dir,
            transform=self.val_transform,
        )

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self):
        """Create DataLoaders."""

        train_dataset, val_dataset, test_dataset = self.load_datasets()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return (
            train_loader,
            val_loader,
            test_loader,
            train_dataset.classes,
        )

    def summary(self):
        """Print dataset information."""

        train_dataset, val_dataset, test_dataset = self.load_datasets()

        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print(f"Train Images      : {len(train_dataset)}")
        print(f"Validation Images : {len(val_dataset)}")
        print(f"Test Images       : {len(test_dataset)}")

        print()

        print(f"Classes           : {train_dataset.classes}")
        print(f"Class Mapping     : {train_dataset.class_to_idx}")

        print("=" * 60)


if __name__ == "__main__":

    dataset = DatasetManager(
        dataset_root="dataset",
        image_size=128,
        batch_size=16,
        num_workers=0,
    )

    dataset.summary()

    train_loader, val_loader, test_loader, classes = (
        dataset.create_dataloaders()
    )

    print(f"\nClasses : {classes}")

    images, labels = next(iter(train_loader))

    print(f"Image Batch Shape : {images.shape}")
    print(f"Label Shape       : {labels.shape}")