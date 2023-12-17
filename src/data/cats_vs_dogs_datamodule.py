import os
from typing import Tuple, Optional, Any

import albumentations
import numpy as np
from PIL import Image
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, random_split, DataLoader


class CatsVSDogsDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.classes = ['Dog', 'Cat']
        self.filepaths = []

        self.rescaler = albumentations.SmallestMaxSize(max_size=256)
        self.cropper = albumentations.CenterCrop(height=256, width=256)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        Image.open(filepath)
                        self.filepaths.append(filepath)
                    except IOError:
                        print(f"Cannot open image at {filepath}. Skipping.")

    def __getitem__(self, idx: int) -> Tensor:
        image = Image.open(self.filepaths[idx])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __len__(self):
        return len(self.filepaths)


class CatsVSDogsDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = './data',
            train_val_test_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 2

    def setup(self, stage: str) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            full_dataset = CatsVSDogsDataset(self.hparams.data_dir)

            train_size = int(len(full_dataset) * self.hparams.train_val_test_ratio[0])
            val_size = int(len(full_dataset) * self.hparams.train_val_test_ratio[1])
            test_size = len(full_dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset,
                                                                                   [train_size, val_size, test_size])

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
