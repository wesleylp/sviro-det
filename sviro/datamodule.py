from typing import Optional

import albumentations as A
import pytorch_lightning as pl
#TODO: implement parameters variations with hydra
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms as transform_lib

from sviro.dataset import SVIRODetection


def collate_fn(batch):
    return tuple(zip(*batch))


class SVIRODataModule(pl.LightningDataModule):

    name = "sviro"
    extra_args = {}

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        train_car: [str | list[str]] = ["x5"],
        val_car: [str | list[str]] = ["zoe"],
        test_car: [str | list[str]] = ["aclass"],
        labels_mapping=None,
        seed: int = 42,
        pin_memory: bool = True,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        """SVIRO DataModule

        Args:
            data_dir (str): Folder containing the dataset
            batch_size (int): batch size
            num_workers (int): number of workers
            train_car (str  |  list[str]], optional): "all" or list of car names. Defaults to ["x5"].
            val_car (str  |  list[str]], optional): "all" or list of car names. Defaults to ["zoe"].
            test_car (str  |  list[str]], optional): "all" or list of car names. Defaults to ["aclass"].
            labels_mapping (None | dict, optional): mapping class names into indexes. Defaults to None.
            seed (int, optional): seed. Defaults to 42.
            pin_memory (bool, optional): See pytorch docs. Defaults to True.
            debug (bool, optional): Load small portion for debugging purposes. Defaults to False.
        """

        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.DATASET = SVIRODetection
        self.train_car = train_car
        self.val_car = val_car
        self.test_car = test_car
        self.labels_mapping = {
            'empty': 0,
            'infant_seat': 1,
            'child_seat': 2,
            'person': 3,
            'everyday_object': 4
        } if labels_mapping is None else labels_mapping
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
        self.debug = debug

    def prepare_data(self) -> None:
        # download, tokenize, etc...
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = (self.default_transforms()
                      if self.train_transforms is None else self.train_transforms)

        dataset = self.DATASET(self.data_dir,
                               car=self.train_car,
                               split='train',
                               transform=transforms,
                               mapping=self.labels_mapping,
                               filter_empty=True,
                               debug=self.debug)

        dataset_length = len(dataset)

        self.train_idxs, self.val_idxs = train_test_split(range(dataset_length),
                                                          test_size=0.25,
                                                          random_state=self.seed)

    def train_dataloader(self) -> DataLoader:
        transforms = (self.default_transforms()
                      if self.train_transforms is None else self.train_transforms)

        train_dataset = self.DATASET(self.data_dir,
                                     car=self.train_car,
                                     split='train',
                                     transform=transforms,
                                     mapping=self.labels_mapping,
                                     filter_empty=True,
                                     debug=self.debug)

        # train_dataset = Subset(train_dataset, self.train_idxs)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.custom_collate_fn,
        )

        return dataloader

    def val_dataloader(self) -> DataLoader:
        transforms = (self.default_transforms()
                      if self.val_transforms is None else self.val_transforms)

        val_dataset = self.DATASET(self.data_dir,
                                   car=self.val_car,
                                   split='train',
                                   transform=transforms,
                                   mapping=self.labels_mapping,
                                   filter_empty=True,
                                   debug=self.debug)

        # val_dataset = Subset(val_dataset, self.val_idxs)

        dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.custom_collate_fn,
        )

        return dataloader

    def test_dataloader(self) -> DataLoader:
        transforms = (self.default_transforms()
                      if self.test_transforms is None else self.test_transforms)

        test_dataset = self.DATASET(self.data_dir,
                                    car=self.test_car,
                                    split='test',
                                    transform=transforms,
                                    mapping=self.labels_mapping,
                                    filter_empty=True,
                                    debug=self.debug)

        dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.custom_collate_fn,
        )

        return dataloader

    def default_transforms(self):
        transforms = A.Compose([
            ToTensorV2(),
        ])

        return transforms

    def custom_collate_fn(self, batch):
        return collate_fn(batch)
