import argparse
import os
import sys
from pprint import pprint

import albumentations as A
import click
import torchvision.models.detection as detection_arch
from albumentations.pytorch import ToTensorV2
from mlflow import log_artifact, log_metric, log_params
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights, faster_rcnn,
                                          fasterrcnn_resnet50_fpn)

from sviro.datamodule import SVIRODataModule
from sviro.models import DetectorModel

#TODO: implement with config file (hydra)


@click.command()
@click.option('--dataset_path',
              default='/home/wesley.passos/repos/sviro/data',
              type=str,
              help='Path to dataset')
@click.option('--max_epochs', default=50, type=int, help='Number max of epochs')
def main(dataset_path, max_epochs):

    seed_everything(42)

    out_dirpath = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(out_dirpath, exist_ok=True)

    datamodule = SVIRODataModule(
        data_dir=dataset_path,
        batch_size=4,
        num_workers=4,
        train_car=["x5"],
        val_car=["zoe"],
        test_car=["aclass"],
        labels_mapping={'infant_seat': 1},
    )

    datamodule.setup()

    # TODO: implement with config file
    # maybe as the number of hparams is increasing
    # it is better to use a config file hydra
    datamodule.train_transforms = A.Compose([
        A.Blur(p=0.5),
        A.MedianBlur(p=0.5),
        A.CLAHE(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HorizontalFlip(),
        ToTensorV2(),
    ])

    arch = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get number of input features for the classifier
    in_features = arch.roi_heads.box_predictor.cls_score.in_features
    arch.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
        in_channels=in_features,
        num_classes=2,
    )

    model = DetectorModel(model=arch, )

    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch_average',
                                          dirpath=out_dirpath,
                                          filename='file',
                                          save_last=True)

    model = model.cuda()

    trainer = Trainer(
        num_sanity_val_steps=1,
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        precision=16,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model,
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader())


if __name__ == '__main__':
    main()
