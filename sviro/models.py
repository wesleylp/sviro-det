import torch
from objdetecteval.metrics.coco_metrics import get_coco_stats
from pytorch_lightning import LightningModule

from sviro.utils.predictions import aggregate_preds_outputs


def collate_fn(batch):
    return tuple(zip(*batch))


class DetectorModel(LightningModule):
    def __init__(self, model=None, prediction_confidence_threshold=0.5):
        super(DetectorModel, self).__init__()
        self.model = model
        self.prediction_confidence_threshold = prediction_confidence_threshold

        # outputs are stored here during training and validation steps
        # https://github.com/Lightning-AI/lightning/pull/16520
        self.training_step_outputs = []
        self.validation_step_outputs = {'loss': [], 'preds': [], 'targets': []}

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        logging_losses = {f'train_{k}': v.detach() for k, v in loss_dict.items()}
        [
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=False)
            for k, v in logging_losses.items()
        ]

        self.training_step_outputs.append(total_loss)

        return {'loss': total_loss}

    def on_train_epoch_end(self):
        # TODO: chech if self.log(on_epoch=True) in <phase>_step is enough
        loss_epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("train_loss_epoch_average", loss_epoch_average)
        self.training_step_outputs.clear()  # free memory

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)

        preds = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        # compute losses
        with torch.no_grad():
            self.model.train()
            loss_dict = self.model(images, targets)
            self.model.eval()

        total_loss = sum(loss for loss in loss_dict.values())
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        logging_losses = {f'val_{k}': v.detach() for k, v in loss_dict.items()}
        [
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=False)
            for k, v in logging_losses.items()
        ]

        self.validation_step_outputs['loss'].append(total_loss)
        self.validation_step_outputs['preds'].append(preds)
        self.validation_step_outputs['targets'].append(targets)

        return {'loss': total_loss}

    def on_validation_epoch_end(self):

        # TODO: chech if self.log(on_epoch=True) in <phase>_step is enough
        loss_epoch_average = torch.stack(self.validation_step_outputs['loss']).mean()
        self.log("val_loss_epoch_average",
                 loss_epoch_average,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        self.validation_step_outputs['loss'].clear()  # free memory

        stats = 0

        image_ids, predicted_bboxes, predicted_class_confidences, predicted_class_labels = aggregate_preds_outputs(
            self.validation_step_outputs, self.prediction_confidence_threshold)

        truth_image_ids = []
        truth_labels = []
        truth_boxes = []
        for batch in self.validation_step_outputs['targets']:
            for target_img_dict in batch:
                truth_image_ids.append(target_img_dict['image_id'].item())
                truth_boxes.append(target_img_dict['boxes'].detach().cpu().numpy())
                truth_labels.append(target_img_dict['labels'].detach().cpu().numpy())

        stats = get_coco_stats(
            prediction_image_ids=image_ids,
            predicted_class_confidences=predicted_class_confidences,
            predicted_bboxes=predicted_bboxes,
            predicted_class_labels=predicted_class_labels,
            target_image_ids=truth_image_ids,
            target_bboxes=truth_boxes,
            target_class_labels=truth_labels,
        )['All']

        [self.log(k, v, on_step=True, on_epoch=True, prog_bar=False) for k, v in stats.items()]

        self.validation_step_outputs['preds'].clear()  # free memory
        self.validation_step_outputs['targets'].clear()  # free memory

        return {"val_loss_epoch_average": loss_epoch_average, 'metrics': stats}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
        )

        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'epoch',
            'monitor': 'val_loss_epoch_average'
        }]
