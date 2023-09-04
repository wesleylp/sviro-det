import albumentations as A
import click
import torch
from albumentations.pytorch import ToTensorV2
from objdetecteval.metrics.coco_metrics import get_coco_stats
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights, faster_rcnn,
                                          fasterrcnn_resnet50_fpn)

from sviro.datamodule import SVIRODataModule
from sviro.datamodule import collate_fn as custom_collate_fn
from sviro.dataset import SVIRODetection
from sviro.models import DetectorModel
from sviro.utils.predictions import (post_process_batch_detections,
                                     postprocess_single_image_detections)


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    arch = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get number of input features for the classifier
    in_features = arch.roi_heads.box_predictor.cls_score.in_features
    arch.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
        in_channels=in_features,
        num_classes=2,
    )

    model = DetectorModel(model=arch, )
    model.load_state_dict(state_dict=checkpoint['state_dict'])

    return model


def model_inference(model, dataloader, prediction_confidence_threshold=0.5, device='cuda'):
    outputs = []
    targets = []
    for batch_images, batch_targets in dataloader:

        images = list(image.to(device) for image in batch_images)

        with torch.no_grad():
            batch_outputs = model(images)

        outputs.append(batch_outputs)
        targets.append(batch_targets)

    truth = {'image_ids': [], 'labels': [], 'boxes': []}
    prediction = {'image_ids': [], 'labels': [], 'boxes': [], 'scores': []}

    for batch_outputs, batch_targets in zip(outputs, targets):
        for output, target in zip(batch_outputs, batch_targets):
            image_id = target['image_id'].item()
            output = postprocess_single_image_detections(output, prediction_confidence_threshold)

            truth['image_ids'].append(image_id)
            truth['labels'].append(target['labels'].detach().cpu().numpy())
            truth['boxes'].append(target['boxes'].detach().cpu().numpy())

            prediction['image_ids'].append(image_id)
            prediction['labels'].append(output['classes'])
            prediction['boxes'].append(output['boxes'])
            prediction['scores'].append(output['scores'])

    stats = get_coco_stats(
        prediction_image_ids=prediction['image_ids'],
        predicted_class_confidences=prediction['scores'],
        predicted_bboxes=prediction['boxes'],
        predicted_class_labels=prediction['labels'],
        target_image_ids=truth['image_ids'],
        target_bboxes=truth['boxes'],
        target_class_labels=truth['labels'],
    )['All']

    return stats, prediction, truth


@click.command()
@click.option('--checkpoint-path', default='outputs/last.ckpt', type=str, required=True)
@click.option('--dataset-path',
              default='/home/wesley.passos/repos/sviro/data',
              type=str,
              required=True)
@click.option('--prediction-confidence-threshold', default=0.5, type=float, required=True)
@click.option('--device', default='cuda', type=str, required=True)
def run_inference(checkpoint_path, dataset_path, prediction_confidence_threshold, device):

    device = torch.device(device)

    transforms = A.Compose([
        ToTensorV2(),
    ])
    dataset = SVIRODetection(
        dataroot=dataset_path,
        car=["aclass"],
        split='test',
        mapping={'infant_seat': 1},
        transform=transforms,
    )

    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=4,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=False,
                                              collate_fn=custom_collate_fn)

    model = load_model(checkpoint_path)
    model.to(device)
    model.eval()

    model_inference(model,
                    test_loader,
                    prediction_confidence_threshold=prediction_confidence_threshold,
                    device=device)


if __name__ == "__main__":
    run_inference()
