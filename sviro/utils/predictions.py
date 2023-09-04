import numpy as np


def post_process_batch_detections(detections, prediction_confidence_threshold=0.5):
    predicted_bboxes = []
    predicted_class_confidences = []
    predicted_class_labels = []
    img_ids = []

    iterable_detections = detections.items() if isinstance(detections, dict) else detections

    for img_id, preds in iterable_detections:
        img_ids.append(img_id)
        predicted = postprocess_single_image_detections(preds, prediction_confidence_threshold)
        predicted_bboxes.append(predicted['boxes'])
        predicted_class_confidences.append(predicted['scores'])
        predicted_class_labels.append(predicted['classes'])

    return img_ids, predicted_bboxes, predicted_class_confidences, predicted_class_labels


def postprocess_single_image_detections(detections, prediction_confidence_threshold=0.5):
    boxes = detections['boxes'].detach().cpu().numpy()
    scores = detections['scores'].detach().cpu().numpy()
    classes = detections['labels'].detach().cpu().numpy()

    indexes = np.where(scores > prediction_confidence_threshold)[0]
    boxes = boxes[indexes]

    return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}


def aggregate_preds_outputs(outputs, prediction_confidence_threshold=0.5):
    image_ids = []
    predicted_bboxes = []
    predicted_class_confidences = []
    predicted_class_labels = []

    for batch_pred in outputs['preds']:
        img_ids, pred_bboxes, pred_class_confidences, pred_class_labels = post_process_batch_detections(
            batch_pred, prediction_confidence_threshold=prediction_confidence_threshold)

        image_ids.extend(img_ids)
        predicted_bboxes.extend(pred_bboxes)
        predicted_class_confidences.extend(pred_class_confidences)
        predicted_class_labels.extend(pred_class_labels)

    return (
        image_ids,
        predicted_bboxes,
        predicted_class_confidences,
        predicted_class_labels,
    )
