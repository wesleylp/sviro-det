import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def draw_bounding_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on image
    Args:
        image (torch.Tensor or numpy.array): image to draw bounding boxes on
        boxes (torch.Tensor): bounding boxes
        color (tuple): color of bounding box
        thickness (int): thickness of bounding box
    Returns:
        numpy.array: image with bounding boxes drawn on
    """

    for box in boxes:
        image = draw_bouding_box(image, box, color, thickness)
    return image


def draw_bouding_box(image, bbox, color=(255, 0, 0), thickness=2):
    """
    Draw a bounding box on image
    Args:
        image (torch.Tensor or numpy.array): image to draw bounding boxes on
        box (torch.Tensor): bounding box
        color (tuple): color of bounding box
        thickness (int): thickness of bounding box
    Returns:
        numpy.array: image with bounding box drawn on
    """
    if isinstance(image, torch.Tensor):
        image = image[0].cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image
    else:
        raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(image)}")

    if image.ndim == 2:
        image = np.concatenate([image[..., np.newaxis]] * 3, axis=2)

    box = bbox[1:].cpu().numpy()
    return cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                         color=color,
                         thickness=thickness)
