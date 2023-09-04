import csv
import re
from pathlib import Path

import albumentations
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from sviro.mapping_dicts import CLASS2DET_DICT, DET_IDXS_LABELS


class SVIRODetection(Dataset):

    CLASSES_DICT = CLASS2DET_DICT

    def __init__(self,
                 dataroot: str,
                 car: [str | list[str]],
                 split: str,
                 mapping: dict = {
                     'empty': 0,
                     'infant_seat': 1,
                     'child_seat': 2,
                     'person': 3,
                     'everyday_object': 4
                 },
                 filter_empty: bool = False,
                 transform: [None | albumentations.Compose] = None,
                 debug=False):
        """SVIRO dataset

        Args:
            dataroot (str): folder containing the dataset
            car (str  |  list[str]]): "all" or list of car names
            split (str): "train", "test" or "all"
            mapping (dict, optional): mapping classes names into indexes. Defaults to { 'empty': 0, 'infant_seat': 1, 'child_seat': 2, 'person': 3, 'everyday_object': 4 }.
            filter_empty (bool, optional): filter images without bounding boxes. Defaults to False.
            transform (None  |  albumentations.Compose], optional): Image augmentations. Defaults to None.
            debug (bool, optional): Load small portion for debugging. Defaults to False.
        """

        self.dataroot = Path(dataroot)

        self._split = '*' if split == 'all' else f"{split}*"  # train, test or all
        assert car == 'all' or isinstance(car, list)
        self._car = ['*'] if car == 'all' else car

        self.transform = transform
        self.filter_empty = filter_empty
        self.mapping = mapping

        # get all the grayscale images of full size
        all_images = []
        for car in self._car:
            all_images.extend(self._get_list_of_images(car, self._split))

        classes_keep = list(self.mapping.keys())
        # filter images and boxes by labels
        images_to_keep, boxes_files_to_keep = self._filter_images_by_labels(
            all_images, classes_keep, self.filter_empty)

        if debug:
            images_to_keep = images_to_keep[:10]
            boxes_files_to_keep = boxes_files_to_keep[:10]

        self.images = images_to_keep
        self.boxes_files = boxes_files_to_keep

        assert len(self.images) == len(self.boxes_files)

    def _get_list_of_images(self, car, split):
        list_of_images = sorted(
            list(self.dataroot.glob(f"{car}/{split}/grayscale_wholeImage/*.png")))
        return list_of_images

    def _filter_images_by_labels(self, images, labels, filter_empty=False):
        # [car_name]_train_imageID_[imageID]_GT_[GT_label_left]_[GT_label_middle]_[GT_label_right].file_extension
        image_labels_rule = r'GT_([0-9]+)_([0-9]+)_([0-9]+)'

        img_labels_to_keep = []
        for label in labels:
            img_labels_to_keep.extend(self.CLASSES_DICT[label]['cls'])
        img_labels_to_keep = list(set(img_labels_to_keep))

        # filter images and boxes by labels
        images_to_keep = []
        boxes_files_to_keep = []
        for image_filepath in images:
            image_name = image_filepath.stem
            image_labels = re.findall(image_labels_rule, image_name)[0]

            # remove images with no bounding boxes
            if filter_empty and all(f'{label}' == '0' for label in image_labels):
                continue

            if any(f'{label}' in image_labels for label in img_labels_to_keep):
                images_to_keep.append(image_filepath)
                boxes_files_to_keep.append(
                    Path(
                        str(image_filepath).replace('grayscale_wholeImage',
                                                    'boundingBoxes_wholeImage').replace(
                                                        '.png', '.txt')))

        return images_to_keep, boxes_files_to_keep

    def __len__(self):
        return len(self.images)

    def _read_label_file(self, file):
        data = []
        with open(file, mode="r") as f:
            reader = csv.reader(f)
            for row in reader:
                if DET_IDXS_LABELS[int(row[0])] not in self.mapping.keys():
                    continue
                data.append([int(x) for x in row])

        data = np.array(data)
        return data

    def __map_labels(self, labels):
        labels = [DET_IDXS_LABELS[label_idx] for label_idx in labels]
        idxs_mapped = [self.mapping[label] for label in labels]

        return idxs_mapped

    def __getitem__(self, idx):

        # load image and boxes
        image = Image.open(self.images[idx])
        image = np.array(image)

        annotation = self._read_label_file(self.boxes_files[idx])
        boxes = annotation[:, 1:]
        labels_idxs = annotation[:, 0]
        labels_idxs_mapped = self.__map_labels(labels_idxs)

        if self.transform is not None:
            transformed = self.transform(image=image, boxes=boxes)
            image = transformed["image"]
            boxes = transformed["boxes"]

        image = image / 255.0

        target = {
            'image_id': torch.tensor([idx]),
            'labels': torch.as_tensor(labels_idxs_mapped, dtype=torch.int64),
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
        }

        return image, target
