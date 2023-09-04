# SVIRO Object Detection

The code in this repository aims to build an ML solution for object detection, more specifically infant seat localization in the passenger seat of vehicles.

## Dataset
The data set used in this work is available at: https://sviro.kl.dfki.de/.

It can be downloaded using the `get_data.sh` and the auxiliary files `.txt` that contain the image and bounding boxes links.

The data must be placed in a folder named `data`.

## Solution

We used a Faster R-CNN with ResNet-50 and FPN as backbone.
The dataset is artificially increased through Albumentations library.

## Usage

- create an environment using the `environment.yml` file.
- Download the data and unzip it to a folder named `data`.
- run `tools/train.py`
- run `tools/inference.py`
  - optional: download our trained weights

We have tried to make the code as modular as possible. So that, one can modify to `train.py` and `inference.py` to, respectively, train and evaluate for other cars models, use different augmentations strategies and so on.

Checkout the `notebooks`!


## Results

We trained the model for 50 epochs on using only the images from `x5` car and evaluated this model on the test set of `aclass` car. Bellow are the COCO metrics:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.647
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.517
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638
```

## References
- https://github.com/SteveCruz/sviro_helperfunctions
- https://github.com/alexhock/object-detection-metrics
- https://github.com/albumentations-team/albumentations/

## TODO
- [x] Train an object detection model
- [x] Evaluate results
- [ ] Implement using config file

