from mmdet.apis import DetInferencer
import numpy as np
import torch
import sys
import functools
import os
import mmcv
import warnings

# Suppress UserWarnings
warnings.filterwarnings("ignore", message="Failed to add <class 'mmengine.visualization.vis_backend.LocalVisBackend'>.*")


def disable_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Redirect stdout to suppress print output
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            result = func(*args, **kwargs)
        # Restore stdout
        sys.stdout = sys.__stdout__
        return result
    return wrapper


@disable_print
def getInferencer(model, weights):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DetInferencer(model=model, weights=weights, device=device)


@disable_print
class Detector:
    def __init__(self, model, weights):
        self.inferencer = getInferencer(model, weights)

    @disable_print
    def detect_objects(self, img, scoreThreshold):

        # Perform inference
        data = self.inferencer([img], show=False, return_vis=False, return_datasamples=True)

        # Access the predicted instances
        data = data['predictions'][0].pred_instances

        # Access the bounding boxes, labels, and scores
        bboxes = data.bboxes.detach().cpu().numpy()
        labels = data.labels.detach().cpu().numpy()
        scores = data.scores.detach().cpu().numpy()
        masks = data.masks.detach().cpu().numpy()

        detections = []

        for i in range(len(scores)):
            if scores[i] > scoreThreshold:
                detection = {
                    'bbox': list(map(int, bboxes[i])), #convert to int
                    'score': scores[i],
                    'mask': masks[i]
                }
                detections.append(detection)

        return detections
