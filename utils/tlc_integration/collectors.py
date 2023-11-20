# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Metrics collectors - 3LC integration
"""
import torch
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import (YOLOAnnotation,
                                                                                        YOLOBoundingBoxMetricsCollector,
                                                                                        YOLOGroundTruth, YOLOPrediction)
from tlc.client.torch.metrics.metrics_collectors.metrics_collector_base import MetricsCollectorBase

from ..general import non_max_suppression, scale_boxes
from ..tlc_integration.utils import DatasetMode
from .utils import xyxy_to_xywh


class YOLOv5BoundingBoxMetricsCollector(YOLOBoundingBoxMetricsCollector):

    def compute_metrics(self, batch, _1=None, _2=None):
        with torch.no_grad():
            device = next(self.model.parameters()).device  # get sample device
            # half precision only supported on CUDA and only when model is fp16
            half = (device.type != 'cpu' and next(self.model.parameters()).dtype == torch.float16)
            self.model.eval()
            im, targets, paths, shapes = batch
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
            im = im.half() if half else im.float()
            im /= 255
            nb, _, height, width = im.shape  # batch size, channels, height, width

            # Forward
            preds = self.model(im, augment=False)

        return super().compute_metrics(batch, preds)


class NoOpMetricsCollectorWithModel(MetricsCollectorBase):
    """ This metrics collector does nothing, except to block 3LC from performing a forward pass.

    """

    def compute_metrics(self, batch, predictions=None, hook_outputs=None):
        return {}

    @property
    def model(self):
        return torch.nn.Identity()


class Preprocessor:

    def __init__(self, nms_kwargs, dataset_mode=DatasetMode.COLLECT):
        self.nms_kwargs = nms_kwargs
        self.dataset_mode = dataset_mode

    def __call__(self, batch, predictions):
        # Apply NMS
        predictions = non_max_suppression(predictions, **self.nms_kwargs)

        images, targets, paths, shapes = batch
        batch_size = len(paths)

        # Ground truth
        processed_batch = []

        targets = targets.cpu().numpy()

        for i in range(batch_size):
            height, width = shapes[i][0]

            labels = targets[targets[:, 0] == i, 1:]
            xywh_boxes = labels[:, 1:5]
            num_boxes = labels.shape[0]

            # Create annotations with YOLO format
            annotations = [
                YOLOAnnotation(
                    category_id=labels[j, 0],
                    bbox=xywh_boxes[j],
                    score=1.0,
                ) for j in range(num_boxes)]
            ground_truth = YOLOGroundTruth(
                file_name=paths[i],
                height=height,
                width=width,
                annotations=annotations,
            )
            processed_batch.append(ground_truth)

        # Predictions
        processed_predictions = []
        for i, prediction in enumerate(predictions):
            height, width = shapes[i][0]
            scaled_boxes = scale_boxes(
                images[i].shape[1:],
                prediction[:, :4],
                shapes[i][0],
                shapes[i][1] if not self.dataset_mode == DatasetMode.COLLECT else None,
            ).round()
            prediction = prediction.cpu().numpy()
            annotations = [
                YOLOAnnotation(
                    category_id=prediction[j, 5],
                    bbox=xyxy_to_xywh(scaled_boxes[j, :].tolist(), height=height, width=width),
                    score=prediction[j, 4],
                ) for j in range(prediction.shape[0])]
            processed_predictions.append(YOLOPrediction(annotations=annotations))

        return processed_batch, processed_predictions


def tlc_create_metrics_collectors(model,
                                  names,
                                  conf_thres: float = 0.45,
                                  nms_iou_thres: float = 0.45,
                                  max_det: int = 300,
                                  iou_thres: float = 0.4):
    """ Sets up the default metrics collectors for YOLO bounding box metrics collection.

    :param model: The model to use for metrics collection.
    :param conf_thres: Confidence threshold for predictions. Anything under is discarded.
    :param nms_iou_thres: IoU threshold to use for NMS. Boxes with IoU > nms_iou_thres are
        collapsed to the one with greatest confidence.
    :param max_det: Maximum number of detections for a sample.
    :param iou_thres: IoU threshold to use for computing True Positives.

    :returns metrics_collectors: A list of metrics collectors to use.

    """
    nms_kwargs = {
        'conf_thres': conf_thres,
        'iou_thres': nms_iou_thres,
        'classes': None,  # TODO: Add this? Filters to subset of classes.
        'agnostic': False,  # TODO: Add this as a kwarg option? 3LC doesn't really support it?
        'max_det': max_det}

    preprocess_fn = Preprocessor(nms_kwargs)
    metrics_collectors = [
        YOLOv5BoundingBoxMetricsCollector(
            model=model,
            classes=list(names.values()),
            label_mapping={i: i
                           for i in range(len(names))},
            iou_threshold=iou_thres,
            compute_derived_metrics=True,
            preprocess_fn=preprocess_fn,
        ),
        NoOpMetricsCollectorWithModel(metric_names=[]),  # Avoids extra 3LC forward pass
    ]
    return metrics_collectors
