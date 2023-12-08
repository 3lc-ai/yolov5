# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Metrics collectors - 3LC integration
"""
import torch
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import (YOLOAnnotation,
                                                                                        YOLOBoundingBoxMetricsCollector,
                                                                                        YOLOGroundTruth, YOLOPrediction)
from tlc.client.torch.metrics.metrics_collectors.metrics_collector_base import MetricsCollectorBase
from tlc.core.builtins.constants.number_roles import NUMBER_ROLE_NN_EMBEDDING
from tlc.core.schema import DimensionNumericValue, Float32Value, Schema

from ..general import non_max_suppression, scale_boxes, xywh2xyxy, xyxy2xywhn
from .utils import xyxy_to_xywh

ACTIVATIONS = []


class YOLOv5BoundingBoxMetricsCollector(YOLOBoundingBoxMetricsCollector):
    """A YOLOv5 specific bounding box metrics collector."""

    def __init__(self, *args, collect_embeddings=False, compute_loss=None, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Support other layers
        # Collecting embeddings for the last backbone layer.
        # This is the output of the SPPF layer with mean pooling the spatial dimensions, resulting
        # in a 512-dimensional vector.
        self._collect_embeddings = collect_embeddings
        self._compute_loss = compute_loss
        self._activation_size = 512  # 512 for most models. TODO: Infer this - for special YOLOv5 models with different size.

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
            preds, train_out = self.model(im, augment=False)

        # Read the activations written during the forward pass for the batch
        # We remove it once it has been read
        if self._collect_embeddings:
            assert len(ACTIVATIONS) == 1
            activations = ACTIVATIONS.pop()

        assert len(ACTIVATIONS) == 0

        metrics = super().compute_metrics(batch, preds)

        # Compute and add loss values to metrics
        # TODO: Batched computation
        if self._compute_loss is not None:
            metrics.update({'loss': [], 'box_loss': [], 'obj_loss': [], 'cls_loss': []})
            train_out = [to.cpu() for to in train_out]
            targets = targets.cpu()
            # Pretend batch size is 1 and compute loss for each sample
            for i in range(nb):
                train_out_sample = [to[i:i + 1, ...]
                                    for to in train_out]  # Get the train_out for the sample, keep batch dim
                targets_sample = targets[targets[:, 0] == i, :]  # Get the targets for the sample
                targets_sample[:, 0] = 0  # Set the batch index to 0
                losses = self._compute_loss(train_out_sample, targets_sample)[1].numpy()
                metrics['loss'].append(losses.sum())
                metrics['box_loss'].append(losses[0])
                metrics['obj_loss'].append(losses[1])
                metrics['cls_loss'].append(losses[2])

        # Add embeddings to metrics
        if self._collect_embeddings and activations is not None:
            metrics['embeddings'] = activations.cpu().numpy()

        return metrics

    @property
    def column_schemas(self):
        _column_schemas = super().column_schemas

        # Loss schemas
        if self._compute_loss is not None:
            _column_schemas['loss'] = Schema(description='Sample loss',
                                             writable=False,
                                             value=Float32Value(),
                                             display_importance=3003)
            _column_schemas['box_loss'] = Schema(description='Box loss',
                                                 writable=False,
                                                 value=Float32Value(),
                                                 display_importance=3004)
            _column_schemas['obj_loss'] = Schema(description='Object loss',
                                                 writable=False,
                                                 value=Float32Value(),
                                                 display_importance=3005)
            _column_schemas['cls_loss'] = Schema(description='Classification loss',
                                                 writable=False,
                                                 value=Float32Value(),
                                                 display_importance=3006)

        # Embedding schema
        if self._collect_embeddings:
            embedding_schema = Schema('Embedding',
                                      'Large NN embedding',
                                      writable=False,
                                      computable=False,
                                      value=Float32Value(number_role=NUMBER_ROLE_NN_EMBEDDING))
            # 512 for all YOLO detection models
            embedding_schema.size0 = DimensionNumericValue(value_min=self._activation_size,
                                                           value_max=self._activation_size,
                                                           enforce_min=True,
                                                           enforce_max=True)
            _column_schemas['embeddings'] = embedding_schema

        return _column_schemas


class NoOpMetricsCollectorWithModel(MetricsCollectorBase):
    """ This metrics collector does nothing, except to block 3LC from performing a forward pass.

    """

    def compute_metrics(self, batch, predictions=None, hook_outputs=None):
        return {}

    @property
    def model(self):
        return torch.nn.Identity()


class Preprocessor:

    def __init__(self, nms_kwargs):
        self.nms_kwargs = nms_kwargs

    def __call__(self, batch, predictions):
        # Apply NMS
        predictions = non_max_suppression(predictions, **self.nms_kwargs)

        images, targets, paths, shapes = batch
        batch_size = len(paths)

        # Ground truth
        processed_batch = []

        nb, _, height, width = images.shape  # batch size, channels, height, width
        targets = targets.cpu()
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=targets.device)

        for i in range(batch_size):
            height, width = shapes[i][0]

            labels = targets[targets[:, 0] == i, 1:]
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_boxes(images[i].shape[1:], tbox, shapes[i][0], shapes[i][1])
            # This is xyxy scaled boxes. Now go back to xywh-normalized and write these
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # normalized labels
            xywh_boxes = xyxy2xywhn(labelsn[:, 1:5], w=width, h=height)
            num_boxes = labelsn.shape[0]

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
                shapes[i][1],
            )
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
                                  iou_thres: float = 0.4,
                                  compute_embeddings: bool = False,
                                  compute_loss=None):
    """ Sets up the default metrics collectors for YOLO bounding box metrics collection.

    :param model: The model to use for metrics collection.
    :param conf_thres: Confidence threshold for predictions. Anything under is discarded.
    :param nms_iou_thres: IoU threshold to use for NMS. Boxes with IoU > nms_iou_thres are
        collapsed to the one with greatest confidence.
    :param max_det: Maximum number of detections for a sample.
    :param iou_thres: IoU threshold to use for computing True Positives.
    :param compute_embeddings: Whether to compute embeddings for each sample.
    :param compute_loss: Function to compute loss for each sample.

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
            derived_metrics_mode='strict',
            preprocess_fn=preprocess_fn,
            collect_embeddings=compute_embeddings,
            compute_loss=compute_loss,
        ),
        NoOpMetricsCollectorWithModel(metric_names=[]),  # Avoids extra 3LC forward pass
    ]
    return metrics_collectors
