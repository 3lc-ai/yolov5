# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Base class for 3LC loggers used for training and validation."""
from __future__ import annotations

import numpy as np
import tlc
import torch

from utils.general import scale_boxes, xywh2xyxy, xyxy2xywhn
from utils.loggers.tlc import yolo
from utils.loggers.tlc.constants import TRAINING_PHASE
from utils.loggers.tlc.utils import (
    construct_bbox_struct,
    get_names_from_yolo_table,
    training_phase_schema,
    yolo_image_embeddings_schema,
    yolo_loss_schemas,
    yolo_predicted_bounding_box_schema,
)
from utils.metrics import box_iou


class BaseTLCCallback:
    """
    Base class for 3LC Callbacks.

    The 3LC integration has two loggers, one for train.py and one for val.py. Shared functionality is implemented here.
    """

    @property
    def metrics_schema(self) -> dict[str, tlc.Schema]:
        """
        Returns the extra metrics schema for the logger, adding fields based on the 3LC settings.

        :return: Metrics schema
        """
        schema = {
            tlc.PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(self.data_dict["names"]),
            TRAINING_PHASE: training_phase_schema(),
        }

        if self._settings.collect_loss:
            schema.update(yolo_loss_schemas(num_classes=self.data_dict["nc"]))

        if self._settings.image_embeddings_dim != 0:
            schema.update(yolo_image_embeddings_schema())

        return schema

    def should_collect_metrics(self) -> bool:
        """
        Returns whether the current logger should collect metrics given the current state.

        :return: Whether the current logger should collect metrics
        """
        return True  # Always collect metrics by default

    def _save_batch_metrics(
        self,
        batch_i: int,
        images: list[torch.Tensor],
        targets: list[torch.Tensor],
        paths: list[str],
        shapes: list[tuple[int, int]],
        outputs: list[torch.Tensor],
        train_out: list[torch.Tensor],
        epoch: int | None = None,
    ) -> None:
        """
        Saves the metrics for the current batch to the current metrics writer.

        :param batch_i: Batch index
        :param images: Images for the batch
        :param targets: Targets for the batch
        :param paths: Paths for the batch
        :param shapes: Shapes for the batch
        :param outputs: Outputs for the batch
        :param train_out: Train out, used for computing loss.
        :param epoch: Current epoch, if available. None if not available.
        """

        # Knowing the batch_i, we know the example ids
        batch_size = images.shape[0]
        example_indices = list(range(batch_i * batch_size, (batch_i + 1) * batch_size))
        # Map indices to example ids (rect reorder is already handled for these upon dataloader creation)
        example_ids = [self.example_ids[i] for i in example_indices]

        predictions = []
        for si, pred in enumerate(outputs):
            if len(pred) == 0:
                predictions.append([])
                continue

            image = images[si]
            labels = targets[targets[:, 0] == si, 1:]
            shape = shapes[si]
            predn, labelsn = self.preprocess_prediction(image, labels, shape, pred)

            # Process the predictions
            detections = predn[predn[:, 4] > self._settings.conf_thres]

            detections = detections[
                detections[:, 4].argsort(descending=True)[: self._settings.max_det]
            ]  # sort by confidence and remove excess boxes

            if labelsn is not None:
                # Compute IoU
                iou = box_iou(labelsn[:, 1:], detections[:, :4])

                pred_classes = detections[:, 5]
                target_classes = labelsn[:, 0]

                mask = target_classes[:, None] == pred_classes
                iou = iou * mask

                ious = iou.cpu().numpy().max(axis=0)
            else:
                ious = [0.0] * len(detections)

            # Convert back to xywhn for saving 3LC box predictions
            pred_xywhn = detections.clone()
            pred_xywhn[:, :4] = xyxy2xywhn(pred_xywhn[:, :4], w=shape[0][1], h=shape[0][0])

            pred_xywhn = pred_xywhn.cpu().tolist()

            # Save the boxes as dicts
            annotations = [
                dict(
                    category_id=int(prediction[5]), score=float(prediction[4]), bbox=prediction[:4], iou=float(ious[i])
                )
                for i, prediction in enumerate(pred_xywhn)
            ]
            predictions.append(annotations)

        # Convert the predictions to 3LC format
        predicted_bounding_boxes = [
            construct_bbox_struct(
                annotations,
                image_width=shapes[i][0][1],
                image_height=shapes[i][0][0],
                inverse_label_mapping=self.label_mapping,
            )
            for i, annotations in enumerate(predictions)
        ]

        # Always collect example ids and predicted bounding boxes
        metrics_batch = {
            tlc.EXAMPLE_ID: example_ids,
            tlc.PREDICTED_BOUNDING_BOXES: predicted_bounding_boxes,
        }

        # Collect epoch only when training
        if epoch is not None:
            metrics_batch[tlc.EPOCH] = [epoch] * batch_size
            metrics_batch[TRAINING_PHASE] = [1 if self._reached_final_validation else 0] * batch_size

        # Embeddings
        if self._settings.image_embeddings_dim != 0:
            assert len(
                yolo.global_activations
            ), f"Should only have one set of embeddings per batch, was {len(yolo.global_activations)}"
            metrics_batch["embeddings"] = yolo.global_activations.pop()

        # Loss
        if self._settings.collect_loss:
            _, _, width, height = images.shape
            targets[:, 2:] /= torch.tensor(
                (width, height, width, height), device=targets.device
            )  # xyxy back to xywhn (undo val.py inplace op)
            loss_metrics = self.compute_loss(train_out, targets)
            metrics_batch.update(loss_metrics)

        self.metrics_writer.add_batch(metrics_batch=metrics_batch)

    def flush_metrics_writer(self, input_table: tlc.Table) -> None:
        """
        Flushes the current metrics in the metrics writer.

        Called at the end of a pass over one of the splits.
        """
        self.run.add_input_table(input_table)

        # Flush the metrics writer and update the Run with the metrics-infos.
        self.metrics_writer.finalize()
        metrics_infos = self.metrics_writer.get_written_metrics_infos()

        self.run.update_metrics(metrics_infos)

    def compute_loss(self, train_out: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute loss for the batch.

        :param train_out: Train out
        :param targets: Targets
        :return: Loss metrics
        """
        lbox, lobj, lcls = self._unreduced_loss_fn(train_out, targets)

        metrics = {
            "box_loss": lbox,
            "obj_loss": lobj,
        }

        if lcls is not None:
            metrics["cls_loss"] = lcls

        return metrics

    def preprocess_prediction(
        self, image: torch.Tensor, labels: torch.Tensor, shape: list[list[int], list[list[int]]], pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocesses the prediction for saving, mapping them back to native space.

        :param image: Images
        :param labels: Labels
        :param shape: Shape
        :param pred: Predictions
        :return: Preprocessed predictions and labels
        """
        nl, _ = labels.shape[0], pred.shape[0]

        # Predictions
        if self.opt.single_cls:
            pred[:, 5] = 0

        predn = pred.clone()
        scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])  # native-space pred

        labelsn = None
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_boxes(image.shape[1:], tbox, shape[0], shape[1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels

        return predn, labelsn

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, _=None) -> None:
        if self.run is not None:
            self.write_per_class_metrics_table(nt, tp, fp, p, r, f1, ap, ap50, ap_class)

    def write_per_class_metrics_table(
        self,
        nt: np.ndarray,
        tp: np.ndarray,
        fp: np.ndarray,
        p: np.ndarray,
        r: np.ndarray,
        f1: np.ndarray,
        ap: np.ndarray,
        ap50: np.ndarray,
        ap_class: np.ndarray,
    ) -> None:
        """
        Writes the per-class metrics table for the current split.

        :param nt: Number of targets per class
        :param tp: True positives per class
        :param fp: False positives per class
        :param p: Precision per class
        :param r: Recall per class
        :param f1: F1 per class
        :param ap: Average precision per class
        :param ap50: Average precision @ 50 per class
        :param ap_class: Classes for which the metrics are computed (without nt=0 classes)
        :param confusion_matrix: Confusion matrix
        """
        names = get_names_from_yolo_table(self.table)
        column_schemas = tlc.SampleType.from_structure(
            {
                "Split": tlc.CategoricalLabel(name="Split", classes=["train", "val"]),
                "Class": tlc.CategoricalLabel(name="Class", classes=list(names.values())),
                "Targets": tlc.Int,
                "TPs": tlc.Int,
                "FPs": tlc.Int,
                "Precision": tlc.Float,
                "Recall": tlc.Float,
                "F1": tlc.Float,
                "AP": tlc.Float,
                "AP50": tlc.Float,
            }
        ).schema.values

        metrics = {
            "Split": [int(self.split == "train")] * len(ap_class),
            "Class": ap_class,
            "Targets": [int(t) for t in nt if t > 0],
            "TPs": tp,
            "FPs": fp,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "AP": ap,
            "AP50": ap50,
        }

        self.run.add_metrics_data(
            metrics=metrics,
            override_column_schemas=column_schemas,
            table_writer_base_name=f"{self.split}_per_class_metrics",
        )
