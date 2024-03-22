# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Collect 3LC metrics on one or more splits of a dataset outside of training.

Usage:
    $ python val.py --task collect

Available environment variables to configure collection:
    TLC_IMAGE_EMBEDDINGS_DIM: Dimension of the image embeddings to collect. If 0, no embeddings are collected.
    TLC_COLLECT_LOSS: Whether to collect loss. Default: false.
    TLC_COLLECTION_SPLITS: Comma-separated list of splits to collect metrics on. Default: train,val.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import tlc

    from utils.loggers.tlc.version import check_tlc_version

    check_tlc_version()
except ImportError:
    raise ImportError("Install 3LC with `pip install tlc` to collect metrics.")

import val as validate

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from typing import Any

from models.common import DetectMultiBackend
from models.yolo import DetectionModel

from utils.callbacks import Callbacks
from utils.general import LOGGER, check_img_size, increment_path, yaml_save
from utils.loggers.tlc.base import BaseTLCCallback
from utils.loggers.tlc.constants import TLC_COLLECT_PATH, TLC_COLORSTR
from utils.loggers.tlc.dataloaders import create_dataloader
from utils.loggers.tlc.settings import Settings
from utils.loggers.tlc.utils import get_names_from_yolo_table, tlc_check_dataset
from utils.loggers.tlc.yolo import TLCDetectionModel
from utils.loss import ComputeLoss
from utils.torch_utils import select_device


def collect_metrics(opt: argparse.Namespace) -> None:
    """
    Run validation and collect metrics for all the splits in the dataset.

    :param opt: Options dictionary - command line arguments from val.py.
    """
    settings = Settings.from_env()
    settings.verify(opt, training=False)

    # Register and/or read the data from 3LC table
    tables = tlc_check_dataset(opt.data, get_splits=settings.collection_splits)

    names = get_names_from_yolo_table(tables[settings.collection_splits[0]])
    data_dict = {"nc": len(names), "names": names}
    label_mapping = {i: i for i in range(len(names))}

    # Get the model
    model, half, batch_size, imgsz = load_model(opt, settings)

    # Sanity checks
    assert all(
        split in tables for split in settings.collection_splits
    ), f"Not all splits {settings.collection_splits} are in the dataset {tables}"
    project_name = tables[settings.collection_splits[0]].project_name

    run = tlc.init(project_name=project_name)

    parameters = vars(opt)
    parameters["weights"] = str(parameters["weights"])
    parameters["project"] = str(parameters["project"])
    run.set_parameters(parameters=parameters)

    for split, table in tables.items():
        dataloader = create_dataloader(
            (TLC_COLLECT_PATH, table, False, settings.exclude_zero_weight_collection),
            opt.imgsz,
            batch_size=batch_size,
            stride=model.stride,
            pad=0.5,
            rect=True,
            workers=opt.workers,
        )[0]

        example_ids = dataloader.dataset.example_ids
        batch_size = dataloader.batch_size

        # Create callback to collect metrics and register it
        callbacks = Callbacks()
        loss_fn = ComputeLoss(model) if settings.collect_loss else None
        tlc_callback = TLCCollectionCallback(
            split,
            opt,
            run,
            table,
            data_dict,
            label_mapping,
            settings,
            example_ids=example_ids,
            loss_fn=loss_fn,
            batch_size=batch_size,
        )
        callbacks.register_action("on_val_batch_end", callback=tlc_callback.on_val_batch_end)
        callbacks.register_action("on_val_end", callback=tlc_callback.on_val_end)

        print(f"{TLC_COLORSTR}Collecting metrics on {split} split")
        project = ROOT / "runs/collect"
        save_dir = increment_path(project / opt.name, exist_ok=opt.exist_ok, mkdir=True)
        yaml_save(save_dir / "opt.yaml", vars(opt))
        validate.run(
            data_dict,
            batch_size=batch_size,
            imgsz=imgsz,
            half=half,
            model=model,
            dataloader=dataloader,
            save_dir=save_dir,
            plots=True,
            callbacks=callbacks,
            compute_loss=None,
        )
        # TODO: Forward most arguments (save to json etc.)?

        tlc_callback.flush_metrics_writer(input_table=table)

    if settings.image_embeddings_dim > 0:
        split_to_reduce_by = "val" if "val" in settings.collection_splits else settings.collection_splits[-1]
        table_url_to_reduce_by = tables[split_to_reduce_by].url
        run.reduce_embeddings_by_foreign_table_url(
            table_url_to_reduce_by, method=settings.image_embeddings_reducer, n_components=settings.image_embeddings_dim
        )


def load_model(opt: argparse.Namespace, settings: Settings) -> tuple[DetectMultiBackend, bool, int, int]:
    """Loads model like in val.py."""
    device = select_device(opt.device, batch_size=opt.batch_size)
    batch_size = opt.batch_size

    # Load model
    model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)

    # Patch model with embeddings writing method
    if isinstance(model.model, DetectionModel) and settings.image_embeddings_dim > 0:
        model.model.collect_embeddings = True
        model.model._forward_once = TLCDetectionModel._forward_once.__get__(model.model, DetectionModel)

    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

    model.eval()
    return model, half, batch_size, imgsz


class TLCCollectionCallback(BaseTLCCallback):
    """Callback for collecting metrics on a single split of the dataset."""

    def __init__(
        self,
        split: str,
        opt: argparse.Namespace,
        run: tlc.Run,
        table: tlc.Table,
        data_dict: dict[str, Any],
        label_mapping: dict[int, int],
        settings: Settings,
        example_ids: list[int] | None = None,
        loss_fn: ComputeLoss | None = None,
        batch_size: int = 1,
    ) -> None:
        self.split = split
        self.opt = opt
        self.run = run
        self.table = table
        self.data_dict = data_dict
        self.label_mapping = label_mapping
        self.example_ids = example_ids
        self._loss_fn = loss_fn
        self._settings = settings

        self.metrics_writer = tlc.MetricsTableWriter(
            run_url=self.run.url,
            foreign_table_url=self.table.url,
            foreign_table_display_name=self.table.dataset_name,
            column_schemas=self.metrics_schema,
        )

        self.example_ids_for_batch = list(tlc.batched_iterator(range(len(self.example_ids)), batch_size=batch_size))

    def on_val_batch_end(
        self,
        batch_i: int,
        images: list[torch.Tensor],
        targets: list[torch.Tensor],
        paths: list[str],
        shapes: list[tuple[int, int]],
        outputs: list[torch.Tensor],
        train_out: list[torch.Tensor] | None = None,
    ):
        self._save_batch_metrics(batch_i, images, targets, paths, shapes, outputs, train_out)
