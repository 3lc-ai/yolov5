# YOLOv5 🚀 AGPL-3.0 license
"""3LC utils."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import TypeVar

import tlc
import yaml
from tlc.helpers import AnnotationHelper, AnnotationType, SchemaHelper
from tlc_ultralytics import create_tables_from_yaml_file
from tlc_ultralytics.detect.utils import check_det_table
from tlc_ultralytics.utils.dataset import parse_3lc_yaml_file

from utils.general import LOGGER, check_dataset
from utils.loggers.tlc.constants import TLC_COLORSTR, TLC_PREFIX

T = TypeVar("T")


def batched_iterator(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """
    Yield successive batches from an iterable. The last batch may be smaller than batch_size.

    Replaces `tlc.client.utils.batched_iterator`, which was removed in 3lc 3.0.

    :param iterable: The iterable to batch.
    :param batch_size: The number of items per batch.
    """
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def yolo_predicted_bounding_box_schema(categories: dict[int, str]) -> tlc.Schema:
    """
    Create a 3LC bounding box schema for YOLOv5.

    :param categories: Categories for the current dataset.
    :returns: The YOLO bounding box schema for predicted boxes.
    """
    from tlc_ultralytics.detect.utils import yolo_predicted_bounding_box_schema as predicted_bounding_box_schema

    label_value_map = {float(i): tlc.schemas.MapElement(class_name) for i, class_name in categories.items()}

    return predicted_bounding_box_schema(label_value_map)


def yolo_loss_schemas(num_classes: int) -> dict[str, tlc.Schema]:
    """
    Create a 3LC schema for YOLOv5 loss metrics.

    :returns: The YOLO loss schemas.
    """
    schemas = {}
    schemas["box_loss"] = tlc.schemas.Float32Schema(description="Box loss", writable=False)
    schemas["obj_loss"] = tlc.schemas.Float32Schema(description="Object loss", writable=False)
    if num_classes > 1:
        schemas["cls_loss"] = tlc.schemas.Float32Schema(description="Classification loss", writable=False)
    return schemas


def get_metrics_collection_epochs(start: int, epochs: int, interval: int, disable: bool) -> list[int]:
    """
    Compute the epochs to collect metrics for.

    :param start: The starting epoch. If -1, metrics are not collected during training.
    :param epochs: The total number of epochs.
    :param interval: How frequently to collect metrics. 1 means every epoch, 2 means every other epoch, and so on.
    :param disable: Whether metrics collection is disabled.
    """
    if disable:
        return []

    if start >= epochs:
        return []

    # If start is less than zero, we don't collect during training
    if start < 0:
        return []

    if interval <= 0:
        raise ValueError(f"Invalid interval {interval}, must be non-zero")
    else:
        return list(range(start, epochs, interval))


def create_tlc_info_string_before_training(metrics_collection_epochs: list[int], disable: bool) -> str:
    """
    Creates a 3LC info string to print before training.

    :param metrics_collection_epochs: The epochs to collect metrics for.
    :param disable: Whether metrics collection is disabled.
    :returns: The 3LC info string.
    """
    if disable:
        return "Metrics collection disabled for this run."

    if not metrics_collection_epochs:
        tlc_mc_string = "Metrics collection only after completed training for this run."
    else:
        plural_epochs = len(metrics_collection_epochs) > 1
        mc_epochs_str = ",".join(map(str, metrics_collection_epochs))
        tlc_mc_string = f"Collecting metrics for epoch{'s' if plural_epochs else ''} {mc_epochs_str}"
        tlc_mc_string += " and after training for this run."

    return tlc_mc_string


def write_3lc_yaml(data_file: str, tables: dict[str, tlc.Table]) -> None:
    """
    Write a 3LC YAML file for the given tables.

    :param data_file: The path to the original YOLO YAML file.
    :param tables: The 3LC tables.
    """
    new_yaml_url = tlc.Url(data_file.replace(".yaml", "_3lc.yaml"))
    if new_yaml_url.exists():
        LOGGER.info(
            f"{TLC_COLORSTR}3LC YAML file already exists: {str(new_yaml_url)}. To use this file,"
            f" add a 3LC prefix: --data 3LC://{str(new_yaml_url)}."
        )
        return

    # Get relative paths for each table to write to YAML file
    split_paths = {split: str(tables[split].url.apply_aliases()) for split in tables}

    # Add :latest to each
    split_paths_latest = {split: f"{path}:latest" for split, path in split_paths.items()}

    # Create 3LC yaml file
    new_yaml_url.write_text(yaml.dump(split_paths_latest, sort_keys=False))

    LOGGER.info(
        f"{TLC_COLORSTR}Created 3LC YAML file: {str(new_yaml_url)}. To use this file,"
        f" add a 3LC prefix: --data 3LC://{str(new_yaml_url)}."
    )


def tlc_check_dataset(data_file: str, get_splits: tuple | list = ("train", "val")) -> dict[str, tlc.Table]:
    """
    Parse the data file and get or create corresponding 3LC tables. If no 3LC YAML exists, create one.

    :param data_file: The path to the original YOLO YAML file.
    :param get_splits: The splits to get tables for.
    :returns: The 3LC tables.
    :raises: FileNotFoundError if the YAML file does not exist.
    """
    # Regular YAML file
    if not data_file.startswith(TLC_PREFIX):
        if not (data_file_url := tlc.Url(data_file)).exists():
            raise FileNotFoundError(f"Could not find YAML file {data_file_url}")

        try:
            data_dict = check_dataset(data_file)  # Download, etc.
        except AssertionError as e:
            raise AssertionError(
                "YOLOv5 dataset check failed. If you are using a 3LC YAML file, remember the 3LC:// prefix."
            ) from e

        splits = [key for key in ("train", "val", "test") if data_dict.get(key)]
        yolo_yaml_name = Path(data_file).name

        tables = create_tables_from_yaml_file(
            data_file,
            task="detect",
            project_name="yolov5-" + Path(data_file).stem,
            splits=splits,
            description=f"Created with YOLOv5 integration from {yolo_yaml_name}",
        )

        # Write all tables to the 3LC YAML file
        write_3lc_yaml(data_file, tables)

        # Always use the latest table for YOLO YAML based tables
        for split, table in tables.items():
            latest_table = table.latest()
            if latest_table != table:
                LOGGER.info(f"{TLC_COLORSTR}Using latest {split} table from YAML file {data_file}: {latest_table.url}")
            tables[split] = latest_table

        # Remove any tables that are not in get_splits
        tables = {split: table for split, table in tables.items() if split in get_splits}

    # 3LC YAML file
    else:
        tables = parse_3lc_yaml_file(data_file)
        tables = {split: table for split, table in tables.items() if split in get_splits}

        for split, table in tables.items():
            try:
                check_det_table(table)
            except ValueError as e:
                raise ValueError(f"Table {table.url} is not compatible with YOLOv5") from e

            table.ensure_fully_defined()
            LOGGER.info(f"{TLC_COLORSTR}Using {split} revision {table.url}")

    # Check that the tables have the same bounding box value maps
    value_maps = [get_names_from_yolo_table(table) for table in tables.values()]
    assert all(value_maps[0] == value_maps[i] for i in range(1, len(value_maps)))

    return tables


def get_names_from_yolo_table(table: tlc.Table) -> dict[int, str]:
    """
    Get the category names from a YOLO table.

    :param table: The YOLO table.
    :returns: The category names for YOLO.
    """
    annotation = AnnotationHelper.find(table, type=AnnotationType.BOUNDING_BOXES)
    if annotation is None or annotation.label_path is None:
        raise ValueError(f"No bounding box label column found in Table {table.url}.")

    value_map = table.get_value_map(annotation.label_path)
    if value_map is None:
        raise ValueError(f"Failed to get value map for Table {table.url}.")

    return SchemaHelper.to_simple_value_map(value_map)


def verify_model_table_compatible(model, table: tlc.Table) -> None:
    table_names = get_names_from_yolo_table(table)

    # Check that the model and table have the same number of classes
    assert len(model.names) == len(table_names), (
        "The selected model was trained on a different number of classes than the table. Please select a model with the same number of classes as the table."
    )

    # Check that the model and table have the same exact classes
    assert model.names == table_names, (
        "The selected model was trained on different classes than the table. Please select a model with the same classes as the table."
    )
