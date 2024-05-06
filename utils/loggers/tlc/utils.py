# YOLOv5 🚀 AGPL-3.0 license
"""3LC utils."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

import tlc
import yaml
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import (
    _TLCPredictedBoundingBox,
    _TLCPredictedBoundingBoxes,
)

from utils.general import LOGGER, check_dataset
from utils.loggers.tlc.constants import TLC_COLORSTR, TLC_PREFIX, TRAINING_PHASE

T = TypeVar("T", bound=Callable)  # Generic type for environment variable parsing functions


def yolo_predicted_bounding_box_schema(categories: dict[int, str]) -> tlc.Schema:
    """
    Create a 3LC bounding box schema for YOLOv5.

    :param categories: Categories for the current dataset.
    :returns: The YOLO bounding box schema for predicted boxes.
    """
    label_value_map = {float(i): tlc.MapElement(class_name) for i, class_name in categories.items()}

    bounding_box_schema = tlc.BoundingBoxListSchema(
        label_value_map=label_value_map,
        x0_number_role=tlc.NUMBER_ROLE_BB_CENTER_X,
        x1_number_role=tlc.NUMBER_ROLE_BB_SIZE_X,
        y0_number_role=tlc.NUMBER_ROLE_BB_CENTER_Y,
        y1_number_role=tlc.NUMBER_ROLE_BB_SIZE_Y,
        x0_unit=tlc.UNIT_RELATIVE,
        y0_unit=tlc.UNIT_RELATIVE,
        x1_unit=tlc.UNIT_RELATIVE,
        y1_unit=tlc.UNIT_RELATIVE,
        description="Predicted Bounding Boxes",
        writable=False,
        is_prediction=True,
        include_segmentation=False,
    )

    return bounding_box_schema


def training_phase_schema() -> tlc.Schema:
    return tlc.Schema(
        display_name=TRAINING_PHASE,
        description=(
            "'During' metrics are collected with EMA during training, "
            "'After' is with the final model weights after completed training."
        ),
        display_importance=tlc.DISPLAY_IMPORTANCE_EPOCH - 1,  # Right hand side of epoch in the Dashboard
        writable=False,
        computable=False,
        value=tlc.Int32Value(
            value_min=0,
            value_max=1,
            value_map={
                float(0): tlc.MapElement(display_name="During"),
                float(1): tlc.MapElement(display_name="After"),
            },
        ),
    )


def yolo_loss_schemas(num_classes: int) -> dict[str, tlc.Schema]:
    """
    Create a 3LC schema for YOLOv5 loss metrics.

    :returns: The YOLO loss schemas.
    """
    schemas = {}
    schemas["box_loss"] = tlc.Schema(
        description="Box loss", writable=False, value=tlc.Float32Value(), display_importance=3004
    )
    schemas["obj_loss"] = tlc.Schema(
        description="Object loss", writable=False, value=tlc.Float32Value(), display_importance=3005
    )
    if num_classes > 1:
        schemas["cls_loss"] = tlc.Schema(
            description="Classification loss", writable=False, value=tlc.Float32Value(), display_importance=3006
        )
    return schemas


def yolo_image_embeddings_schema(activation_size=512) -> dict[str, tlc.Schema]:
    """
    Create a 3LC schema for YOLOv5 image embeddings.

    :param activation_size: The size of the activation tensor.
    :returns: The YOLO image embeddings schema.
    """
    embedding_schema = tlc.Schema(
        "Embedding",
        "Large NN embedding",
        writable=False,
        computable=False,
        value=tlc.Float32Value(number_role=tlc.NUMBER_ROLE_NN_EMBEDDING),
        size0=tlc.DimensionNumericValue(
            value_min=activation_size, value_max=activation_size, enforce_min=True, enforce_max=True
        ),
    )
    return {"embeddings": embedding_schema}


def construct_bbox_struct(
    predicted_annotations: list[dict[str, int | float | dict[str, float]]],
    image_width: int,
    image_height: int,
    inverse_label_mapping: dict[int, int] | None = None,
) -> _TLCPredictedBoundingBoxes:
    """
    Construct a 3LC bounding box struct from a list of bounding boxes.

    :param predicted_annotations: A list of predicted bounding boxes.
    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :param inverse_label_mapping: A mapping from predicted label to category id.
    """

    bbox_struct = _TLCPredictedBoundingBoxes(
        bb_list=[],
        image_width=image_width,
        image_height=image_height,
    )

    for pred in predicted_annotations:
        bbox, label, score, iou = pred["bbox"], pred["category_id"], pred["score"], pred["iou"]
        label_val = inverse_label_mapping[label] if inverse_label_mapping is not None else label
        bbox_struct["bb_list"].append(
            _TLCPredictedBoundingBox(
                label=label_val,
                confidence=score,
                iou=iou,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
            )
        )

    return bbox_struct


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
        tlc_mc_string = f'Collecting metrics for epoch{"s" if plural_epochs else ""} {mc_epochs_str}'
        tlc_mc_string += " and after training for this run."

    return tlc_mc_string


def get_or_create_tlc_table_from_yolo(yolo_yaml_file: tlc.Url | str, split: str) -> tlc.Table:
    """
    Get or create a 3LC table from a YOLO YAML file.

    :param yolo_yaml_file: The path to the YOLO YAML file.
    :param split: The split to get the table for.
    :returns: The 3LC table.
    """
    tlc.TableIndexingTable.instance().ensure_fully_defined()

    # Resolving logic for YOLO YAML file
    dataset_name_base = Path(yolo_yaml_file).stem
    dataset_name = dataset_name_base + "-" + split
    project_name = "yolov5-" + dataset_name_base

    yolo_yaml_name = Path(yolo_yaml_file).name
    yolo_yaml_file = str(Path(yolo_yaml_file).resolve())  # Ensure absolute path for resolving Table Url

    try:
        table = tlc.Table.from_yolo(
            dataset_yaml_file=yolo_yaml_file,
            split=split,
            structure=None,
            table_name=split,
            dataset_name=dataset_name,
            project_name=project_name,
            if_exists="raise",
            add_weight_column=True,
            description=f"Created with YOLOv5 integration from {yolo_yaml_name}",
        )
        table.write_to_row_cache(create_url_if_empty=True, overwrite_if_exists=False)  # Always cache for YOLO tables
        LOGGER.info(f"{TLC_COLORSTR}Created {split} table {table.url} from YAML file {yolo_yaml_file}")

    except FileExistsError:
        # Table already exists, reuse it instead and log it
        table = tlc.Table.from_yolo(
            dataset_yaml_file=yolo_yaml_file,
            split=split,
            structure=None,
            table_name=split,
            dataset_name=dataset_name,
            project_name=project_name,
            if_exists="reuse",
            add_weight_column=True,
            description=f"Created with YOLOv5 integration from {yolo_yaml_name}",
        )
        latest_table = table.latest()

        if latest_table == table:
            LOGGER.info(f"{TLC_COLORSTR}Using existing {split} table for YAML file {yolo_yaml_file}: {table.url}")
        else:
            LOGGER.info(f"{TLC_COLORSTR}Using latest {split} table from YAML file {yolo_yaml_file}: {latest_table.url}")

        # Always use the latest table for YOLO YAML based tables
        table = latest_table

    table.ensure_fully_defined()

    return table


def get_tlc_table_from_url(table_url: tlc.Url, split: str) -> tlc.Table:
    """
    Get a 3LC table from a URL.

    :param table_url: The Url of the table.
    :param split: The split the table corresponds to.
    :returns: The 3LC table.
    :raises: ValueError if the table does not exist.
    :raises: ValueError if the table is not compatible with YOLOv5.
    """

    try:
        table = tlc.Table.from_url(table_url)
        LOGGER.info(f"{TLC_COLORSTR}Using {split} revision {table_url}")
    except FileNotFoundError:
        raise ValueError(f"Could not find Table {table_url} for {split} split")

    try:
        check_table_compatibility(table)
    except AssertionError as e:
        raise ValueError(f"Table {table_url} is not compatible with YOLOv5") from e

    table.ensure_fully_defined()
    return table


def check_table_compatibility(table: tlc.Table) -> bool:
    """
    Check that the 3LC Table is compatible with YOLOv5.

    :param table: The 3LC Table to check.
    :returns: True if the Table is compatible, False otherwise.
    """

    row_schema = table.row_schema.values
    assert tlc.IMAGE in row_schema
    assert tlc.WIDTH in row_schema
    assert tlc.HEIGHT in row_schema
    assert tlc.BOUNDING_BOXES in row_schema
    assert tlc.BOUNDING_BOX_LIST in row_schema[tlc.BOUNDING_BOXES].values
    assert tlc.SAMPLE_WEIGHT in row_schema
    assert tlc.LABEL in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
    assert tlc.X0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
    assert tlc.Y0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
    assert tlc.X1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
    assert tlc.Y1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values

    X0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X0]
    Y0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y0]
    X1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X1]
    Y1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y1]

    assert X0.value.number_role == tlc.NUMBER_ROLE_BB_CENTER_X
    assert Y0.value.number_role == tlc.NUMBER_ROLE_BB_CENTER_Y
    assert X1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_X
    assert Y1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_Y

    return True


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

    # Common path for train, val, test tables:
    #                                        v           <--          <--          *
    # projects / yolov5-<dataset_name> / datasets / <dataset_name> / tables / <table_url> / files
    path = tables["train"].url.parent.parent.parent

    # Get relative paths for each table to write to YAML file
    split_paths = {split: str(tlc.Url.relative_from(tables[split].url, path).apply_aliases()) for split in tables}

    # Add :latest to each
    split_paths_latest = {split: f"{path}:latest" for split, path in split_paths.items()}

    # Create 3LC yaml file
    data_config = {"path": str(path), **split_paths_latest}
    new_yaml_url.write(yaml.dump(data_config, sort_keys=False, encoding="utf-8"))

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
            check_dataset(data_file)  # Download, etc.
        except AssertionError as e:
            raise AssertionError(
                "YOLOv5 dataset check failed. If you are using a 3LC YAML file, remember the 3LC:// prefix."
            ) from e

        data_file_content = yaml.safe_load(data_file_url.read())
        splits = [
            key for key in data_file_content if key not in ("path", "names", "download") and data_file_content[key]
        ]

        # Create 3LC tables, get root table if already registered
        tables = {split: get_or_create_tlc_table_from_yolo(data_file, split=split) for split in splits}

        # Write all tables to the 3LC YAML file
        write_3lc_yaml(data_file, tables)

        # Remove any tables that are not in get_splits
        tables = {split: table for split, table in tables.items() if split in get_splits}

    # 3LC YAML file
    else:
        # Read the YAML file, removing the prefix
        if not (data_file_url := tlc.Url(data_file.replace(TLC_PREFIX, ""))).exists():
            raise FileNotFoundError(f"Could not find YAML file {data_file_url}")

        data_config = yaml.safe_load(data_file_url.read())

        path = data_config.get("path")
        splits = [key for key in data_config if key != "path"]

        tables = {}
        for split in splits:
            if split not in get_splits:
                continue

            split_path = data_config[split].split(":")[0]
            latest = data_config[split].endswith(":latest")

            if split_path.count(":") > 1:
                raise ValueError(f"Found more than one : in the split path {split_path} for split {split}")
            url = tlc.Url(path) / split_path if path else tlc.Url(split_path)

            table = get_tlc_table_from_url(table_url=url, split=split)

            # Use latest revision if :latest is specified
            if latest:
                prev_url = url
                table = table.latest()
                if prev_url != table.url:
                    LOGGER.info(f"{TLC_COLORSTR}Using latest revision for {split} set: {table.url}.")

            tables[split] = table

    # Check that the tables have the same bounding box value maps
    value_maps = [get_names_from_yolo_table(table) for table in tables.values()]
    assert all(value_maps[0] == value_maps[i] for i in range(1, len(value_maps)))

    return tables


def get_names_from_yolo_table(table: tlc.Table, value_path: str = "bbs.bb_list.label") -> dict[int, str]:
    """
    Get the category names from a YOLO table.

    :param table: The YOLO table.
    :returns: The category names for YOLO.
    """
    value_map = table.get_value_map(value_path)
    return {int(k): v["internal_name"] for k, v in value_map.items()}
