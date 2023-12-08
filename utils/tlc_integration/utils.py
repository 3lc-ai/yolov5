# YOLOv5 🚀 AGPL-3.0 license
"""
3LC Utils
"""

from pathlib import Path

import numpy as np
import tlc
from tlc.core.objects.tables.from_url.utils import get_cache_file_name, resolve_table_url

from utils.general import LOGGER, colorstr


def get_or_create_tlc_table(yolo_yaml_file, split, revision_url='', root_url=None):
    """Get or create a 3LC Table for the given inputs"""

    if not yolo_yaml_file and not revision_url:
        raise ValueError('Either yolo_yaml_file or revision_url must be specified')

    if not split and not revision_url:
        raise ValueError('split must be specified if revision_url is not specified')

    tlc_prefix = colorstr('3LC: ')

    # Ensure complete index before resolving any Tables
    tlc.TableIndexingTable.instance().ensure_fully_defined()

    # Infer dataset and project names
    dataset_name_base = Path(yolo_yaml_file).stem
    dataset_name = dataset_name_base + '-' + split
    project_name = 'yolov5-' + dataset_name_base

    if yolo_yaml_file:  # review this
        yolo_yaml_file = str(Path(yolo_yaml_file).resolve())  # Ensure absolute path for resolving Table Url

    # Resolve a unique Table name using dataset_name, yaml file path, yaml file size (and optionally root_url path and size), and split to create a deterministic url
    # The Table Url will be <3LC Table root> / <dataset_name> / <key><unique name>.json
    table_url_from_yaml = resolve_table_url([yolo_yaml_file, root_url if root_url else '', split],
                                            dataset_name,
                                            prefix='yolo_')

    # If revision_url is specified as an argument, use that Table
    if revision_url:
        try:
            table = tlc.Table.from_url(revision_url)
        except FileNotFoundError:
            raise ValueError(f'Could not find Table {revision_url} for {split} split')

        # If YAML file (--data argument) is also set, write appropriate log messages
        if yolo_yaml_file:
            try:
                root_table = tlc.Table.from_url(table_url_from_yaml)
                if not table.is_descendant_of(root_table):
                    LOGGER.info(
                        f"{tlc_prefix}Revision URL is not a descendant of the Table corresponding to the YAML file's {split} split. Ignoring YAML file."
                    )
            except FileNotFoundError:
                LOGGER.warning(
                    f'{tlc_prefix}Ignoring YAML file {yolo_yaml_file} because --tlc-{split}{"-" if split else ""}revision-url is set'
                )
        try:
            check_table_compatibility(table)
        except AssertionError as e:
            raise ValueError(f'Table {revision_url} is not compatible with YOLOv5') from e

        LOGGER.info(f'{tlc_prefix}Using {split} revision {revision_url}')
    else:

        try:
            table = tlc.Table.from_url(table_url_from_yaml)
            initial_url = table.url
            table = table.latest()
            latest_url = table.url
            if initial_url != latest_url:
                LOGGER.info(f'{tlc_prefix}Using latest version of {split} table: {latest_url.to_str()}')
            else:
                LOGGER.info(f'{tlc_prefix}Using root {split} table: {initial_url.to_str()}')
        except FileNotFoundError:
            cache_url = get_cache_file_name(table_url_from_yaml)
            table = tlc.TableFromYolo(
                url=table_url_from_yaml,
                row_cache_url=cache_url,
                dataset_name=dataset_name,
                project_name=project_name,
                input_url=yolo_yaml_file,
                root_url=root_url,
                split=split,
            )
            table.get_rows_as_binary()  # Force immediate creation of row cache
            LOGGER.info(f'{tlc_prefix}Using {split} table {table.url}')

        try:
            check_table_compatibility(table)
        except AssertionError as e:
            raise ValueError(f'Table {table_url_from_yaml.to_str()} is not compatible with YOLOv5') from e

    table.ensure_fully_defined()
    return table


def unpack_box(bbox):
    return [bbox[tlc.LABEL], bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]


def tlc_table_row_to_yolo_label(row):
    unpacked = [unpack_box(box) for box in row[tlc.BOUNDING_BOXES][tlc.BOUNDING_BOX_LIST]]
    arr = np.array(unpacked, ndmin=2, dtype=np.float32)
    if len(unpacked) == 0:
        arr = arr.reshape(0, 5)
    return arr


def xyxy_to_xywh(xyxy, height, width):
    """Converts a bounding box from XYXY_ABS to XYWH_REL format.

    :param xyxy: A bounding box in XYXY_ABS format.
    :param height: The height of the image the bounding box is in.
    :param width: The width of the image the bounding box is in.

    :returns: The bounding box in XYWH_REL format with XY being centered.
    """
    x0, y0, x1, y1 = xyxy
    return [
        (x0 + x1) / (2 * width),
        (y0 + y1) / (2 * height),
        (x1 - x0) / width,
        (y1 - y0) / height, ]


def create_tlc_info_string_before_training(metrics_collection_epochs, collect_before_training=False):
    """Prints the 3LC info before training.

    :param metrics_collection_epochs: The epochs to collect metrics for.
    :param collect_before_training: Whether to collect metrics before training.
    :param tlc_disable: Whether 3LC metrics collection is disabled.

    :returns: The 3LC info string.
    """
    if not metrics_collection_epochs and not collect_before_training:
        tlc_mc_string = 'Metrics collection disabled for this run.'
    elif not metrics_collection_epochs and collect_before_training:
        tlc_mc_string = 'Collecting metrics only before training.'
    else:
        plural_epochs = len(metrics_collection_epochs) > 1
        mc_epochs_str = ','.join(map(str, metrics_collection_epochs))
        before_string = 'before training and ' if collect_before_training else ''
        tlc_mc_string = f'Collecting metrics {before_string}for epoch{"s" if plural_epochs else ""} {mc_epochs_str}'

    return tlc_mc_string


def check_table_compatibility(table: tlc.Table) -> bool:
    """Check that the 3LC Table is compatible with YOLOv5"""

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
