# YOLOv5 🚀 AGPL-3.0 license
"""
3LC Utils
"""

import enum
from pathlib import Path

import numpy as np
from tlc.core.builtins.constants.column_names import BOUNDING_BOX_LIST, BOUNDING_BOXES, LABEL, X0, X1, Y0, Y1
from tlc.core.objects.table import Table
from tlc.core.objects.tables.from_url import TableFromYolo
from tlc.core.objects.tables.from_url.utils import get_cache_file_name, resolve_table_url
from tlc.core.objects.tables.system_tables.indexing_tables import TableIndexingTable


def get_or_create_tlc_table(input_url, split, revision_url='', root_url=None):
    # Infer dataset and project names
    dataset_name_base = Path(input_url).stem
    dataset_name = dataset_name_base + '-' + split
    project_name = 'yolov5-' + dataset_name_base

    # Create table url
    initial_table_url = resolve_table_url([input_url, root_url if root_url else '', split], dataset_name)

    # Get existing table or create a new one
    try:
        table = Table.from_url(initial_table_url)
    except FileNotFoundError:
        cache_url = get_cache_file_name(initial_table_url)
        table = TableFromYolo(
            url=initial_table_url,
            row_cache_url=cache_url,
            dataset_name=dataset_name,
            project_name=project_name,
            input_url=input_url,
            root_url=root_url,
            split=split,
        )
        table.write_to_url()
        table.get_rows_as_binary()  # Force immediate creation of row cache

    TableIndexingTable.instance().ensure_fully_defined()
    if revision_url:
        revision_table = Table.from_url(revision_url)
        if not revision_table.is_descendant_of(table):
            raise ValueError(f'Revision {revision_url} is not a descendant of {initial_table_url.to_str()}')

        table = revision_table
    else:
        table = table.latest()

    table.ensure_fully_defined()
    return table


def unpack_box(bbox):
    return [bbox[LABEL], bbox[X0], bbox[Y0], bbox[X1], bbox[Y1]]


def tlc_table_row_to_yolo_label(row):
    unpacked = [unpack_box(box) for box in row[BOUNDING_BOXES][BOUNDING_BOX_LIST]]
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


class DatasetMode(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    COLLECT = 'collect'
