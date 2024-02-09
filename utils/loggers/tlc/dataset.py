# YOLOv5 🚀 AGPL-3.0 license
"""
3LC check_dataset patch - required for DDP train.py for RANK != 0
"""

from __future__ import annotations

import tlc

from utils.loggers.tlc.constants import TLC_TRAIN_PATH, TLC_VAL_PATH
from utils.loggers.tlc.settings import Settings
from utils.loggers.tlc.utils import get_names_from_yolo_table, tlc_check_dataset


def check_dataset(data_file: str) -> dict[str, tlc.Table | int | dict[str, int]]:
    """Load the 3LC dataset (and check for errors) for training."""
    tables = tlc_check_dataset(data_file)

    sampling_weights = Settings.from_env().sampling_weights

    assert "train" in tables and "val" in tables, f"Could not find train and val splits in dataset {data_file}."

    names = get_names_from_yolo_table(tables["train"])

    data_dict = {
        "nc": len(names),
        "names": names,
        "train": (TLC_TRAIN_PATH, tables["train"], sampling_weights),
        "val": (TLC_VAL_PATH, tables["val"], False),
    }

    return data_dict
