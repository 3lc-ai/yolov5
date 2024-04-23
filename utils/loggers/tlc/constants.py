# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Constants used for 3LC integration."""

from utils.general import colorstr

# 3LC Constants
TLC_PREFIX = "3LC://"
TLC_COLORSTR = colorstr("3LC: ")
TLC_TRAIN_PATH = "3lc_train"
TLC_VAL_PATH = "3lc_val"
TLC_COLLECT_PATH = "3lc_collect"
TLC_VERSION_REQUIRED = "2.3.0"

# Column names
TRAINING_PHASE = "Training Phase"
