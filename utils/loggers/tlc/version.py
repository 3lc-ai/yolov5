# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Utility to check the 3LC version is compatible with the integration."""

import tlc
from packaging import version

from utils.general import LOGGER, colorstr
from utils.loggers.tlc.constants import TLC_VERSION_REQUIRED

def check_tlc_version() -> None:
    """
    Check that the available 3LC version is supported.

    """
    required_version = version.parse(TLC_VERSION_REQUIRED)
    installed_version = version.parse(tlc.__version__)

    if installed_version < required_version:
        prefix = colorstr("red", "WARNING: ")
        LOGGER.warn(f"{prefix}You are using 3lc version {installed_version}, "
                    f"but the integration is intended for {required_version} or higher. "
                    "Please upgrade 3lc if you run into problems.")