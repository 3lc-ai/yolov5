# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Utility to check the 3LC version is compatible with the integration."""

import tlc
from packaging import version

from utils.loggers.tlc.constants import TLC_VERSION_REQUIRED


def check_tlc_version() -> None:
    """
    Check that the available 3LC version is supported.

    :raises: ValueError if the 3LC version is not supported.
    """
    required_version = version.parse(TLC_VERSION_REQUIRED)
    installed_version = version.parse(tlc.__version__)

    if installed_version < required_version:
        raise ValueError(
            f"tlc version {required_version} or higher is required, but version {installed_version} is installed."
            " Upgrade tlc or use an older version of the YOLOv5 integration."
        )
