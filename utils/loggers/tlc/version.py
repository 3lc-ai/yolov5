# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Utility to check the 3LC version is compatible with the integration."""

import tlc
from packaging import version

from utils.loggers.tlc.constants import TLC_VERSION_REQUIRED

def check_tlc_version() -> None:
    """
    Check that the available 3LC version is supported.

    """
    required_version = version.parse(TLC_VERSION_REQUIRED)
    installed_version = version.parse(tlc.__version__)

    if installed_version < required_version:
        installed_str = ".".join(str(part) for part in installed_version.release[:3])
        raise ValueError(
            f"You are using 3lc=={installed_str}. "
            f"This version of the integration is intended for 3lc>={required_version}. "
            "Please upgrade 3lc with `pip install --upgrade 3lc`."
        )
