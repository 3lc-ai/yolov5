# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Utility to check the 3LC version is compatible with the integration."""

import importlib.metadata

from packaging import version

from utils.loggers.tlc.constants import TLC_ULTRALYTICS_VERSION_REQUIRED, TLC_VERSION_REQUIRED

INSTALL_COMMAND = (
    f'pip install --upgrade "3lc>={TLC_VERSION_REQUIRED}" "3lc-ultralytics>={TLC_ULTRALYTICS_VERSION_REQUIRED}"'
    " --extra-index-url https://pypi.3lc.ai/public/repositories/releases-public"
)


def check_tlc_version() -> None:
    """
    Check that the available 3LC packages are supported by the integration.

    The integration requires both `3lc` and `3lc-ultralytics`, which are published on 3LC's
    public package index, not on PyPI.
    """
    for package, required_str in (("3lc", TLC_VERSION_REQUIRED), ("3lc-ultralytics", TLC_ULTRALYTICS_VERSION_REQUIRED)):
        try:
            installed = version.parse(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            raise ImportError(
                f"The 3LC integration requires {package}, which is not installed. Install it with `{INSTALL_COMMAND}`."
            ) from None

        required = version.parse(required_str)
        if installed < required:
            installed_str = ".".join(str(part) for part in installed.release[:3])
            raise ValueError(
                f"You are using {package}=={installed_str}. "
                f"This version of the integration is intended for {package}>={required}. "
                f"Please upgrade with `{INSTALL_COMMAND}`."
            )
