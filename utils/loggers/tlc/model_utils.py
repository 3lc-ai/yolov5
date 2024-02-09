# YOLOv5 🚀 AGPL-3.0 license
"""3LC model utils."""
import os
from typing import Any

import torch

from models.experimental import attempt_load as yolov5_attempt_load
from utils.general import check_amp as yolov5_check_amp
from utils.loggers.tlc.logger import TLCLogger
from utils.torch_utils import ModelEMA as YOLOv5ModelEMA

RANK = int(os.getenv("RANK", -1))


class ModelEMA(YOLOv5ModelEMA):
    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        super().__init__(model, **kwargs)

        # Give TLCLogger a handle to the model and ema model
        TLCLogger.get_instance().register_ema(self)
        TLCLogger.get_instance().register_model(model)


def check_amp(model: torch.nn.Module) -> None:
    """Check if AMP is available and compatible with the model, but also register it with the 3LC logger."""
    amp = yolov5_check_amp(model)
    if RANK in {0, -1}:
        TLCLogger.get_instance().register_amp(amp)
    return amp


def attempt_load(*args: Any, **kwargs: Any) -> Any:
    """Steal a reference to the model used for final validation, to manipulate the model to save embeddings
    activations.
    """
    model = yolov5_attempt_load(*args, **kwargs)
    model.collect_embeddings = TLCLogger.get_instance()._settings.image_embeddings_dim > 0
    return model
