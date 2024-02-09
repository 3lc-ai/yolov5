# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""3LC yolo implementation to collect embeddings."""
import torch

from models.yolo import DetectionModel
from utils.plots import feature_visualization

global_activations = []


class TLCDetectionModel(DetectionModel):
    def __init__(self, *args, **kwargs):
        self.collect_embeddings = False
        super().__init__(*args, **kwargs)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            if "SPPF" in m.type and self.collect_embeddings:
                # attempt_load() does some spurious forward passes, don't want those NaN activations
                if not torch.isnan(x).any():
                    activations = x.mean(dim=(2, 3))  # TODO: Try max instead of mean
                    global_activations.append(activations.detach().cpu().numpy())
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
