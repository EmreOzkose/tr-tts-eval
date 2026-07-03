

from pathlib import Path

import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan

from pyannote.audio import Model

from brouhaha.pipeline import RegressiveActivityDetectionPipeline

import torch
import lightning_fabric

# lightning_fabric'in torch.load çağrısını weights_only=False ile yap
_original_pl_load = lightning_fabric.utilities.cloud_io._load

def _patched_pl_load(path_or_url, map_location=None):
    return torch.load(path_or_url, map_location=map_location, weights_only=False)

lightning_fabric.utilities.cloud_io._load = _patched_pl_load

class Brouhaha:
    def __init__(self, model_path: str | None = None, name: str = "snr"):
        assert name in ["snr", "c50"]
        root = Path(__file__).resolve().parents[2]
        ckpt = Path(model_path) if model_path else root / "methods" / "brouhaha-vad" / "models" / "best" / "checkpoints" / "best.ckpt"
        model = Model.from_pretrained(ckpt, strict=False)
        self.pipeline = RegressiveActivityDetectionPipeline(segmentation=model)
        self.name = name

    def __call__(self, audio_path: str, name: str | None = None):
        metric_name = name if name is not None else self.name
        assert metric_name in ["snr", "c50"]
        pred = self.pipeline({"uri": Path(audio_path).stem, "audio": audio_path})
        return float(pred[metric_name].mean())
