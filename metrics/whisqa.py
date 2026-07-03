from pathlib import Path
import sys

import torch
import torchaudio


class WhiSQA:
    """
    WhiSQA metric wrapper.

    Uses the WhiSQA checkpoints from methods/WhiSQA/checkpoints and returns
    MOS scaled to [1, 5]. For the multi-head model, the first output (MOS)
    is returned.
    """

    def __init__(
        self,
        model_type: str = "single",
        checkpoints_dir: str | None = None,
        device: str | None = None,
    ):
        if model_type not in {"single", "multi"}:
            raise ValueError("model_type must be either 'single' or 'multi'")

        repo_root = Path(__file__).resolve().parents[2]
        whisqa_root = repo_root / "methods" / "WhiSQA"

        if str(whisqa_root) not in sys.path:
            sys.path.insert(0, str(whisqa_root))

        try:
            from models.whisper_ni_predictors import (
                whisperMetricPredictorEncoderLayersTransformerSmall,
                whisperMetricPredictorEncoderLayersTransformerSmalldim,
            )
        except ImportError as exc:
            raise ImportError(
                "WhiSQA models could not be imported. "
                f"Expected WhiSQA source under: {whisqa_root}"
            ) from exc

        self.model_type = model_type
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else (whisqa_root / "checkpoints")

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.model_type == "single":
            self._model = whisperMetricPredictorEncoderLayersTransformerSmall()
            checkpoint_path = self.checkpoints_dir / "single_head_model.pt"
        else:
            self._model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
            checkpoint_path = self.checkpoints_dir / "multi_head_model.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"WhiSQA checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _prepare_waveform(audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono to satisfy WhiSQA input shape expectations.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        return waveform

    def __call__(self, audio_path: str) -> float:
        waveform = self._prepare_waveform(audio_path).to(self.device)
        with torch.no_grad():
            score = self._model(waveform)

        if self.model_type == "single":
            return float(score.item() * 5.0)

        score = score.squeeze(0)
        return float(score[0].item() * 5.0)
