import torch
import warnings

warnings.filterwarnings("ignore")


class UTMOSv2:
    """
    UTMOSv2 (VoiceMOS Challenge 2024 winner) naturalness MOS predictor.
    Scores range from 1 (worst) to 5 (best).

    Args:
        device: Target device string (e.g. "cuda", "cpu").  Defaults to CUDA
                when available, otherwise CPU.
    """

    def __init__(self, device: str | None = None):
        try:
            import utmosv2 as _utmosv2
        except ImportError:
            raise ImportError(
                "'utmosv2' library is missing. "
                "Please run: pip install git+https://github.com/sarulab-speech/UTMOSv2.git"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _utmosv2.create_model(pretrained=True)
        try:
            self._model = self._model.to(self.device)
        except AttributeError:
            # Some wrapper classes handle device placement internally.
            pass

    @staticmethod
    def _to_float(value):
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def __call__(self, audio_path: str) -> float:
        """
        Predict the UTMOSv2 naturalness MOS score for a single audio file.

        Args:
            audio_path: Absolute or relative path to the WAV file.

        Returns:
            Predicted MOS as a float in [1, 5].
        """
        score = self._model.predict(input_path=audio_path)
        return self._to_float(score)
