"""
DNSMOS Pro metric — non-intrusive PyTorch-based MOS predictor.

Reference: https://github.com/fcumlin/DNSMOSPro
"""

import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch

warnings.filterwarnings("ignore")


class DNSMOSPro:
    """
    DNSMOS Pro — a non-intrusive, PyTorch-based MOS predictor.

    Three pretrained TorchScript models are available via the ``name`` argument:

    - ``"bvcc"``    : trained on the BVCC dataset
    - ``"nisqa"``   : trained on the NISQA dataset  (default)
    - ``"vcc2018"`` : trained on the VCC 2018 dataset

    Scores are on a MOS scale of approximately 1–5; higher is better.

    Models (~340 KB each) are downloaded automatically on first use and cached
    under ``<project_root>/model_cache/dnsmos_pro/``.

    Args:
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``).
                Defaults to CUDA when available, otherwise CPU.
    """

    _MODEL_URLS: dict = {
        "bvcc":    "https://github.com/fcumlin/DNSMOSPro/raw/main/runs/BVCC/model_best.pt",
        "nisqa":   "https://github.com/fcumlin/DNSMOSPro/raw/main/runs/NISQA/model_best.pt",
        "vcc2018": "https://github.com/fcumlin/DNSMOSPro/raw/main/runs/VCC2018/model_best.pt",
    }

    _SAMPLE_RATE: int = 16_000
    _CLIP_SAMPLES: int = 160_000   # 10 seconds × 16 000 Hz

    # STFT hyper-parameters matching DNSMOSPro training configuration
    _WIN_LENGTH: int = 320
    _HOP_LENGTH: int = 160
    _N_FFT: int = 320

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Cache directory: <project_root>/model_cache/dnsmos_pro/
        self._cache_dir = Path(__file__).resolve().parents[1] / "model_cache" / "dnsmos_pro"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Lazily loaded TorchScript models, keyed by name
        self._models: dict = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, name: str):
        """Return the requested model, downloading it first if not cached."""
        if name in self._models:
            return self._models[name]

        model_path = self._cache_dir / f"{name}_model_best.pt"
        if not model_path.exists():
            url = self._MODEL_URLS[name]
            print(f"  [DNSMOSPro] Downloading '{name}' model from {url} ...")
            urlretrieve(url, str(model_path))
            print(f"  [DNSMOSPro] Saved to {model_path}")

        model = torch.jit.load(str(model_path), map_location=torch.device(self.device))
        model.eval()
        self._models[name] = model
        return model

    @staticmethod
    def _compute_stft(samples: np.ndarray) -> np.ndarray:
        """
        Log-magnitude STFT that replicates DNSMOSPro's ``utils.stft`` defaults.

        Input : float32 array of shape (N,)
        Output: float32 array of shape (T, n_fft//2 + 1) = (T, 161)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "'librosa' is required for DNSMOSPro. "
                "Install it with: pip install librosa"
            )
        spec = librosa.stft(
            y=samples,
            win_length=DNSMOSPro._WIN_LENGTH,
            hop_length=DNSMOSPro._HOP_LENGTH,
            n_fft=DNSMOSPro._N_FFT,
        )
        spec = spec.T                 # (T, F)
        spec = np.abs(spec)           # magnitude
        spec = np.clip(spec, 1e-7, 1e7)
        spec = np.log10(spec)         # log-magnitude
        return spec                   # (T, 161)

    def _preprocess(self, audio_path: str) -> torch.Tensor:
        """Load, resample, repetitively crop, and compute STFT."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "'soundfile' is required. Install it with: pip install soundfile"
            )
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "'librosa' is required for DNSMOSPro. "
                "Install it with: pip install librosa"
            )

        samples, sr = sf.read(audio_path, dtype="float32")

        # Ensure mono
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Resample to 16 kHz if necessary
        if sr != self._SAMPLE_RATE:
            samples = librosa.resample(
                y=samples,
                orig_sr=sr,
                target_sr=self._SAMPLE_RATE,
                res_type="scipy",
            )

        # Repetitively crop to exactly 10 seconds (160 000 samples)
        if len(samples) < self._CLIP_SAMPLES:
            reps = int(np.ceil(self._CLIP_SAMPLES / len(samples)))
            samples = np.tile(samples, reps)
        samples = samples[: self._CLIP_SAMPLES]

        # STFT → FloatTensor of shape (1, 1, T, 161)
        spec = self._compute_stft(samples)
        return torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, audio_path: str, name: str = "nisqa") -> float:
        """
        Predict the MOS score for a single audio file.

        Args:
            audio_path: Absolute or relative path to the WAV file.
            name:       Model variant — one of ``"bvcc"``, ``"nisqa"`` (default),
                        or ``"vcc2018"``.

        Returns:
            Predicted MOS mean as a float in approximately [1, 5].
        """
        assert name in self._MODEL_URLS, (
            f"Unknown DNSMOSPro model '{name}'. "
            f"Choose from: {list(self._MODEL_URLS.keys())}."
        )
        model = self._ensure_model(name)
        spec = self._preprocess(audio_path)
        with torch.no_grad():
            prediction = model(spec)   # shape (1, 2): [:, 0] = MOS mean, [:, 1] = variance
        return float(prediction[:, 0].item())
