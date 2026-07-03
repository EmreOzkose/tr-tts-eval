import numpy as np
import soundfile as sf


class LowLevel:
    def __init__(self, clip_threshold: float = 0.999):
        self.clip_threshold = clip_threshold

    def __call__(self, audio_path: str, name: str = "dc_offset"):
        assert name in ["dc_offset", "lufs", "peak_clips", "sibilance_ratio"]

        audio, sr = sf.read(audio_path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            return float("nan")

        x = audio.astype(np.float64)
        eps = 1e-12

        if name == "dc_offset":
            return float(np.mean(x))

        if name == "lufs":
            try:
                import pyloudnorm as pyln

                return float(pyln.Meter(sr).integrated_loudness(x))
            except Exception:
                rms = np.sqrt(np.mean(x ** 2) + eps)
                return float(20.0 * np.log10(rms + eps))

        if name == "peak_clips":
            return float(np.mean(np.abs(x) >= self.clip_threshold))

        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / sr)
        sib = spec[(freqs >= 4000.0) & (freqs <= 10000.0)].sum()
        ref = spec[(freqs >= 300.0) & (freqs < 4000.0)].sum()
        return float(sib / (ref + eps))
