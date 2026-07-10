from pathlib import Path
import sys


class NatScoreMetric:
    """
    NatScore metric wrapper.

    Returns a pointwise naturalness score where higher values indicate
    more natural speech.
    """

    def __init__(
        self,
        model_id: str = "harrrshall/natscore-small-v0",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device

        try:
            import natscore as ns
        except ImportError:
            # Fallback for local checkout usage without editable install.
            repo_root = Path(__file__).resolve().parents[2]
            natscore_src = repo_root / "methods" / "natscore" / "src"
            if str(natscore_src) not in sys.path:
                sys.path.insert(0, str(natscore_src))
            try:
                import natscore as ns
            except ImportError as exc:
                raise ImportError(
                    "NatScore is not importable. Install with: "
                    "pip install -e 'methods/natscore[dev]'"
                ) from exc

        self._scorer = ns.load(self.model_id, device=self.device)

    def __call__(self, audio_path: str) -> float:
        return float(self._scorer.score(audio_path))
