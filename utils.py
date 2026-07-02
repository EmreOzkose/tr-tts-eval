
import builtins
import logging
import os
import warnings


GREEN = "\033[92m"
RESET = "\033[0m"


_SUPPRESSED_PRINTS = {
    "Using default parameters optimized on Brouhaha",
    "ESPnet is not installed",
}

_real_print = builtins.print


def _filtered_print(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    if any(s in text for s in _SUPPRESSED_PRINTS):
        return
    if text.startswith("{'data_type'") or text.startswith("{'model'"):
        return
    _real_print(*args, **kwargs)


def suppress_noisy_logs():
    warnings.filterwarnings("ignore")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("s3prl").setLevel(logging.ERROR)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    builtins.print = _filtered_print


def is_better(metric: str, cand: float, best: float) -> bool:
    # For offset/clipping metrics, lower is better.
    if metric.endswith("dc_offset"):
        return abs(cand) < abs(best)
    if metric.endswith("peak_clips"):
        return cand < best
    if metric.startswith("scoreq_") and metric.endswith("_ref"):
        return cand < best
    # WER: lower is better (0 = perfect transcription).
    if metric in ("wer", "wer_v2"):
        return cand < best
    return cand > best
