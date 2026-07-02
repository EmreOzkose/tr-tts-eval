"""
chunk_speaker_similarity.py
───────────────────────────
Temporal consistency analysis for TTS evaluation.

Extends SpeakerSimilarity with two chunking strategies:
  A) Fixed-duration windows (overlapping)
  B) Word-level boundaries via WhisperX forced alignment

For each chunk computes:
  - speaker_sim vs reference clip   → absolute drift
  - speaker_sim vs adjacent chunk   → relative consistency

Outputs per-utterance and per-model aggregate statistics
suitable for the SPECOM paper (drift slope, consistency std,
first→last delta).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── your existing class (imported or pasted here) ───────────────────────────
# from speaker_similarity import SpeakerSimilarity
# For standalone use paste the class below; otherwise import it.


class SpeakerSimilarity:
    def __init__(self, model_name: str = "english", device: str | None = None):
        import wespeaker
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = wespeaker.load_model(model_name)
        self.model.set_device(self.device)
        self._patch_compute_features_device()
        self.embedding_cache: dict[str, torch.Tensor] = {}

    def _patch_compute_features_device(self):
        original = self.model.compute_features

        def patched(wavform, sample_rate=16000, cmn=True):
            feat = original(wavform, sample_rate=sample_rate, cmn=cmn)
            if isinstance(feat, torch.Tensor):
                return feat.to(self.model.device)
            return feat

        self.model.compute_features = patched

    def _get_embedding(self, path: str):
        if path not in self.embedding_cache:
            self.embedding_cache[path] = self.model.extract_embedding(path)
        return self.embedding_cache[path]

    def _compute_similarity(self, path1: str, path2: str) -> float:
        e1, e2 = self._get_embedding(path1), self._get_embedding(path2)
        if e1 is None or e2 is None:
            return 0.0
        return float(self.model.cosine_similarity(e1, e2))

    def __call__(self, path1: str, path2: str) -> float:
        try:
            return self._compute_similarity(path1, path2)
        except RuntimeError as exc:
            msg = str(exc)
            if "Input type" in msg and "weight type" in msg and self.device != "cpu":
                self.device = "cpu"
                self.model.set_device("cpu")
                self._patch_compute_features_device()
                self.embedding_cache.clear()
                return self._compute_similarity(path1, path2)
            raise


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    """Similarity scores for a single chunk."""
    chunk_idx: int
    start_sec: float
    end_sec: float
    norm_position: float          # 0.0 = utterance start, 1.0 = end
    sim_vs_reference: float       # absolute: chunk vs. speaker reference clip
    sim_vs_gold_chunk: float      # absolute: chunk vs. gold's same time region
    sim_vs_prev_chunk: float      # relative: chunk vs. previous chunk (NaN for idx=0)


@dataclass
class UtteranceResult:
    """Temporal consistency stats for one utterance × one model."""
    utt_id: str
    model: str
    strategy: str                  # "fixed" | "word"
    duration_sec: float
    n_chunks: int
    chunks: list[ChunkResult] = field(default_factory=list)

    # ── aggregate stats (computed post-hoc) ──
    drift_slope: float = float("nan")          # linear regression: position → sim_vs_ref
    consistency_std: float = float("nan")      # std of sim_vs_reference across chunks
    first_last_delta: float = float("nan")     # sim[-1] - sim[0]  (negative = degrading)
    gold_drift_slope: float = float("nan")     # same but vs gold chunks
    mean_sim_vs_ref: float = float("nan")

    def compute_aggregates(self):
        if not self.chunks:
            return
        sims = [c.sim_vs_reference for c in self.chunks]
        positions = [c.norm_position for c in self.chunks]
        gold_sims = [c.sim_vs_gold_chunk for c in self.chunks]

        self.mean_sim_vs_ref = float(np.mean(sims))
        self.consistency_std = float(np.std(sims))
        self.first_last_delta = sims[-1] - sims[0]

        # linear regression slope (position → sim)
        if len(sims) >= 2:
            self.drift_slope = float(np.polyfit(positions, sims, 1)[0])
            valid_gold = [(p, s) for p, s in zip(positions, gold_sims) if not np.isnan(s)]
            if len(valid_gold) >= 2:
                gp, gs = zip(*valid_gold)
                self.gold_drift_slope = float(np.polyfit(gp, gs, 1)[0])


# ── audio helpers ────────────────────────────────────────────────────────────

def _load_mono_16k(path: str) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at 16 kHz."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy(), 16000


def _save_temp_wav(audio: np.ndarray, sr: int = 16000) -> str:
    """Write numpy array to a named temp wav file; caller must delete."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


# ── chunking strategies ──────────────────────────────────────────────────────

def fixed_chunks(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    min_chunk_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Returns list of (start_sec, end_sec) with overlapping windows.
    Last partial window kept only if >= min_chunk_sec.
    """
    total = len(audio) / sr
    segments = []
    start = 0.0
    while start < total:
        end = min(start + window_sec, total)
        if (end - start) >= min_chunk_sec:
            segments.append((start, end))
        if end >= total:
            break
        start += hop_sec
    return segments


def word_chunks(
    audio_path: str,
    group_size: int = 4,
    language: str = "tr",
) -> list[tuple[float, float]]:
    """
    Word-boundary chunks via WhisperX forced alignment.
    Requires:  pip install whisperx

    Groups consecutive words into segments of `group_size` words.
    Returns list of (start_sec, end_sec).
    """
    try:
        import whisperx
    except ImportError as e:
        raise ImportError(
            "whisperx is required for word-level chunking.\n"
            "Install with: pip install whisperx"
        ) from e

    # Force WhisperX onto CPU to avoid GPU-specific compute type issues.
    device = "cpu"

    # transcribe + align
    model = _load_whisperx_transcriber(device, language)
    result = model.transcribe(audio_path, language=language)

    align_model, metadata = _get_whisperx_aligner(device, language)
    aligned = whisperx.align(result["segments"], align_model, metadata,
                              audio_path, device)

    words = [w for seg in aligned["segments"] for w in seg.get("words", [])]
    words = [w for w in words if "start" in w and "end" in w]

    if not words:
        return []

    segments = []
    for i in range(0, len(words), group_size):
        group = words[i: i + group_size]
        segments.append((group[0]["start"], group[-1]["end"]))

    return segments


@lru_cache(maxsize=4)
def _get_whisperx_transcriber(device: str, language: str, compute_type: str):
    import whisperx

    return whisperx.load_model(
        "large-v3",
        device,
        language=language,
        compute_type=compute_type,
    )


def _load_whisperx_transcriber(device: str, language: str):
    preferred_compute_type = "float16" if device.startswith("cuda") else "int8"
    try:
        return _get_whisperx_transcriber(device, language, preferred_compute_type)
    except (RuntimeError, ValueError) as exc:
        if "float16" not in str(exc) and "compute type" not in str(exc).lower():
            raise
        return _get_whisperx_transcriber(device, language, "float32")


@lru_cache(maxsize=4)
def _get_whisperx_aligner(device: str, language: str):
    import whisperx

    return whisperx.load_align_model(language, device)


# ── main analyzer ────────────────────────────────────────────────────────────

class ChunkSpeakerSimilarity:
    """
    Temporal consistency analyzer.

    Parameters
    ----------
    speaker_sim : SpeakerSimilarity
        Your existing scorer instance.
    strategy : "fixed" | "word"
        Chunking strategy.
    window_sec : float
        Window length for fixed strategy.
    hop_sec : float
        Hop size for fixed strategy.
    word_group_size : int
        Words per chunk for word strategy.
    language : str
        Language code for WhisperX (word strategy only).
    """

    def __init__(
        self,
        speaker_sim: SpeakerSimilarity,
        strategy: str = "fixed",
        window_sec: float = 2.0,
        hop_sec: float = 1.0,
        word_group_size: int = 4,
        language: str = "tr",
    ):
        assert strategy in ("fixed", "word"), "strategy must be 'fixed' or 'word'"
        self.sim = speaker_sim
        self.strategy = strategy
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.word_group_size = word_group_size
        self.language = language

    # ── internal helpers ─────────────────────────────────────────────────────

    def _get_segments(
        self, audio: np.ndarray, sr: int, audio_path: str
    ) -> list[tuple[float, float]]:
        if self.strategy == "fixed":
            return fixed_chunks(audio, sr, self.window_sec, self.hop_sec)
        return word_chunks(audio_path, self.word_group_size, self.language)

    @staticmethod
    def _slice_audio(
        audio: np.ndarray, sr: int, start: float, end: float
    ) -> np.ndarray:
        return audio[int(start * sr): int(end * sr)]

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        synth_path: str,
        reference_path: str,
        gold_path: Optional[str] = None,
        utt_id: str = "utt",
        model_name: str = "model",
    ) -> UtteranceResult:
        """
        Compute temporal chunk-level speaker similarity for one utterance.

        Parameters
        ----------
        synth_path : str
            Path to synthesized audio.
        reference_path : str
            Short speaker reference clip (used for all absolute comparisons).
        gold_path : str, optional
            Gold (human) recording of the same utterance.
            If provided, each synthetic chunk is also compared to the
            temporally aligned gold chunk.
        utt_id : str
            Utterance identifier for logging.
        model_name : str
            TTS model name for logging.
        """
        synth_audio, sr = _load_mono_16k(synth_path)
        duration = len(synth_audio) / sr

        gold_audio: Optional[np.ndarray] = None
        gold_duration: Optional[float] = None
        if gold_path:
            gold_audio, _ = _load_mono_16k(gold_path)
            gold_duration = len(gold_audio) / sr

        segments = self._get_segments(synth_audio, sr, synth_path)
        if not segments:
            raise ValueError(f"No chunks produced for {synth_path}")

        result = UtteranceResult(
            utt_id=utt_id,
            model=model_name,
            strategy=self.strategy,
            duration_sec=duration,
            n_chunks=len(segments),
        )

        tmp_files: list[str] = []
        prev_chunk_tmp: Optional[str] = None

        try:
            for idx, (start, end) in enumerate(segments):
                norm_pos = (start + (end - start) / 2) / duration

                # ── synthetic chunk → temp file ──────────────────────────
                chunk_audio = self._slice_audio(synth_audio, sr, start, end)
                chunk_tmp = _save_temp_wav(chunk_audio)
                tmp_files.append(chunk_tmp)

                # ── sim vs. speaker reference (absolute drift) ────────────
                sim_ref = self.sim(chunk_tmp, reference_path)

                # ── sim vs. gold chunk (temporally aligned) ───────────────
                sim_gold = float("nan")
                if gold_audio is not None and gold_duration is not None:
                    # scale time position to gold duration
                    g_start = (start / duration) * gold_duration
                    g_end = (end / duration) * gold_duration
                    g_end = min(g_end, gold_duration)
                    gold_chunk = self._slice_audio(gold_audio, sr, g_start, g_end)
                    if len(gold_chunk) > int(0.3 * sr):   # skip if < 300 ms
                        gold_tmp = _save_temp_wav(gold_chunk)
                        tmp_files.append(gold_tmp)
                        sim_gold = self.sim(chunk_tmp, gold_tmp)

                # ── sim vs. previous chunk (relative consistency) ─────────
                sim_prev = float("nan")
                if prev_chunk_tmp is not None:
                    sim_prev = self.sim(chunk_tmp, prev_chunk_tmp)

                result.chunks.append(ChunkResult(
                    chunk_idx=idx,
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    norm_position=round(norm_pos, 4),
                    sim_vs_reference=round(sim_ref, 4),
                    sim_vs_gold_chunk=round(sim_gold, 4),
                    sim_vs_prev_chunk=round(sim_prev, 4),
                ))

                prev_chunk_tmp = chunk_tmp

        finally:
            for f in tmp_files:
                Path(f).unlink(missing_ok=True)

        result.compute_aggregates()
        return result

    def analyze_batch(
        self,
        utterances: list[dict],
        show_progress: bool = True,
    ) -> list[UtteranceResult]:
        """
        Batch analysis over multiple utterances.

        Each item in `utterances` should be a dict with keys:
            utt_id        : str
            model         : str
            synth_path    : str
            reference_path: str
            gold_path     : str  (optional)

        Example
        -------
        utterances = [
            {
                "utt_id": "utt_001",
                "model": "cosyvoice",
                "synth_path": "outputs/cosyvoice/utt_001.wav",
                "reference_path": "reference/speaker.wav",
                "gold_path": "gold/utt_001.wav",
            },
            ...
        ]
        """
        results = []
        iterator = utterances
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(utterances, desc=f"ChunkSim [{self.strategy}]")
            except ImportError:
                pass

        for item in iterator:
            try:
                r = self.analyze(
                    synth_path=item["synth_path"],
                    reference_path=item["reference_path"],
                    gold_path=item.get("gold_path"),
                    utt_id=item["utt_id"],
                    model_name=item["model"],
                )
                results.append(r)
            except Exception as exc:
                print(f"[WARN] {item['utt_id']} / {item['model']} failed: {exc}")

        return results


# ── aggregate reporting ───────────────────────────────────────────────────────

def aggregate_by_model(results: list[UtteranceResult]) -> dict[str, dict]:
    """
    Compute per-model summary statistics across all utterances.

    Returns
    -------
    {
        "cosyvoice": {
            "n_utterances": 100,
            "mean_sim_vs_ref": 0.882,
            "drift_slope_mean": -0.031,      # negative = degrading toward end
            "drift_slope_std": 0.012,
            "consistency_std_mean": 0.018,   # lower = more stable
            "first_last_delta_mean": -0.024,
            ...
        },
        ...
    }
    """
    from collections import defaultdict
    buckets: dict[str, list[UtteranceResult]] = defaultdict(list)
    for r in results:
        buckets[r.model].append(r)

    summary = {}
    for model, utts in buckets.items():
        def _mean(attr):
            vals = [getattr(u, attr) for u in utts if not np.isnan(getattr(u, attr))]
            return float(np.mean(vals)) if vals else float("nan")

        def _std(attr):
            vals = [getattr(u, attr) for u in utts if not np.isnan(getattr(u, attr))]
            return float(np.std(vals)) if vals else float("nan")

        summary[model] = {
            "n_utterances": len(utts),
            "mean_n_chunks": float(np.mean([u.n_chunks for u in utts])),
            "mean_sim_vs_ref": _mean("mean_sim_vs_ref"),
            "drift_slope_mean": _mean("drift_slope"),
            "drift_slope_std": _std("drift_slope"),
            "gold_drift_slope_mean": _mean("gold_drift_slope"),
            "consistency_std_mean": _mean("consistency_std"),
            "first_last_delta_mean": _mean("first_last_delta"),
        }
    return summary


def save_results(results: list[UtteranceResult], path: str):
    """Serialize results to JSON."""
    data = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} utterance results → {path}")


# ── minimal usage example ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Instantiate your existing scorer
    base_sim = SpeakerSimilarity()

    # 2. Wrap with chunk analyzer — try both strategies
    for strategy, kwargs in [
        ("fixed", {"window_sec": 2.0, "hop_sec": 1.0}),
        ("word",  {"word_group_size": 4, "language": "tr"}),
    ]:
        analyzer = ChunkSpeakerSimilarity(
            speaker_sim=base_sim,
            strategy=strategy,
            **kwargs,
        )

        # 3. Build utterance list (long group, 4 models × 100 utts)
        # utterances = build_utterance_list(...)   # your path builder

        # 4. Run batch
        # results = analyzer.analyze_batch(utterances)

        # 5. Save per-utterance detail + model summary
        # save_results(results, f"chunk_results_{strategy}.json")
        # summary = aggregate_by_model(results)
        # print(json.dumps(summary, indent=2))

        print(f"[{strategy}] analyzer ready — "
              f"window={kwargs.get('window_sec', '—')}s  "
              f"hop={kwargs.get('hop_sec', '—')}s  "
              f"word_group={kwargs.get('word_group_size', '—')}")