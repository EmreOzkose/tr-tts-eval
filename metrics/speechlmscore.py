"""
SpeechLMScore — speech quality metric via Speech Language Model perplexity.

Reference: https://github.com/soumimaiti/speechlmscore_tool
Paper:     https://arxiv.org/abs/2212.04559

Pipeline:
    audio → HuBERT features (layer 6) → K-Means (50 clusters) →
    LSTM Speech LM → average NLL per token

Lower score = better (more natural / fluent speech according to the speech LM).

The three required checkpoints are downloaded automatically on first use and
cached under <project_root>/model_cache/speechlmscore/:
  • HuBERT base  (facebook/hubert-base-ls960, ≈ 360 MB) — via Hugging Face hub
  • 50-cluster k-means (km50.bin, ≈ 2 MB)               — from Meta GSLM release
  • LSTM speech LM (soumi-maiti/speech-ulm-lstm, ≈ 101 MB) — via Hugging Face hub
"""

import logging
import warnings
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Inline LSTM Language Model
#  Adapted from utils/lstm_lm.py and utils/lm.py in speechlmscore_tool.
# ---------------------------------------------------------------------------

class _LSTMLM(nn.Module):
    """
    Lightweight LSTM language model over discrete speech units.

    Mirrors the LSTMLM class from utils/lstm_lm.py of the speechlmscore_tool
    repository so that state-dict keys match exactly and the pretrained
    weights from soumi-maiti/speech-ulm-lstm load without remapping.
    """

    def __init__(
        self,
        vocab_size: int,
        unit: int = 650,
        nhid: int = None,
        nlayers: int = 2,
        dropout_rate: float = 0.0,
        ignore_id: int = 0,
    ):
        super().__init__()
        ninp = unit
        if nhid is None:
            nhid = unit
        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=ignore_id)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout_rate, batch_first=True)
        self.decoder = nn.Linear(nhid, vocab_size)
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(
        self, input: torch.Tensor, hidden
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        )
        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            hidden,
        )


def _make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Return a (B, T) bool mask where True marks padding positions."""
    lengths_list = lengths.long().tolist()
    bs = len(lengths_list)
    maxlen = int(max(lengths_list))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths_list).unsqueeze(-1)
    return seq_range_expand >= seq_length_expand  # (B, T)


class _SpeechLM(nn.Module):
    """
    LM wrapper that adds <sos>/<eos> handling and NLL computation.

    Mirrors the LM class from utils/lm.py of the speechlmscore_tool
    repository.  The sub-module is named ``self.lm`` so that pretrained
    state-dict keys (prefixed ``lm.*``) load without remapping.
    """

    def __init__(self, lm: _LSTMLM, vocab_size: int, ignore_id: int = 0):
        super().__init__()
        self.lm = lm
        # In the pretrained vocab, <sos/eos> occupies the last index (52).
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.ignore_id = ignore_id

    def nll(
        self,
        text: torch.Tensor,          # (B, L)  int64 — LM vocab token IDs
        text_lengths: torch.Tensor,  # (B,)    int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-position NLL for a batch of token sequences.

        Returns:
            nll:       (B, L+1) float — per-position negative log-likelihood.
            x_lengths: (B,)    int64 — actual sequence lengths (including EOS).
        """
        batch_size = text.size(0)
        text = text[:, : int(text_lengths.max())]

        # Build input/target pair:
        #   x = [<sos>, token_1, …, token_L]
        #   t = [token_1, …, token_L, <eos>]
        x = F.pad(text, [1, 0], "constant", self.eos)
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, length in enumerate(text_lengths):
            t[i, int(length)] = self.sos
        x_lengths = text_lengths + 1

        y, _ = self.lm(x, None)  # (B, L+1, vocab_size)
        nll = F.cross_entropy(
            y.view(-1, y.shape[-1]), t.view(-1), reduction="none"
        )
        nll.masked_fill_(
            _make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0
        )
        return nll.view(batch_size, -1), x_lengths  # (B, L+1), (B,)


# ---------------------------------------------------------------------------
#  K-Means quantiser
#  Adapted from 02b_dump_km_label.py in speechlmscore_tool.
# ---------------------------------------------------------------------------

class _ApplyKmeans:
    """
    Vectorised k-means assignment using pre-trained cluster centres.

    Mirrors the ApplyKmeans class from 02b_dump_km_label.py of the
    speechlmscore_tool repository.
    """

    def __init__(self, km_path: str, device: str = "cpu"):
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "'joblib' is required for SpeechLMScore. "
                "Install with: pip install joblib"
            )
        km = joblib.load(km_path)
        C = km.cluster_centers_.T            # (D, K)
        self.C = torch.from_numpy(C).float().to(device)
        self.Cnorm = (self.C ** 2).sum(0, keepdim=True)  # (1, K)
        self.device = device

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """
        Assign each frame to its nearest cluster centre.

        Args:
            x: (T, D) float tensor of HuBERT frame features.

        Returns:
            (T,) int numpy array of cluster IDs in [0, K).
        """
        x = x.float().to(self.device)
        dist = x.pow(2).sum(1, keepdim=True) - 2.0 * (x @ self.C) + self.Cnorm
        return dist.argmin(dim=1).cpu().numpy()  # (T,)


# ---------------------------------------------------------------------------
#  Main metric class
# ---------------------------------------------------------------------------

class SpeechLMScore:
    """
    SpeechLMScore — reference-free speech quality metric based on the
    perplexity assigned by a Speech Language Model (LSTM LM over discrete
    HuBERT units).

    Scores are *average negative log-likelihood per token* — **lower is
    better** (the speech sounds more natural / fluent to the model).

    All three pretrained checkpoints are downloaded automatically on first use
    and cached under ``<project_root>/model_cache/speechlmscore/``:

    1. **HuBERT base** (``facebook/hubert-base-ls960``, ≈ 360 MB)
       — downloaded from Hugging Face hub on first call
    2. **K-Means 50** (``km50.bin``, ≈ 2 MB)
       — downloaded from Meta's GSLM release on first call
    3. **LSTM Speech LM** (``soumi-maiti/speech-ulm-lstm``, ≈ 101 MB)
       — downloaded from Hugging Face hub on first call

    All three models are loaded once in ``__init__`` and reused across every
    ``__call__`` to avoid repeated GPU-memory allocation and OOM errors.

    Args:
        device:     Target device string (e.g. ``"cuda"``, ``"cpu"``).
                    Defaults to CUDA when available, otherwise CPU.
        layer:      HuBERT transformer-layer index whose hidden states are
                    used as frame features.  Default ``6`` matches the GSLM
                    50-cluster k-means model that ships with this metric.
        max_chunk:  Maximum number of raw audio samples fed to HuBERT in a
                    single forward pass.  Reduce if you hit OOM on long files.
                    Default ``1_600_000`` ≈ 100 s at 16 kHz.
    """

    # ------------------------------------------------------------------ URLs
    _KM_URL = (
        "https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin"
    )
    _HUBERT_HF_ID = "facebook/hubert-base-ls960"
    _ULM_HF_ID = "soumi-maiti/speech-ulm-lstm"
    _ULM_FILES = ("config.yaml", "tokens.txt", "valid.loss.best.pth")

    def __init__(
        self,
        device: str | None = None,
        layer: int = 6,
        max_chunk: int = 1_600_000,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer
        self.max_chunk = max_chunk

        # Cache directory: <project_root>/model_cache/speechlmscore/
        self._cache_dir = (
            Path(__file__).resolve().parents[1] / "model_cache" / "speechlmscore"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        print("  [SpeechLMScore] Initialising — loading 3 models ...")
        self._hubert = self._load_hubert()
        self._kmeans = self._load_kmeans()
        self._lm, self._token2id, self._unk_id = self._load_lm()
        print("  [SpeechLMScore] All models ready.")

    # ------------------------------------------------------------------
    #  Model-loading helpers
    # ------------------------------------------------------------------

    def _load_hubert(self):
        """Load (and cache via HF hub) the HuBERT base feature extractor."""
        try:
            from transformers import HubertModel
        except ImportError:
            raise ImportError(
                "'transformers' is required for SpeechLMScore. "
                "Install with: pip install transformers"
            )
        print(f"  [SpeechLMScore] Loading HuBERT ({self._HUBERT_HF_ID}) ...")
        model = HubertModel.from_pretrained(self._HUBERT_HF_ID)
        return model.eval().to(self.device)

    def _load_kmeans(self) -> _ApplyKmeans:
        """Download (once) and load the 50-cluster k-means model."""
        km_path = self._cache_dir / "km50.bin"
        if not km_path.exists():
            print(
                f"  [SpeechLMScore] Downloading k-means model from "
                f"{self._KM_URL} ..."
            )
            urlretrieve(self._KM_URL, str(km_path))
            print(f"  [SpeechLMScore] Saved to {km_path}")
        print("  [SpeechLMScore] Loading k-means model ...")
        return _ApplyKmeans(str(km_path), device=self.device)

    def _load_lm(self) -> Tuple[_SpeechLM, dict, int]:
        """Download (once) and load the pretrained LSTM Speech LM."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "'PyYAML' is required for SpeechLMScore. "
                "Install with: pip install pyyaml"
            )

        ulm_dir = self._cache_dir / "speech-ulm-lstm"
        ulm_dir.mkdir(parents=True, exist_ok=True)

        # Download missing files from Hugging Face hub
        base_url = f"https://huggingface.co/{self._ULM_HF_ID}/resolve/main"
        for fname in self._ULM_FILES:
            fpath = ulm_dir / fname
            if not fpath.exists():
                url = f"{base_url}/{fname}"
                print(f"  [SpeechLMScore] Downloading '{fname}' from {url} ...")
                urlretrieve(url, str(fpath))
                print(f"  [SpeechLMScore] Saved to {fpath}")

        # ---- Parse LM config ----
        with open(ulm_dir / "config.yaml", "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        lm_conf: dict = cfg.get("lm_conf", {})
        ignore_id: int = cfg.get("model_conf", {}).get("ignore_id", 0)

        # ---- Build vocabulary lookup ----
        with open(ulm_dir / "tokens.txt", "r", encoding="utf-8") as fh:
            token_list = [line.rstrip() for line in fh]
        vocab_size = len(token_list)
        token2id = {tok: idx for idx, tok in enumerate(token_list)}
        unk_id = token2id.get("<unk>", 1)

        # ---- Instantiate the model (architecture from config.yaml) ----
        # lm_conf keys: unit=1024, nlayers=3, dropout_rate=0.2
        # nhid is absent → defaults to unit inside _LSTMLM.__init__
        lm_inner = _LSTMLM(vocab_size=vocab_size, ignore_id=ignore_id, **lm_conf)
        lm = _SpeechLM(lm=lm_inner, vocab_size=vocab_size, ignore_id=ignore_id)

        # ---- Load pretrained weights ----
        model_path = ulm_dir / "valid.loss.best.pth"
        print("  [SpeechLMScore] Loading LSTM LM weights ...")
        try:
            # Prefer weights_only=True (safer, PyTorch ≥ 1.13)
            state = torch.load(
                str(model_path), map_location=self.device, weights_only=True
            )
        except TypeError:
            # Older PyTorch versions do not accept weights_only keyword
            state = torch.load(str(model_path), map_location=self.device)
        lm.load_state_dict(state)
        return lm.eval().to(self.device), token2id, unk_id

    # ------------------------------------------------------------------
    #  Inference helpers
    # ------------------------------------------------------------------

    def _extract_hubert_features(self, audio_path: str) -> torch.Tensor:
        """
        Load audio, resample to 16 kHz, and return HuBERT hidden-state
        features of shape (T, D) for the configured layer.
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "'soundfile' is required for SpeechLMScore. "
                "Install with: pip install soundfile"
            )
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "'librosa' is required for SpeechLMScore. "
                "Install with: pip install librosa"
            )

        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16_000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16_000)

        x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)  # (1, T)
        feats = []
        with torch.no_grad():
            for start in range(0, x.size(1), self.max_chunk):
                chunk = x[:, start : start + self.max_chunk]
                hs = self._hubert(chunk, output_hidden_states=True).hidden_states
                feats.append(hs[self.layer])  # (1, T_i, D)
        return torch.cat(feats, dim=1).squeeze(0)  # (T_total, D)

    def _compute_nll(self, token_ids: np.ndarray) -> float:
        """Compute average NLL per token for one utterance."""
        if len(token_ids) == 0:
            return 0.0
        ids = (
            torch.tensor(token_ids, dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )
        lengths = torch.tensor([len(token_ids)], dtype=torch.long).to(self.device)
        with torch.no_grad():
            nll, x_lengths = self._lm.nll(ids, lengths)
        total_nll = float(nll[0].sum().item())
        n_tokens = int(x_lengths[0].item())
        return total_nll / n_tokens if n_tokens > 0 else 0.0

    # ------------------------------------------------------------------
    #  Public interface — matches the pattern of all other metrics
    # ------------------------------------------------------------------

    def __call__(self, audio_path: str) -> float:
        """
        Compute the SpeechLMScore for a single audio file.

        Args:
            audio_path: Absolute or relative path to a WAV file.
                        Non-16 kHz / stereo files are handled automatically.

        Returns:
            Average negative log-likelihood per token (float).
            **Lower values indicate more natural / fluent speech.**
        """
        # 1. HuBERT feature extraction → (T, D) tensor
        feats = self._extract_hubert_features(audio_path)

        # 2. K-Means quantisation → (T,) int array of cluster IDs [0, 49]
        cluster_ids = self._kmeans(feats)

        # 3. Map cluster IDs to LM vocabulary indices.
        #    tokens.txt stores cluster IDs as their string representations
        #    (e.g. cluster 7 → "7"), so we convert each int to str for lookup.
        token_ids = np.array(
            [self._token2id.get(str(int(cid)), self._unk_id) for cid in cluster_ids],
            dtype=np.int64,
        )

        # 4. Average NLL per token under the speech LM
        return self._compute_nll(token_ids)
