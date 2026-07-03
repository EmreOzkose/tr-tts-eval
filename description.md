# Metric Descriptions

This document summarizes the metrics used in the evaluation pipeline. Grouping follows
the paper's Section 4 (Evaluation Methodology): reference-free naturalness/quality
predictors, intelligibility/signal-level metrics, speaker similarity, distributional
fidelity, and low-level acoustic descriptors.

## Reference-Free Naturalness and Quality Predictors

### audiobox_pq
Audiobox PQ is a learned perceptual quality score produced by the Audiobox Aesthetics model.
It is designed to reflect perceived audio quality from a listener perspective.
Higher is generally better.

### dnsmos_pro
DNSMOS-Pro is a non-intrusive, PyTorch-based MOS predictor (https://github.com/fcumlin/DNSMOSPro).
Three pretrained heads are available, each trained on a different MOS dataset: `bvcc`, `nisqa`,
and `vcc2018`. The bare `dnsmos_pro` metric name is an alias for the `nisqa` head. Scores are on
an approximate 1-5 MOS scale.
Higher is better.

### dnsmos_pro_bvcc
DNSMOS-Pro scored with the head trained on the BVCC dataset.
Higher is better.

### dnsmos_pro_nisqa
DNSMOS-Pro scored with the head trained on the NISQA dataset (same as the bare `dnsmos_pro` metric).
Higher is better.

### dnsmos_pro_vcc2018
DNSMOS-Pro scored with the head trained on the VCC2018 dataset.
Higher is better.

### scoreq_synthetic_nr
SCOREQ (Speech QUality Assessment with Contrastive Regression, NeurIPS 2024), `synthetic`
data-domain / `nr` (no-reference) mode. Trained on synthetic-speech quality judgments and scores
audio without needing a reference signal.
Higher is better.
Other SCOREQ variants (`natural_nr`, `natural_ref`, `synthetic_ref`) are implemented in
`metrics/scoreq_metric.py` but are not part of the paper's headline metrics.

### whisqa
WhiSQA is a non-intrusive speech quality predictor built on Whisper encoder features
(International Conference on Speech and Computer 2025). Uses the "single-head" checkpoint.
Higher is better.

### natscore
NatScore is a small, preference-supervised naturalness scorer for modern neural TTS.
Higher is better.

### speechlmscore
SpeechLMScore evaluates speech generation using a speech language model (ICASSP 2023). Audio is
quantized to discrete units (k-means clustering over self-supervised features), then scored as
the average negative log-likelihood per token under an LSTM language model trained on real speech.
Lower is better (lower NLL indicates the audio is more predictable/"speech-like" under the LM).

### utmos_v2
UTMOSv2 is a learned MOS predictor for naturalness (successor to UTMOS, the UTokyo-SaruLab
system). Scores are on an approximate 1-5 MOS scale.
Higher is better.

## Intelligibility and Signal-Level Metrics

### squim_pesq
SQUIM PESQ is a non-intrusive estimate of perceptual speech quality.
Higher is better.

### squim_stoi
SQUIM STOI is a non-intrusive estimate of speech intelligibility.
Higher is better.

### squim_si_sdr
SQUIM SI-SDR is a non-intrusive estimate of source-to-distortion quality.
Higher is better.

### squim_mos (optional)
SQUIM MOS predicts a subjective mean opinion score.
It is currently available in code but may be disabled in the active metric list.
Higher is better.

### brouhaha
Brouhaha runs a pretrained regression pipeline and reports frame-level speech condition estimates.
In this setup, it returns the mean SNR prediction over the audio file.
Higher SNR is generally better.

## Speaker Similarity

### speaker_similarity
Speaker similarity is computed with Wespeaker (ResNet221-LM, trained on VoxCeleb) by comparing
each file's speaker embedding against a reference clip via cosine similarity.
It is a practical proxy for how consistently the system preserves speaker identity.
Higher is better.

## Distributional Fidelity

### ttsds
TTSDS (TTS Distribution Score, v2) measures how closely the feature distributions of synthetic
speech match those of natural (reference) speech, aggregating several sub-benchmarks (e.g.
hubert, wav2vec2, wavlm, pitch, dvector, wada_snr) into an overall score.
Higher is better (closer to the natural-speech feature distribution).

## Low-Level Acoustic Descriptors

### low_level_dc_offset
DC offset is the mean waveform value.
It indicates whether the signal is centered around zero.
Values closer to 0 are better.

### low_level_lufs
LUFS estimates integrated loudness (perceived loudness over time).
If pyloudnorm is unavailable, a dBFS-like fallback is used.
The "best" value depends on your loudness target rather than always maximizing or minimizing;
in the paper, proximity to the gold reference's LUFS is the relevant reading.

### low_level_peak_clips
Peak clips is the fraction of samples whose absolute amplitude exceeds a clipping threshold.
It estimates clipping risk and distortion.
Lower is better.

### low_level_sibilance_ratio
Sibilance ratio is the energy ratio between high-frequency sibilant band (about 4-10 kHz)
and a lower reference speech band (about 0.3-4 kHz).
Higher values often indicate stronger sibilance.
The ideal range is task-dependent; in the paper, proximity to the gold reference is the
relevant reading.
