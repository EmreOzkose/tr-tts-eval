# tr-tts-eval

**A Comprehensive Objective Evaluation of Modern Text-to-Speech for Turkish Using Speech Quality Assessment Models**

Multi-metric benchmark of contemporary TTS systems (Chatterbox, CosyVoice, OmniVoice, VoxCPM2) and a conventional VITS baseline, evaluated across fine-tuned and zero-shot configurations against a natural-speech (gold) anchor. Eighteen complementary objective metrics span learned naturalness, intelligibility, signal quality, speaker similarity, distributional fidelity, and low-level acoustics, plus utterance-length and long-form temporal-consistency analyses.

📄 Accepted at [SPECOM 2026](https://specom.nw.ru/2026/)

## Running the metrics with Docker

We do not share our evaluation data, but you can run the same metric pipeline on your own audio. You need Docker and the NVIDIA Container Toolkit.

Build the image:

```bash
docker build -t tr-tts-eval .
```

Put your audio in two folders. Each subfolder is one system. The reference folder must use the same subfolder names and contain exactly one reference `.wav` per system (used by `speaker_similarity` and `ttsds`).

```
my_data/
  tts/
    system_a/*.wav
    system_b/*.wav
  refs/
    system_a/ref.wav
    system_b/ref.wav
```

Run all metrics from the paper:

```bash
mkdir -p ~/.cache/tr-tts-eval my_data/results

docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v $(pwd)/my_data:/data \
  -v ~/.cache/tr-tts-eval:/cache \
  tr-tts-eval --data /data/tts --ref /data/refs --out /data/results/results.csv
```

Results are written to `results_average.csv`, `results_detailed.csv` and `results_ttsds_detailed.csv` in `my_data/results`.

To run only some metrics, use `--metrics`, e.g. `--metrics utmos_v2 dnsmos_pro squim_pesq`. Run the image without arguments to see all options. Metric descriptions are in [description.md](description.md).

Notes:

- Model weights are not in the image. They are downloaded on the first run (about 8 GB) into the folder mounted at `/cache`, so keep using the same folder.
- At the moment the WhiSQA checkpoint can not be downloaded from its original repository ([leto19/WhiSQA](https://github.com/leto19/WhiSQA)) because the repo is over its Git LFS quota. The image still builds, but any run that includes `whisqa` (the default metric list does) stops with a missing checkpoint error. Until this is fixed upstream, leave `whisqa` out with `--metrics`, or if you already have `single_head_model.pt`, mount its folder with `-v /path/to/checkpoints:/workspace/methods/WhiSQA/checkpoints`. We will update this when the checkpoint is available again.
- `--user $(id -u):$(id -g)` keeps the output files owned by you instead of root.
