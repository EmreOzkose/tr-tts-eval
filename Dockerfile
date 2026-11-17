# tr-tts-eval: objective TTS metric pipeline (run.py) in a GPU container.
#
# Build:  docker build -t tr-tts-eval .
# Run:    see the "Docker" section in README.md
#
# Model weights are NOT baked into the image. They are downloaded on first use
# into /cache, which should be mounted as a volume so downloads are reused.

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python-is-python3 \
        git git-lfs wget ca-certificates build-essential \
        ffmpeg libsndfile1 libsox-dev espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---------------------------------------------------------------------------
# Third-party metric code, pinned to the commits used for the paper.
# The metric wrappers expect these under /workspace/methods.
# ---------------------------------------------------------------------------
ARG WHISQA_COMMIT=0ad007e97093aa54c0c8cdac1af3e6ad6305af3f
ARG BROUHAHA_COMMIT=1e175a9dee0f4d7c618653d64796a45405a768e1
ARG TTSDS_COMMIT=2d21a383dd43d4294f066b058ea8408b286bee5b
ARG NATSCORE_COMMIT=099d13eb42d30e148cbf535a16278bd0a55390d2
ARG UTMOSV2_COMMIT=cc2700db57bb83ee13dc31ebe1b868c254e15d09

# LFS is enabled system-wide because ttsds also clones an LFS dataset at runtime.
# Smudge is skipped only inside this RUN so that we can pick which LFS files to fetch.
RUN git lfs install --system \
    && export GIT_LFS_SKIP_SMUDGE=1 \
    # WhiSQA: only the single-head checkpoint is used by run.py. Upstream LFS
    # downloads can be unavailable (quota); in that case mount the checkpoint
    # at runtime (see README) instead of failing the build.
    && git clone https://github.com/leto19/WhiSQA.git methods/WhiSQA \
    && git -C methods/WhiSQA checkout ${WHISQA_COMMIT} \
    && (git -C methods/WhiSQA lfs pull --include "checkpoints/single_head_model.pt" \
        || echo "WARNING: WhiSQA checkpoint download failed; mount it at runtime to use the whisqa metric.") \
    # WhiSQA loads mel filters from a cwd-relative path; make it absolute.
    && sed -i 's#np.load("models/mel_filters.npz"#np.load("/workspace/methods/WhiSQA/models/mel_filters.npz"#' \
           methods/WhiSQA/models/whisper_wrapper.py \
    && grep -q '/workspace/methods/WhiSQA/models/mel_filters.npz' methods/WhiSQA/models/whisper_wrapper.py \
    # Brouhaha: checkpoint is a regular git file. Drop its numpy==1.24.4 pin.
    && git clone https://github.com/marianne-m/brouhaha-vad.git methods/brouhaha-vad \
    && git -C methods/brouhaha-vad checkout ${BROUHAHA_COMMIT} \
    && sed -i 's/^numpy==1.24.4/# numpy==1.24.4/' methods/brouhaha-vad/requirements.txt \
    # TTSDS: reference/noise distributions are stored with Git LFS.
    && git clone https://github.com/ttsds/ttsds.git methods/ttsds \
    && git -C methods/ttsds checkout ${TTSDS_COMMIT} \
    && git -C methods/ttsds lfs pull \
    # NatScore (optional metric).
    && git clone https://github.com/harrrshall/natscore.git methods/natscore \
    && git -C methods/natscore checkout ${NATSCORE_COMMIT} \
    && rm -rf methods/*/.git

# ---------------------------------------------------------------------------
# Python dependencies. torch is pinned (CUDA 12.4 build) and constrained so
# that no other package can silently upgrade it.
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
           --index-url https://download.pytorch.org/whl/cu124 \
    && printf "torch==2.5.1\ntorchaudio==2.5.1\ntorchvision==0.20.1\n" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN pip install \
        -e methods/ttsds \
        -e methods/natscore \
        -e methods/brouhaha-vad \
        "pyannote.audio==3.3.0" \
        "audiobox_aesthetics==0.0.4" \
        scoreq \
        "git+https://github.com/sarulab-speech/UTMOSv2.git@${UTMOSV2_COMMIT}" \
        "scikit-learn==1.7.2" \
        # transformers 5.x requires torch>=2.6 and silently disables torch 2.5.
        "transformers>=4.46,<5" \
        pyloudnorm librosa soundfile pandas tqdm tabulate torchinfo pyyaml

# ---------------------------------------------------------------------------
# Pipeline code
# ---------------------------------------------------------------------------
COPY . /workspace/tr-tts-eval
WORKDIR /workspace/tr-tts-eval

# run.py also resolves some paths relative to the working directory
# (methods/WhiSQA/checkpoints, methods/ttsds/src), and a few metrics cache
# weights under <repo>/model_cache. Point both to the right places.
# allosaurus (used by ttsds) downloads its model into its own package folder,
# which is not writable when running as a non-root user; redirect it to /cache.
RUN ln -s /workspace/methods methods \
    && ln -s /cache/model_cache model_cache \
    && ln -s /cache/allosaurus "$(python -c 'import allosaurus, os; print(os.path.dirname(allosaurus.__file__))')/pretrained" \
    && mkdir -p /cache && chmod 777 /cache

# Every download (Hugging Face, torch hub, ttsds, wespeaker, ...) goes under /cache.
ENV HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

ENTRYPOINT ["sh", "-c", "mkdir -p /cache/model_cache /cache/allosaurus && exec python run.py \"$@\"", "--"]
CMD ["--help"]
