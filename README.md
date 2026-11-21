# Sagalee-only Whisper Large-v3 baseline

Target: reproduce the [Sagalee paper](https://arxiv.org/abs/2502.00421)'s
**10.82% test word error rate (WER)** before adding other datasets. This is a
reconstruction from the paper and public WeNet code, not the authors' exact
training environment. The original GPU setup, language prompt, and exact
decoding/scoring settings were not established. Record the measured WER even
if it is different from 10.82%.

## Project layout

| Path | Purpose |
| --- | --- |
| `scripts/` | Downloads and corpus-specific manifest preparation |
| `configs/` | Training configuration tracked in Git |
| `data/raw/<corpus>/<release>/` | Original archive, recorded checksum, and extracted data |
| `data/prepared/<corpus>/<version>/` | Versioned WeNet manifests and scoring references |
| `models/` | Pretrained checkpoints |
| `runs/<run-id>/` | Configs, manifest hashes, logs, checkpoints, and scores |
| `upstream-sagalee/` | Pinned upstream code only |

Data, weights, and run outputs stay on the server and are ignored by Git. For
each additional corpus, add a preparation script and a new manifest directory.
Keep Sagalee's official dev/test manifests fixed. Give each new training mix
and experiment its own name so they can be compared with this baseline.

## Server setup

Run from this repository on the Linux GPU server. Use a separate Python 3.10
environment rather than changing the Jupyter Python 3.11 environment. The
pinned PyTorch wheels use CUDA 12.1. DeepSpeed 0.14.5 accepts the server's
CUDA 12.5 toolkit (`nvcc`) with those wheels; its CUDA 12 minor-version list
does not include the notebook's PyTorch CUDA 12.8 build. Verify the extension
build below rather than bypassing the version check. Budget about 64 GB of
RAM available to the container and 175 GB of free disk for CPU
optimizer offload, the corpus, and checkpoints. CPU offload reduces VRAM use
but can make training substantially slower. Use a repository path without
spaces: the upstream checkpoint tools assume this. The current 30 GB allocation
is suitable for environment setup, downloads, and manifest preparation. Raise
the limit to 64 GB before the GPU smoke test or full run: even the smoke test
allocates the full optimizer and exports a full checkpoint. This is a planning
estimate, not a measured minimum.

The Jupyter server reports host RAM with `free -h`. Check the container's
actual limits and shared memory before training:

```bash
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/cpu.max \
         /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  if [ -r "$f" ]; then printf '%s: %s\n' "$f" "$(<"$f")"; fi
done
df -h /dev/shm
ldconfig -p | grep -i libsox
```

The memory limit is in bytes; `max` means no limit at that cgroup level.
Torchaudio's SoX backend needs the `libsox.so` shared library (on Ubuntu, the
`libsox-dev` package normally provides the unversioned symlink). If the audio
check below fails for that library and you cannot install system packages,
ask the Jupyter administrator to provide it in the image.

```bash
git clone --no-checkout https://github.com/turinaf/sagalee.git upstream-sagalee
git -C upstream-sagalee checkout --detach 6a6996e105de33ff754439c447cde7a547140d08
git -C upstream-sagalee submodule update --init

python3.10 -m venv .venv
source .venv/bin/activate
mkdir -p .venv/tmp .venv/torch-extensions
export TMPDIR="$PWD/.venv/tmp"
export TORCH_EXTENSIONS_DIR="$PWD/.venv/torch-extensions"
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=2
python -m pip install 'torch==2.2.2+cu121' 'torchaudio==2.2.2+cu121' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r upstream-sagalee/wenet/requirements.txt 'numpy==1.26.4' 'deepspeed==0.14.5' ninja 'py7zr>=1,<2'
python -m pip check
PYTHONPATH="$PWD/upstream-sagalee/wenet" python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda, "Toolkit:", CUDA_HOME)
assert CUDA_HOME, "CUDA toolkit not found; verify nvcc is on PATH"
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA is required"

from wenet.dataset import processor
from deepspeed.ops.op_builder import CPUAdamBuilder

CPUAdamBuilder().load()
print("WeNet audio imports and DeepSpeed CPUAdam preflight passed")
PY
```

The first `CPUAdamBuilder().load()` compiles the extension. Keep the same
virtual environment and exported `TMPDIR` and `TORCH_EXTENSIONS_DIR` when
running training from a new terminal. This keeps builds and installs off the
server's 1 GB `/tmp`.

Download the [official OpenSLR release](https://openslr.org/157/) (CC BY-NC
4.0) and the original OpenAI checkpoint on the server:

```bash
bash scripts/fetch.sh sagalee
bash scripts/fetch.sh large-v3
bash baseline.sh prepare
```

Downloads resume from `.part` files. The Sagalee archive hash records what was
downloaded; the OpenAI checkpoint is checked against its published hash.
`prepare` expects `data/raw/sagalee/openslr-157/extracted/sagalee` to contain
`train`, `dev`, and `test` with paired WAV/TXT files. To use a different
extraction path for the same release, run `bash baseline.sh prepare /path/to/split-root`.
`prepare` writes to `data/prepared/sagalee/openslr-157-v1` without modifying the
source or the upstream checkout. Compare the printed utterance counts with the
official mirror (50,008 train, 2,387 dev, 1,178 test); do not reshuffle splits.

## Run

```bash
bash baseline.sh smoke
bash baseline.sh train
bash baseline.sh evaluate
```

- `smoke` runs one short epoch on 16 train and 8 dev items to catch CUDA,
  DeepSpeed CPUAdam, and checkpoint export errors. It never reads test audio
  and writes to a separate directory. It does not establish that all batches fit.
- `train` updates all parameters for 15 epochs, with batch size 2, learning
  rate `1e-5`, a 12,000-step warmup, a randomly initialized convolutional
  frontend, and CTC initialized from the decoder embeddings. The pinned WeNet
  Conv2d recipe supplies settings not spelled out by the paper, including CTC
  weight 0.3 and its default language handling. DeepSpeed uses BF16 and ZeRO-2
  with CPU Adam offload to target a single 24 GB RTX 3090. The DeepSpeed JSON
  microbatch field stays at 1 because WeNet requires it for its custom loader;
  the actual data batch size is 2.
- `evaluate` averages epochs 13-15 (checkpoint indices 12-14), runs attention
  decoding with beam size 10 on the official test split, checks that every
  utterance was decoded, and uses WeNet's case-insensitive **word-level** scorer.

The score and error counts are in `runs/sagalee-large-v3/test.wer`. The run
directory records WeNet's revision, installed packages, data manifest hashes,
and both training configs. Evaluation checks the hashes before scoring. Use
`RUN_ID=sagalee-large-v3-repeat bash baseline.sh train` for a separate run and
pass the same `RUN_ID` to `evaluate`. Runs save model-only epoch checkpoints,
not optimizer state for exact resumption.
