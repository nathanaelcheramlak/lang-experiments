#!/usr/bin/env bash
set -euo pipefail

root=$(dirname "$(realpath "$0")")
upstream="$root/upstream-sagalee"
wenet="$upstream/wenet"
recipe="$wenet/examples/aishell/whisper"
raw="$root/data/raw/sagalee/openslr-157"
data="$root/data/prepared/sagalee/openslr-157-v1"
checkpoint="$root/models/openai-whisper/large-v3.pt"
ds_config="$root/configs/deepspeed-3090.json"
run_id=${RUN_ID:-sagalee-large-v3}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  printf 'Invalid RUN_ID: %s\n' "$run_id" >&2
  exit 2
fi
run="$root/runs/$run_id"
initial="$run/init/wenet_whisper.remove-subsample.init-ctc.pt"
model="$run/model"
epochs=15
train_list="$data/train/data.list"
dev_list="$data/dev/data.list"

case ${1:-} in
  prepare|smoke|train|evaluate) ;;
  *) printf 'Usage: bash baseline.sh prepare [split-root] | {smoke|train|evaluate}\n' >&2; exit 2 ;;
esac

export PYTHONPATH="$wenet${PYTHONPATH:+:$PYTHONPATH}"

case $1 in
  prepare)
    source_dir=${2:-"$raw/extracted/sagalee"}
    python "$root/scripts/prepare_sagalee.py" "$source_dir" "$data"
    if [[ $# -eq 1 && -f "$raw/archive.sha256" ]]; then
      cp "$raw/archive.sha256" "$data/source-archive.sha256"
    fi
    ;;
  smoke|train)
    test -s "$train_list" && test -s "$dev_list" && test -s "$data/test/data.list" && \
      test -s "$checkpoint" || {
      printf 'Run fetch and prepare first (see README.md).\n' >&2
      exit 1
    }
    python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "A BF16-capable CUDA GPU is required")'
    if [[ $1 == smoke ]]; then
      run="$run/smoke"
      model="$run/model"
      epochs=1
      mkdir -p "$run"
      train_list="$run/train.list"
      dev_list="$run/dev.list"
      head -n 16 "$data/train/data.list" > "$train_list"
      head -n 8 "$data/dev/data.list" > "$dev_list"
    fi
    if [[ -e "$model" ]]; then
      printf 'Refusing to overwrite an existing training run: %s\n' "$model" >&2
      exit 1
    fi
    if [[ ! -s "$initial" ]]; then
      init_dir=$(dirname "$initial")
      mkdir -p "$init_dir"
      python "$wenet/wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py" \
        --whisper_ckpt "$checkpoint" --output_dir "$init_dir"
      python "$recipe/local/modify_ckpt.py" --remove_list encoder.embed.conv \
        --add_list '{"ctc.ctc_lo.weight": "decoder.embed.0.weight"}' \
        --input_ckpt "$init_dir/wenet_whisper.pt" --output_ckpt "$initial.tmp"
      mv "$initial.tmp" "$initial"
    fi

    # The pinned loader requires input_dim for Whisper but the sample YAML omits it.
    python - "$recipe/conf/finetune_whisper_largev3_conv2d4.yaml" "$run/config.yaml" "$epochs" <<'PY'
import sys
import yaml

source, target, epochs = sys.argv[1:]
with open(source, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
config["input_dim"] = config["dataset_conf"]["log_mel_spectrogram_conf"]["num_mel_bins"]
config["max_epoch"] = int(epochs)
config["dataset_conf"]["batch_conf"].update(batch_type="static", batch_size=2)
with open(target, "w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY
    cp "$ds_config" "$run/deepspeed.json"
    sha256sum "$train_list" "$dev_list" "$data/test/data.list" "$data/test/text" > "$run/manifests.sha256"
    if [[ -f "$data/source-archive.sha256" ]]; then
      cp "$data/source-archive.sha256" "$run/source-archive.sha256"
    fi
    git -C "$wenet" rev-parse HEAD > "$run/wenet-revision.txt"
    python -m pip freeze > "$run/packages.txt"
    mkdir -p "$model"
    cd "$recipe"
    torchrun --standalone --nproc_per_node=1 "$wenet/wenet/bin/train.py" \
      --train_engine deepspeed --deepspeed_config "$run/deepspeed.json" \
      --deepspeed.save_states model_only --ddp.dist_backend nccl \
      --config "$run/config.yaml" \
      --data_type raw --train_data "$train_list" --cv_data "$dev_list" \
      --checkpoint "$initial" --model_dir "$model" --tensorboard_dir "$run/tensorboard" \
      --num_workers 2 --prefetch 2 --pin_memory 2>&1 | tee "$run/train.log"
    ;;
  evaluate)
    for epoch in 12 13 14; do
      test -s "$model/epoch_$epoch.pt" || {
        printf 'Missing epoch %s checkpoint in %s\n' "$epoch" "$model" >&2
        exit 1
      }
    done
    sha256sum --check "$run/manifests.sha256"
    cd "$recipe"
    # This filter selects exactly the final three epochs, not the three best epochs.
    python "$wenet/wenet/bin/average_model.py" --src_path "$model" --dst_model "$run/avg-last3.pt" \
      --mode epoch --min_epoch 12 --max_epoch 14 --num 3 --val_best
    python "$wenet/wenet/bin/recognize.py" --gpu 0 --modes attention \
      --config "$model/train.yaml" --data_type raw --test_data "$data/test/data.list" \
      --checkpoint "$run/avg-last3.pt" --beam_size 10 --batch_size 4 --result_dir "$run/test"
    ref="$data/test/text"
    hyp="$run/test/attention/text"
    # WeNet's scorer silently skips missing hypotheses; compare IDs before scoring.
    LC_ALL=C sort -u -c <(cut -f 1 "$ref" | LC_ALL=C sort)
    diff -u <(cut -f 1 "$ref" | LC_ALL=C sort) <(cut -d ' ' -f 1 "$hyp" | LC_ALL=C sort)
    python "$wenet/tools/compute-wer.py" --char=0 --v=0 "$ref" "$hyp" | tee "$run/test.wer"
    ;;
esac
