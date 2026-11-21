"""Convert the official Sagalee splits into WeNet raw manifests."""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

SPLITS = ("train", "dev", "test")


def normalize(text):
    # Match turinaf/sagalee's prepare_wenet_data.py. Fold all whitespace to keep
    # the reference for each utterance on a single line.
    for original, replacement in (
        ("\u00e8", "e"), ("\u2092", ""), ("\u2022", ""), ("\u02bc", ""),
        ("''", ""), ("_", " "), ("\xa0", " "),
    ):
        text = text.replace(original, replacement)
    text = re.sub(r"(?!\b'\b)(?<!\d)\.(?!\d)|[^\w\s'.]", "", text)
    text = text.replace(" '", " ").replace("' ", " ").strip("'")
    return " ".join(text.upper().split())


def prepare(source, output):
    source = source.resolve()
    if output.exists():
        raise ValueError(f"{output} already exists; keep or move it before preparing again")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, keys = {}, set()
    with TemporaryDirectory(dir=output.parent, prefix=".prepare-") as temporary:
        staging = Path(temporary) / "manifests"
        for split in SPLITS:
            clips = sorted((source / split).rglob("*.wav"))
            if not clips:
                raise ValueError(f"No WAV files found under {source / split}")
            target = staging / split
            target.mkdir(parents=True)
            with (target / "data.list").open("w", encoding="utf-8") as manifest, \
                    (target / "text").open("w", encoding="utf-8") as references:
                for wav in clips:
                    key = wav.stem
                    if key in keys or any(char.isspace() for char in key):
                        raise ValueError(f"Duplicate or invalid utterance ID: {wav}")
                    keys.add(key)
                    text = normalize(wav.with_suffix(".txt").read_text(encoding="utf-8"))
                    if not text:
                        raise ValueError(f"Empty normalized transcript: {wav}")
                    record = {"key": key, "wav": str(wav), "txt": text}
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    references.write(f"{key}\t{text}\n")
            counts[split] = len(clips)
        staging.rename(output)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory containing train, dev and test")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        for split, count in prepare(args.source, args.output).items():
            print(f"{split}: {count} utterances")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
