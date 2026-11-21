import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.prepare_sagalee import SPLITS, prepare


class PrepareSagaleeTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "raw"
        self.output = self.root / "prepared"
        for split in SPLITS:
            speaker = self.source / split / "speaker"
            speaker.mkdir(parents=True)
            (speaker / f"{split}.wav").touch()
            (speaker / f"{split}.txt").write_text("Har'a_lafa,\n 1.5.", encoding="utf-8")

    def test_preserves_splits_and_normalizes_text(self):
        self.assertEqual(prepare(self.source, self.output), dict.fromkeys(SPLITS, 1))
        for split in SPLITS:
            directory = self.output / split
            record = json.loads((directory / "data.list").read_text(encoding="utf-8"))
            self.assertEqual(record["key"], split)
            self.assertEqual(Path(record["wav"]), self.source / split / "speaker" / f"{split}.wav")
            self.assertEqual(record["txt"], "HAR'A LAFA 1.5.")
            self.assertEqual((directory / "text").read_text(encoding="utf-8"),
                             f"{split}\tHAR'A LAFA 1.5.\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.source, self.output)

    def test_failed_prepare_does_not_publish_partial_manifests(self):
        (self.source / "test/speaker/test.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_ids_across_splits(self):
        (self.source / "dev/speaker/dev.wav").rename(self.source / "dev/speaker/train.wav")
        (self.source / "dev/speaker/dev.txt").rename(self.source / "dev/speaker/train.txt")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())
