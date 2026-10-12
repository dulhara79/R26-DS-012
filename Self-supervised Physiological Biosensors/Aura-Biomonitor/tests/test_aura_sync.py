"""Offline safety tests for the rolling Aura mirror. Run: python -m unittest discover -s tests -p 'test_aura_sync.py'"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.aura_sync import (
    SOURCE, TARGET, STATE_FILE, SyncError, checked_source_files,
    validate_existing_target, write_snapshot,
)


class MirrorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        (self.source / "ChestStrap_V3").mkdir(parents=True)
        (self.source / "README.md").write_text("Aura documentation\n")
        (self.source / "ChestStrap_V3" / "firmware.ino").write_text("void setup() {}\n")
        self.destination.mkdir()

    def mirror(self):
        write_snapshot(self.source, self.destination, "a" * 40, [])

    def test_initial_import_does_not_touch_unrelated_components(self):
        important = self.destination / "Other-Component" / "model.py"
        important.parent.mkdir()
        important.write_text("keep me")
        self.mirror()
        self.assertEqual(important.read_text(), "keep me")
        self.assertTrue((self.destination / TARGET / "ChestStrap_V3" / "firmware.ino").is_file())
        state = json.loads((self.destination / TARGET / STATE_FILE).read_text())
        self.assertEqual(state["source"], SOURCE)
        self.assertEqual(
            state["files"]["README.md"],
            hashlib.sha256(b"Aura documentation\n").hexdigest(),
        )

    def test_repeat_is_idempotent(self):
        self.mirror()
        first = (self.destination / TARGET / STATE_FILE).read_bytes()
        self.mirror()
        self.assertEqual(first, (self.destination / TARGET / STATE_FILE).read_bytes())

    def test_manual_change_fails_closed(self):
        self.mirror()
        (self.destination / TARGET / "README.md").write_text("independent edit")
        with self.assertRaises(SyncError):
            self.mirror()
        self.assertEqual((self.destination / TARGET / "README.md").read_text(), "independent edit")

    def test_unowned_extra_file_fails_closed(self):
        self.mirror()
        (self.destination / TARGET / "untracked.txt").write_text("do not delete")
        with self.assertRaises(SyncError):
            self.mirror()
        self.assertTrue((self.destination / TARGET / "untracked.txt").exists())

    def test_missing_manifest_fails_closed(self):
        (self.destination / TARGET).mkdir(parents=True)
        (self.destination / TARGET / "important.txt").write_text("unowned")
        with self.assertRaises(SyncError):
            self.mirror()

    def test_upstream_removal_only_removes_owned_file(self):
        self.mirror()
        (self.source / "README.md").write_text("Updated\n")
        old = self.source / "ChestStrap_V3" / "firmware.ino"
        old.unlink()
        (self.source / "ChestStrap_V3" / "new.ino").write_text("updated")
        self.mirror()
        self.assertFalse((self.destination / TARGET / "ChestStrap_V3" / "firmware.ino").exists())
        self.assertTrue((self.destination / TARGET / "ChestStrap_V3" / "new.ino").exists())

    def test_source_symlink_fails_closed(self):
        (self.source / "external").symlink_to(self.source / "README.md")
        with self.assertRaises(SyncError):
            checked_source_files(self.source)

    def test_reserved_source_manifest_is_rejected(self):
        (self.source / STATE_FILE).write_text("{}")
        with self.assertRaises(SyncError):
            self.mirror()


if __name__ == "__main__":
    unittest.main()
