import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "e156-submission" / "config.json"
PORTABLE_FILES = [
    REPO_ROOT / "build_html.py",
    REPO_ROOT / "README.md",
]


class PortabilityContracts(unittest.TestCase):
    def test_submission_config_uses_repo_relative_root(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["path"], "..")
        resolved_root = (CONFIG_PATH.parent / payload["path"]).resolve()
        self.assertEqual(resolved_root, REPO_ROOT.resolve())

    def test_release_surface_avoids_hardcoded_metashift_root(self) -> None:
        for path in PORTABLE_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("C:/MetaShift", text, path)
            self.assertNotIn(r"C:\MetaShift", text, path)


if __name__ == "__main__":
    unittest.main()
