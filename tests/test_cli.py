from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from smzdm_notice import cli


class CliTests(unittest.TestCase):
    def test_setup_creates_local_files_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            (root / ".env").write_text("CUSTOM=1\n", encoding="utf-8")

            code, output = _run_cli("--root", str(root), "setup")

            self.assertEqual(code, 0)
            self.assertEqual((root / ".env").read_text(encoding="utf-8"), "CUSTOM=1\n")
            self.assertTrue((root / "preference.md").exists())
            self.assertTrue((root / "inventory.md").exists())
            self.assertTrue((root / "workspace/state").is_dir())
            self.assertIn("exists: .env", output)

    def test_doctor_passes_with_complete_local_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            _write_ready_project(root)

            code, output = _run_cli("--root", str(root), "doctor")

            self.assertEqual(code, 0)
            self.assertIn("OK: required local files exist", output)

    def test_doctor_fails_when_smzdm_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            _write_ready_project(
                root,
                extra_env="SMZDM_CLIENT_PLATFORM=\nSMZDM_APP_VERSION=\nSMZDM_SIGN_KEY=\nSMZDM_USER_AGENT=",
            )

            code, output = _run_cli("--root", str(root), "doctor")

            self.assertEqual(code, 1)
            self.assertIn(
                "FAIL: missing required .env values: "
                "SMZDM_CLIENT_PLATFORM, SMZDM_APP_VERSION, SMZDM_SIGN_KEY, SMZDM_USER_AGENT",
                output,
            )

    def test_save_and_diff_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            _write_ready_project(root)

            save_code, save_output = _run_cli("--root", str(root), "save-config")
            list_code, list_output = _run_cli("--root", str(root), "diff-config", "--list")
            diff_code, diff_output = _run_cli("--root", str(root), "diff-config")

            self.assertEqual(save_code, 0)
            self.assertEqual(list_code, 0)
            self.assertEqual(diff_code, 0)
            self.assertIn("saved: workspace/backups/inventory.md.", save_output)
            self.assertIn("inventory.md:", list_output)
            self.assertIn("=== inventory.md.", diff_output)


def _run_cli(*args: str) -> tuple[int, str]:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = cli.main(list(args))
    return code, stdout.getvalue()


def _write_ready_project(root: Path, extra_env: str = "") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".env").write_text(
        "\n".join(
            [
                "FEISHU_APP_ID=cli_real_app_id",
                "FEISHU_APP_SECRET=real-secret",
                "SMZDM_CLIENT_PLATFORM=iphone",
                "SMZDM_APP_VERSION=11.1.70",
                "SMZDM_SIGN_KEY=real-smzdm-sign-key",
                "SMZDM_USER_AGENT=real-smzdm-user-agent",
                "LLM_API_KEY=real-key",
                "RANKING_NAMES=综合榜-全部",
                extra_env.strip(),
            ]
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "preference.md").write_text("偏好\n", encoding="utf-8")
    (root / "inventory.md").write_text("库存\n", encoding="utf-8")
