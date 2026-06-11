from __future__ import annotations

import io
import json
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
            self.assertTrue((root / "llm_models.json").exists())
            routing_data = json.loads((root / "llm_models.json").read_text(encoding="utf-8"))
            self.assertNotIn("model_id", routing_data["agents"]["filter"])
            self.assertEqual(routing_data["agents"]["filter"]["request"]["temperature"], 0.3)
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

    def test_doctor_fails_without_llm_models_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            _write_ready_project(root)
            (root / "llm_models.json").unlink()

            code, output = _run_cli("--root", str(root), "doctor")

            self.assertEqual(code, 1)
            self.assertIn("FAIL: llm_models.json missing", output)

    def test_doctor_uses_custom_llm_models_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            _write_ready_project(root, extra_env="LLM_MODELS_FILE=config/llm_models.json")
            (root / "config").mkdir()
            (root / "config/llm_models.json").write_text(
                (root / "llm_models.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "llm_models.json").unlink()

            code, output = _run_cli("--root", str(root), "doctor")

            self.assertEqual(code, 0)
            self.assertIn("OK: required local files exist", output)
            self.assertIn("OK: llm_models.json is readable", output)

    def test_migrate_llm_config_creates_routing_without_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            root.mkdir(parents=True, exist_ok=True)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "LLM_API_KEY=secret-key",
                        "LLM_BASE_URL=https://api.deepseek.com/v1",
                        "LLM_MODEL=deepseek-chat",
                        "LLM_ARBITER_MODEL=deepseek-reasoner",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            code, output = _run_cli("--root", str(root), "migrate-llm-config")

            self.assertEqual(code, 0)
            data = json.loads((root / "llm_models.json").read_text(encoding="utf-8"))
            self.assertEqual(data["connections"]["default"]["api_key_env"], "LLM_API_KEY")
            self.assertNotIn("secret-key", (root / "llm_models.json").read_text(encoding="utf-8"))
            self.assertEqual(data["agents"]["arbiter"]["model_id"], "deepseek-reasoner")
            self.assertIn("created: llm_models.json", output)

    def test_migrate_llm_config_preserves_draft_inheriting_arbiter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            root.mkdir(parents=True, exist_ok=True)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "LLM_API_KEY=secret-key",
                        "LLM_BASE_URL=https://api.deepseek.com/v1",
                        "LLM_MODEL=deepseek-chat",
                        "LLM_ARBITER_API_KEY=arbiter-key",
                        "LLM_ARBITER_BASE_URL=https://open.bigmodel.cn/api/paas/v4",
                        "LLM_ARBITER_MODEL=glm-4-flash",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            code, _output = _run_cli("--root", str(root), "migrate-llm-config")

            self.assertEqual(code, 0)
            data = json.loads((root / "llm_models.json").read_text(encoding="utf-8"))
            self.assertEqual(data["agents"]["arbiter"]["connection"], "arbiter")
            self.assertEqual(data["agents"]["arbiter"]["model_id"], "glm-4-flash")
            self.assertEqual(data["agents"]["draft"]["connection"], "arbiter")
            self.assertEqual(data["agents"]["draft"]["model_id"], "glm-4-flash")

    def test_migrate_llm_config_refuses_to_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            root.mkdir(parents=True, exist_ok=True)
            (root / ".env").write_text(
                "LLM_API_KEY=secret-key\nLLM_BASE_URL=https://api.deepseek.com/v1\nLLM_MODEL=deepseek-chat\n",
                encoding="utf-8",
            )
            (root / "llm_models.json").write_text("{}\n", encoding="utf-8")

            code, output = _run_cli("--root", str(root), "migrate-llm-config")

            self.assertEqual(code, 1)
            self.assertIn("use --force", output)

    def test_doctor_uses_runtime_llm_routing_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            root = Path(tmp)
            _write_ready_project(root)
            data = json.loads((root / "llm_models.json").read_text(encoding="utf-8"))
            data["agents"]["filter"]["request"] = {"extra_body": "bad"}
            (root / "llm_models.json").write_text(json.dumps(data), encoding="utf-8")

            code, output = _run_cli("--root", str(root), "doctor")

            self.assertEqual(code, 1)
            self.assertIn("FAIL: llm_models.json invalid:", output)
            self.assertIn("extra_body", output)

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
                "LLM_DEEPSEEK_API_KEY=real-key",
                "RANKING_NAMES=综合榜-全部",
                extra_env.strip(),
            ]
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "preference.md").write_text("偏好\n", encoding="utf-8")
    (root / "inventory.md").write_text("库存\n", encoding="utf-8")
    (root / "llm_models.json").write_text(
        (
            '{"connections":{"deepseek":{"provider":"openai_compatible","label":"DeepSeek",'
            '"base_url":"https://api.deepseek.com/v1","api_key_env":"LLM_DEEPSEEK_API_KEY"}},'
            '"defaults":{"connection":"deepseek","model_id":"deepseek-chat"},'
            '"agents":{"filter":{},"arbiter":{},"draft":{}}}'
        ),
        encoding="utf-8",
    )
