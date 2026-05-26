"""Command line interface for smzdm-notice."""

from __future__ import annotations

import argparse
import difflib
import importlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from smzdm_notice import __version__


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    os.environ["SMZDM_NOTICE_HOME"] = str(root)
    return args.func(args, root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smzdm-notice", description="SMZDM 好价提醒机器人")
    parser.add_argument("--root", default=".", help="项目配置根目录，默认当前目录")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="启动机器人")
    run.set_defaults(func=_cmd_run)

    doctor = subparsers.add_parser("doctor", help="检查本地配置和依赖")
    doctor.set_defaults(func=_cmd_doctor)

    setup = subparsers.add_parser("setup", help="初始化 .env、个人配置文件和 workspace")
    setup.set_defaults(func=_cmd_setup)

    save_config = subparsers.add_parser("save-config", help="保存 preference.md 和 inventory.md 快照")
    save_config.set_defaults(func=_cmd_save_config)

    diff_config = subparsers.add_parser("diff-config", help="查看配置文件和备份之间的 diff")
    diff_config.add_argument("--list", action="store_true", help="列出所有备份版本")
    diff_config.add_argument(
        "versions", nargs="*", type=int, help="版本号，传 1 个表示 vN vs vN+1，传 2 个表示 vN vs vM"
    )
    diff_config.set_defaults(func=_cmd_diff_config)
    return parser


def _cmd_run(_args: argparse.Namespace, _root: Path) -> int:
    from smzdm_notice import runtime

    runtime.main()
    return 0


def _cmd_setup(_args: argparse.Namespace, root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    workspace_dirs = ["workspace/state", "workspace/logs", "workspace/audit", "workspace/backups"]
    for relative in workspace_dirs:
        (root / relative).mkdir(parents=True, exist_ok=True)

    file_map = {
        ".env": ".env.example",
        "preference.md": "preference.md.template",
        "inventory.md": "inventory.md.template",
    }
    for target_name, source_name in file_map.items():
        if not _ensure_file_from(root / target_name, _template_source(root, source_name)):
            return 1
    return 0


def _template_source(root: Path, source_name: str) -> Path:
    local_source = root / source_name
    if local_source.exists():
        return local_source
    return Path(__file__).resolve().parents[2] / source_name


def _ensure_file_from(target: Path, source: Path) -> bool:
    if target.exists():
        print(f"exists: {target.name}")
        return True
    if not source.exists():
        print(f"error: {source.name} not found, cannot create {target.name}")
        return False
    target.write_text(source.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
    print(f"created: {target.name}")
    return True


def _cmd_save_config(_args: argparse.Namespace, root: Path) -> int:
    backup_dir = root / "workspace/backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for filename in _config_filenames():
        source = root / filename
        if not source.exists():
            print(f"missing: {filename}")
            continue
        target = backup_dir / f"{filename}.{stamp}.bak"
        shutil.copy2(source, target)
        print(f"saved: {target.relative_to(root)}")
    return 0


def _cmd_diff_config(args: argparse.Namespace, root: Path) -> int:
    if args.list:
        _print_backup_versions(root)
        return 0
    if len(args.versions) > 2:
        raise SystemExit("diff-config accepts at most two version numbers")
    if not args.versions:
        _print_current_diff(root)
        return 0
    start = args.versions[0]
    end = args.versions[1] if len(args.versions) == 2 else start + 1
    _print_version_diff(root, start, end)
    return 0


def _config_filenames() -> tuple[str, str]:
    return ("inventory.md", "preference.md")


def _backup_paths(root: Path, filename: str) -> list[Path]:
    return sorted((root / "workspace/backups").glob(f"{filename}.*.bak"))


def _print_backup_versions(root: Path) -> None:
    for filename in _config_filenames():
        backups = _backup_paths(root, filename)
        if not backups:
            print(f"{filename}: no backups")
            continue
        print(f"{filename}:")
        for index, path in enumerate(backups, 1):
            print(f"  [{index}] {_backup_stamp(path, filename)}")


def _backup_stamp(path: Path, filename: str) -> str:
    prefix = f"{filename}."
    suffix = ".bak"
    name = path.name
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return name


def _print_current_diff(root: Path) -> None:
    for filename in _config_filenames():
        backups = _backup_paths(root, filename)
        if not backups:
            print(f"{filename}: no backups")
            continue
        _print_diff(backups[-1], root / filename, f"{backups[-1].name}", filename)


def _print_version_diff(root: Path, start: int, end: int) -> None:
    for filename in _config_filenames():
        backups = _backup_paths(root, filename)
        if not backups:
            print(f"{filename}: no backups")
            continue
        if start < 1 or end < 1 or start > len(backups) or end > len(backups):
            print(f"{filename}: version out of range ({len(backups)} backups)")
            continue
        _print_diff(backups[start - 1], backups[end - 1], f"v{start}", f"v{end}")


def _print_diff(left: Path, right: Path, left_label: str, right_label: str) -> None:
    if not left.exists() or not right.exists():
        print(f"skip diff: {left} or {right} does not exist")
        return
    left_lines = left.read_text(encoding="utf-8").splitlines(keepends=True)
    right_lines = right.read_text(encoding="utf-8").splitlines(keepends=True)
    print(f"=== {left_label} vs {right_label} ===")
    print("".join(difflib.unified_diff(left_lines, right_lines, fromfile=left_label, tofile=right_label)), end="")
    print()


def _cmd_doctor(_args: argparse.Namespace, root: Path) -> int:
    checks = [
        _check_python_version(),
        _check_imports(),
        _check_project_files(root),
        _check_env_file(root),
        _check_workspace(root),
    ]
    ok = True
    for passed, message in checks:
        print(("OK" if passed else "FAIL") + f": {message}")
        ok = ok and passed
    return 0 if ok else 1


def _check_python_version() -> tuple[bool, str]:
    version = sys.version_info
    return version >= (3, 9), f"Python {version.major}.{version.minor}.{version.micro}"


def _check_imports() -> tuple[bool, str]:
    modules = ["httpx", "lark_oapi", "loguru", "openai", "pydantic", "dotenv"]
    missing = [module for module in modules if not _can_import(module)]
    if missing:
        return False, "missing dependencies: " + ", ".join(missing)
    return True, "runtime dependencies importable"


def _can_import(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def _check_project_files(root: Path) -> tuple[bool, str]:
    required = [".env", "preference.md", "inventory.md"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return False, "missing files: " + ", ".join(missing)
    return True, "required local files exist"


def _check_env_file(root: Path) -> tuple[bool, str]:
    env_path = root / ".env"
    if not env_path.exists():
        return False, ".env missing"
    raw = env_path.read_text(encoding="utf-8")
    placeholders = ["cli_xxx", "your-app-secret", "your-api-key"]
    found = [value for value in placeholders if value in raw]
    if found:
        return False, "placeholder values remain in .env"
    values = _parse_env_values(raw)
    required = ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "SMZDM_SIGN_KEY", "SMZDM_USER_AGENT", "LLM_API_KEY"]
    missing = [key for key in required if not values.get(key)]
    if missing:
        return False, "missing required .env values: " + ", ".join(missing)
    return True, ".env has no known placeholders"


def _parse_env_values(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _check_workspace(root: Path) -> tuple[bool, str]:
    workspace = root / "workspace"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        probe = workspace / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"workspace not writable: {exc}"
    return True, "workspace writable"
