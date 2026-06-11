"""Command line interface for smzdm-notice."""

from __future__ import annotations

import argparse
import difflib
import importlib
import json
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

    # --- Migration: 可随旧配置迁移逻辑一起删除 ---
    migrate_llm = subparsers.add_parser("migrate-llm-config", help="从旧 LLM env 配置生成 llm_models.json")
    migrate_llm.add_argument("--force", action="store_true", help="覆盖已有 llm_models.json")
    migrate_llm.set_defaults(func=_cmd_migrate_llm_config)

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
        "llm_models.json": "llm_models.example.json",
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


def _llm_models_path(root: Path, env_values: dict[str, str]) -> Path:
    path = Path(env_values.get("LLM_MODELS_FILE", "llm_models.json"))
    if not path.is_absolute():
        path = root / path
    return path


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
    env_values = _doctor_env_values(root)
    llm_models_check, llm_models_data = _check_llm_models_file(root, env_values)
    checks = [
        _check_python_version(),
        _check_imports(),
        _check_project_files(root),
        llm_models_check,
        _check_env_file(root, env_values, llm_models_data),
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


def _doctor_env_values(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    if not env_path.exists():
        return {}
    return _parse_env_values(env_path.read_text(encoding="utf-8"))


def _check_llm_models_file(root: Path, env_values: dict[str, str]) -> tuple[tuple[bool, str], dict | None]:
    from smzdm_notice.llm.routing import LLMRoutingError, validate_raw

    llm_models = _llm_models_path(root, env_values)
    if not llm_models.exists():
        return (False, f"{llm_models.name} missing"), None
    try:
        data = json.loads(llm_models.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return (False, f"{llm_models.name} invalid: {exc}"), None
    if not isinstance(data, dict):
        return (False, f"{llm_models.name} must contain a JSON object"), None
    try:
        validate_raw(data, env=env_values)
    except LLMRoutingError as exc:
        return (False, f"{llm_models.name} invalid: {exc}"), data
    return (True, f"{llm_models.name} is readable"), data


def _check_env_file(root: Path, values: dict[str, str], llm_models_data: dict | None) -> tuple[bool, str]:
    env_path = root / ".env"
    if not env_path.exists():
        return False, ".env missing"
    raw = env_path.read_text(encoding="utf-8")
    placeholders = ["cli_xxx", "your-app-secret", "your-api-key"]
    found = [value for value in placeholders if value in raw]
    if found:
        return False, "placeholder values remain in .env"
    required = [
        "FEISHU_APP_ID",
        "FEISHU_APP_SECRET",
        "SMZDM_CLIENT_PLATFORM",
        "SMZDM_APP_VERSION",
        "SMZDM_SIGN_KEY",
        "SMZDM_USER_AGENT",
    ]
    required.extend(_required_llm_key_envs_from_data(llm_models_data or {}))
    missing = [key for key in required if not values.get(key)]
    if missing:
        return False, "missing required .env values: " + ", ".join(missing)
    return True, ".env has no known placeholders"


def _required_llm_key_envs_from_data(data: dict) -> list[str]:
    raw_connections = data.get("connections")
    raw_defaults = data.get("defaults")
    raw_agents = data.get("agents")
    connections: dict = raw_connections if isinstance(raw_connections, dict) else {}
    defaults: dict = raw_defaults if isinstance(raw_defaults, dict) else {}
    agents: dict = raw_agents if isinstance(raw_agents, dict) else {}
    connection_names = {str(defaults.get("connection") or "").strip()}
    for agent_name in ("filter", "arbiter", "draft"):
        agent = agents.get(agent_name)
        if isinstance(agent, dict) and str(agent.get("connection") or "").strip():
            connection_names.add(str(agent.get("connection")).strip())

    keys: list[str] = []
    seen: set[str] = set()
    for name in connection_names:
        conn = connections.get(name)
        if not isinstance(conn, dict):
            continue
        key = str(conn.get("api_key_env") or "").strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


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


# ============================================================================
# 旧 LLM 配置迁移逻辑 — 当所有用户完成迁移后可一键删除此区块
# 对应 subparser: migrate-llm-config (见 _build_parser 末尾)
# ============================================================================


def _cmd_migrate_llm_config(args: argparse.Namespace, root: Path) -> int:
    env_path = root / ".env"
    if not env_path.exists():
        print("error: .env missing")
        return 1
    env_values = _parse_env_values(env_path.read_text(encoding="utf-8"))
    target = _llm_models_path(root, env_values)
    if target.exists() and not args.force:
        print(f"exists: {target.name}; use --force to overwrite")
        return 1

    try:
        migrated = _build_migrated_llm_models(env_values)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"created: {target.relative_to(root) if target.is_relative_to(root) else target}")
    print(_format_migrated_llm_summary(migrated))
    return 0


def _build_migrated_llm_models(values: dict[str, str]) -> dict:
    base_url = _required_env_value(values, "LLM_BASE_URL")
    model_id = _required_env_value(values, "LLM_MODEL")
    _required_env_value(values, "LLM_API_KEY")

    connections: dict[str, dict[str, str]] = {}
    connection_by_spec: dict[tuple[str, str], str] = {}

    def add_connection(name: str, label: str, url: str, key_env: str) -> str:
        spec = (url, key_env)
        if spec in connection_by_spec:
            return connection_by_spec[spec]
        final_name = name
        suffix = 2
        while final_name in connections:
            final_name = f"{name}_{suffix}"
            suffix += 1
        connections[final_name] = {
            "provider": "openai_compatible",
            "label": label,
            "base_url": url,
            "api_key_env": key_env,
        }
        connection_by_spec[spec] = final_name
        return final_name

    main_timeout = _float_env_value(values, "LLM_TIMEOUT_SECONDS", 300.0)
    main_spec = {
        "base_url": base_url,
        "model_id": model_id,
        "api_key_env": "LLM_API_KEY",
        "timeout_seconds": main_timeout,
    }
    arbiter_spec = _legacy_agent_spec(
        values,
        fallback=main_spec,
        base_url_key="LLM_ARBITER_BASE_URL",
        api_key_key="LLM_ARBITER_API_KEY",
        model_key="LLM_ARBITER_MODEL",
        timeout_key="LLM_ARBITER_TIMEOUT_SECONDS",
    )
    draft_spec = _legacy_agent_spec(
        values,
        fallback=arbiter_spec,
        base_url_key="LLM_DRAFT_BASE_URL",
        api_key_key="LLM_DRAFT_API_KEY",
        model_key="LLM_DRAFT_MODEL",
        timeout_key="LLM_DRAFT_TIMEOUT_SECONDS",
    )

    default_connection = add_connection("default", "Migrated default LLM", base_url, "LLM_API_KEY")
    defaults = {
        "connection": default_connection,
        "model_id": model_id,
        "timeout_seconds": main_timeout,
        "max_retries": _int_env_value(values, "LLM_MAX_RETRIES", 2),
        "request": {
            "response_format": {"type": "json_object"},
            "extra_body": {},
        },
    }
    agents: dict[str, dict] = {
        "filter": {"request": {"temperature": 0.3}},
        "arbiter": {"request": {"temperature": 0.0}},
        "draft": {"request": {"temperature": 0.0}},
    }

    _apply_migrated_agent_override(
        agents,
        add_connection,
        agent="arbiter",
        spec=arbiter_spec,
        default_spec=main_spec,
    )
    _apply_migrated_agent_override(
        agents,
        add_connection,
        agent="draft",
        spec=draft_spec,
        default_spec=main_spec,
    )

    return {
        "connections": connections,
        "defaults": defaults,
        "agents": agents,
    }


def _legacy_agent_spec(
    values: dict[str, str],
    fallback: dict,
    base_url_key: str,
    api_key_key: str,
    model_key: str,
    timeout_key: str,
) -> dict:
    return {
        "base_url": values.get(base_url_key) or fallback["base_url"],
        "model_id": values.get(model_key) or fallback["model_id"],
        "api_key_env": api_key_key if values.get(api_key_key) else fallback["api_key_env"],
        "timeout_seconds": _float_env_value(values, timeout_key, float(fallback["timeout_seconds"])),
    }


def _apply_migrated_agent_override(
    agents: dict[str, dict],
    add_connection,
    agent: str,
    spec: dict,
    default_spec: dict,
) -> None:
    if spec["base_url"] != default_spec["base_url"] or spec["api_key_env"] != default_spec["api_key_env"]:
        agents[agent]["connection"] = add_connection(
            agent,
            f"Migrated {agent} LLM",
            spec["base_url"],
            spec["api_key_env"],
        )
    if spec["model_id"] != default_spec["model_id"]:
        agents[agent]["model_id"] = spec["model_id"]
    if spec["timeout_seconds"] != default_spec["timeout_seconds"]:
        agents[agent]["timeout_seconds"] = spec["timeout_seconds"]


def _required_env_value(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} missing in .env")
    return value


def _float_env_value(values: dict[str, str], key: str, default: float) -> float:
    raw = values.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number") from exc


def _int_env_value(values: dict[str, str], key: str, default: int) -> int:
    raw = values.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _format_migrated_llm_summary(data: dict) -> str:
    lines = ["LLM routing:"]
    for name, conn in data.get("connections", {}).items():
        lines.append(f"- connection {name}: {conn.get('base_url')} via {conn.get('api_key_env')}")
    defaults = data.get("defaults", {})
    lines.append(f"- default: {defaults.get('connection')}/{defaults.get('model_id')}")
    for agent, cfg in data.get("agents", {}).items():
        connection = cfg.get("connection") or defaults.get("connection")
        model_id = cfg.get("model_id") or defaults.get("model_id")
        lines.append(f"- {agent}: {connection}/{model_id}")
    return "\n".join(lines)


# ============================================================================
# 迁移区块结束
# ============================================================================
