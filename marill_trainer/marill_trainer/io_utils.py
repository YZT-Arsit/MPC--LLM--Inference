from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML config to be a mapping, got {type(data).__name__}")
    return data


def merge_cli_overrides(
    base_cfg: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Apply shallow CLI overrides to the top-level config mapping."""
    merged = copy.deepcopy(dict(base_cfg))
    if not overrides:
        return merged

    for key, value in overrides.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def resolve_run_name(cfg: Mapping[str, Any]) -> str:
    """Resolve a stable run name from config with a UTC timestamp fallback."""
    run_cfg = _as_dict(cfg.get("run"))
    explicit_name = run_cfg.get("run_name")
    if explicit_name:
        return str(explicit_name)

    model_cfg = _as_dict(cfg.get("model"))
    model_name = model_cfg.get("model_name") or model_cfg.get("model_path") or "train"
    model_stem = Path(str(model_name)).name.replace("/", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{model_stem}_{timestamp}"


def prepare_output_dirs(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """
    Create a normalized output directory layout for one training run.

    Returns a dict with string paths so downstream code can pass them directly
    into TrainingArguments and JSON metadata.
    """
    run_name = resolve_run_name(cfg)
    run_cfg = _as_dict(cfg.get("run"))
    paths_cfg = _as_dict(cfg.get("paths"))

    output_root = (
        run_cfg.get("output_root")
        or paths_cfg.get("output_root")
        or "outputs"
    )
    root_path = Path(str(output_root)).expanduser().resolve()
    output_dir = root_path / run_name
    log_dir = output_dir / "logs"
    artifact_dir = output_dir / "artifacts"
    checkpoint_dir = output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # TODO: If later we need a dedicated checkpoint subdirectory, confirm that it
    # does not conflict with MarillTrainer's current output_dir handling.
    return {
        "run_name": run_name,
        "output_root": str(root_path),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "artifact_dir": str(artifact_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }


def save_run_metadata(meta: Mapping[str, Any], output_dir: str) -> str:
    """Persist a small JSON metadata snapshot for the current run."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "run_meta.json"

    payload = dict(meta)
    payload.setdefault("saved_at_utc", datetime.now(timezone.utc).isoformat())

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")

    return str(metadata_path)


def build_run_metadata(
    cfg: Mapping[str, Any],
    directories: Mapping[str, Any],
    cli_args: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a small serializable metadata dictionary for later inspection."""
    return {
        "config": copy.deepcopy(dict(cfg)),
        "directories": dict(directories),
        "cli_args": dict(cli_args or {}),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected config section to be a mapping, got {type(value).__name__}")
    return value
