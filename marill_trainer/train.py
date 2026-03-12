from __future__ import annotations

import argparse
from typing import Any, Dict

from marill_trainer.io_utils import load_yaml_config
from marill_trainer.train_pipeline import run_training


def parse_args() -> argparse.Namespace:
    """Parse the minimal CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Minimal training entrypoint for marill_trainer")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML training config",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional top-level output root override",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run name override",
    )
    return parser.parse_args()


def build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI args into the config shape expected by merge_cli_overrides."""
    overrides: Dict[str, Any] = {}

    if args.output_root is not None:
        overrides["run"] = {"output_root": args.output_root}
    if args.run_name is not None:
        run_cfg = dict(overrides.get("run", {}))
        run_cfg["run_name"] = args.run_name
        overrides["run"] = run_cfg

    return overrides


def apply_nested_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the small nested override set used by the minimal CLI."""
    merged = dict(cfg)
    for section_name, section_value in overrides.items():
        current = dict(merged.get(section_name, {}))
        current.update(section_value)
        merged[section_name] = current
    return merged


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    nested_overrides = build_cli_overrides(args)
    cfg = apply_nested_overrides(cfg, nested_overrides)

    cli_args = {
        "config": args.config,
        "output_root": args.output_root,
        "run_name": args.run_name,
    }
    run_training(cfg, cli_args=cli_args)


if __name__ == "__main__":
    main()


# TODO: Add resume-from-checkpoint support once the server launch path is stable.
# TODO: If we later need more CLI overrides, switch to a clearer dotted-key merge
# mechanism instead of expanding this minimal nested override helper.
