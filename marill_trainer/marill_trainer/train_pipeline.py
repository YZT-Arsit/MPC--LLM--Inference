from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Mapping, Optional

from transformers import AutoTokenizer

from marill_trainer.collators import build_data_collator
from marill_trainer.data import build_eval_dataset, build_train_dataset
from marill_trainer.io_utils import (
    build_run_metadata,
    prepare_output_dirs,
    save_run_metadata,
)
from marill_trainer.trainer import MarillTrainer, MarillTrainingArguments


def build_tokenizer(cfg: Mapping[str, Any]) -> Any:
    """Build the tokenizer from config."""
    model_cfg = _as_dict(cfg.get("model"))
    training_cfg = _as_dict(cfg.get("training"))

    tokenizer_name = (
        model_cfg.get("tokenizer_name")
        or model_cfg.get("model_path")
        or model_cfg.get("model_name")
    )
    if not tokenizer_name:
        raise ValueError("Config must provide model.tokenizer_name or model.model_path")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_name),
        use_fast=bool(model_cfg.get("use_fast_tokenizer", True)),
        cache_dir=model_cfg.get("cache_dir"),
        model_max_length=int(training_cfg.get("model_max_length", 2048)),
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO: Some models may require special tokenizer settings on the server
    # side, especially around padding side and chat template handling.
    return tokenizer


def build_training_args(cfg: Mapping[str, Any], output_dir: str) -> MarillTrainingArguments:
    """Convert config mappings into MarillTrainingArguments."""
    model_cfg = _as_dict(cfg.get("model"))
    training_cfg = _as_dict(cfg.get("training"))
    marill_cfg = _as_dict(cfg.get("marill"))
    logging_cfg = _as_dict(cfg.get("logging"))
    paths_cfg = _as_dict(cfg.get("paths"))

    field_names = {field.name for field in fields(MarillTrainingArguments)}
    raw_values: Dict[str, Any] = {}
    for section in (training_cfg, marill_cfg, logging_cfg):
        raw_values.update(section)

    raw_values["output_dir"] = output_dir
    raw_values["cache_dir"] = model_cfg.get("cache_dir", paths_cfg.get("cache_dir"))

    teacher_model_path = paths_cfg.get("teacher_model_path")
    if teacher_model_path and "teacher_model" in field_names:
        raw_values["teacher_model"] = teacher_model_path

    argument_values = {key: value for key, value in raw_values.items() if key in field_names}

    # Avoid forcing eval in the minimal skeleton when no eval dataset is wired in.
    argument_values.setdefault("evaluation_strategy", "no")
    argument_values.setdefault("save_strategy", "steps")
    argument_values.setdefault("logging_strategy", "steps")
    argument_values.setdefault("report_to", [])

    return MarillTrainingArguments(**argument_values)


def build_trainer(cfg: Mapping[str, Any]) -> MarillTrainer:
    """Build the MarillTrainer and all minimal dependencies around it."""
    directories = prepare_output_dirs(cfg)
    tokenizer = build_tokenizer(cfg)
    train_dataset = build_train_dataset(cfg, tokenizer)
    eval_dataset = build_eval_dataset(cfg, tokenizer)
    data_collator = build_data_collator(tokenizer)
    training_args = build_training_args(cfg, output_dir=directories["output_dir"])

    model_cfg = _as_dict(cfg.get("model"))
    paths_cfg = _as_dict(cfg.get("paths"))
    logging_cfg = _as_dict(cfg.get("logging"))

    model_path = model_cfg.get("model_path") or paths_cfg.get("model_path")
    if not model_path:
        raise ValueError("Config must provide model.model_path or paths.model_path")

    teacher_model_path = paths_cfg.get("teacher_model_path") or str(model_path)
    project_name = str(logging_cfg.get("project_name", "marill"))

    trainer = MarillTrainer(
        model_path=str(model_path),
        teacher_model_path=str(teacher_model_path),
        args=training_args,
        project_name=project_name,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    return trainer


def run_training(cfg: Mapping[str, Any], cli_args: Optional[Mapping[str, Any]] = None) -> MarillTrainer:
    """Run the minimal training flow and return the constructed trainer."""
    directories = prepare_output_dirs(cfg)
    run_metadata = build_run_metadata(cfg, directories, cli_args=cli_args)
    save_run_metadata(run_metadata, directories["output_dir"])

    trainer = build_trainer(cfg)
    trainer.train()
    trainer.finalize()
    return trainer


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected config section to be a mapping, got {type(value).__name__}")
    return value


# TODO: MarillTrainer currently calls torch.distributed.barrier() during init and
# finalize. Single-process execution may fail until the server launch path uses
# torchrun or the trainer is made more defensive.
# TODO: The current skeleton saves metadata before trainer construction. If the
# final resolved TrainingArguments need to be persisted, add a post-build update.
# TODO: Distillation-specific validation is still minimal. If teacher-dependent
# losses are enabled, we should explicitly validate teacher_model_path.
# TODO: Wandb/report_to behavior depends on the server environment. The current
# skeleton defaults to report_to=[] to avoid assuming an external logging setup.
