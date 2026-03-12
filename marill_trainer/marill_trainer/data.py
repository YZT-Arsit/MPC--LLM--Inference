from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from torch.utils.data import Dataset


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    jsonl_path = Path(path).expanduser().resolve()
    records: List[Dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx} of {jsonl_path}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected each JSONL line to be an object, got {type(record).__name__} on line {line_idx}"
                )
            records.append(record)

    return records


def format_record(record: Mapping[str, Any], template_type: str = "text") -> str:
    """
    Convert one raw record into a single training string.

    Supported minimal formats:
    - template_type="text": {"text": "..."}
    - template_type="instruction": {"instruction": "...", "input": "...", "output": "..."}
    """
    if template_type == "text":
        text = record.get("text")
        if not text:
            raise ValueError("template_type='text' expects a non-empty 'text' field")
        return str(text)

    if template_type == "instruction":
        instruction = str(record.get("instruction", "")).strip()
        input_text = str(record.get("input", "")).strip()
        output_text = str(record.get("output", "")).strip()

        if not instruction and not output_text:
            raise ValueError(
                "template_type='instruction' expects at least 'instruction' and/or 'output'"
            )

        parts: List[str] = []
        if instruction:
            parts.append(f"Instruction:\n{instruction}")
        if input_text:
            parts.append(f"Input:\n{input_text}")
        if output_text:
            parts.append(f"Response:\n{output_text}")
        return "\n\n".join(parts)

    raise ValueError(f"Unsupported template_type: {template_type}")


def tokenize_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    model_max_length: int,
    template_type: str = "text",
) -> Dict[str, List[int]]:
    """Format and tokenize a record into minimal model inputs."""
    text = format_record(record, template_type=template_type)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=model_max_length,
        padding=False,
        return_attention_mask=True,
    )

    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("Tokenizer output must contain 'input_ids' and 'attention_mask'")

    return {
        "input_ids": list(input_ids),
        "attention_mask": list(attention_mask),
    }


class JsonlCausalLMDataset(Dataset):
    """Minimal in-memory dataset for causal language model training."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: Any,
        model_max_length: int,
        template_type: str = "text",
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.template_type = template_type

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return tokenize_record(
            self.records[index],
            tokenizer=self.tokenizer,
            model_max_length=self.model_max_length,
            template_type=self.template_type,
        )


def build_dataset(
    data_path: str,
    tokenizer: Any,
    model_max_length: int,
    template_type: str = "text",
) -> JsonlCausalLMDataset:
    """Build a minimal dataset from a JSONL file."""
    records = load_jsonl(data_path)
    return JsonlCausalLMDataset(
        records=records,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        template_type=template_type,
    )


def build_train_dataset(cfg: Mapping[str, Any], tokenizer: Any) -> Dataset:
    """Build the training dataset from config."""
    data_cfg = _as_dict(cfg.get("data"))
    training_cfg = _as_dict(cfg.get("training"))
    train_path = data_cfg.get("train_path")
    if not train_path:
        raise ValueError("Config must provide data.train_path")

    template_type = str(data_cfg.get("template_type", "text"))
    model_max_length = int(training_cfg.get("model_max_length", 2048))
    return build_dataset(
        data_path=str(train_path),
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        template_type=template_type,
    )


def build_eval_dataset(cfg: Mapping[str, Any], tokenizer: Any) -> Optional[Dataset]:
    """Build the eval dataset from config, or return None if not configured."""
    data_cfg = _as_dict(cfg.get("data"))
    training_cfg = _as_dict(cfg.get("training"))
    eval_path = data_cfg.get("eval_path")
    if not eval_path:
        return None

    template_type = str(data_cfg.get("template_type", "text"))
    model_max_length = int(training_cfg.get("model_max_length", 2048))
    return build_dataset(
        data_path=str(eval_path),
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        template_type=template_type,
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected config section to be a mapping, got {type(value).__name__}")
    return value


# TODO: If later we confirm the original training format, replace the current
# text/instruction templates with the paper-specific preprocessing logic.
# TODO: For supervised fine-tuning, we may want to mask prompt tokens and only
# train on response tokens. The current minimal version trains on all tokens.
