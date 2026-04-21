from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data" / "config"

CONFIG_FILE_NAMES = {
    "entity_patterns": "entity_patterns.json",
    "lexicon": "lexicon.json",
    "alias_table": "alias_table.json",
    "knowledge_base": "knowledge_base.json",
    "relation_rules": "relation_rules.json",
    "normalization": "normalization.json",
}


@dataclass(slots=True)
class ResourceConfig:
    entity_patterns: dict[str, list[str]]
    lexicon: dict[str, str]
    alias_table: dict[str, str]
    knowledge_base: dict[str, dict[str, Any]]
    relation_rules: list[dict[str, Any]]
    normalization: dict[str, Any]
    paths: dict[str, Path]


def load_resource_config(
    config_dir: str | Path | None = None,
    entity_patterns_path: str | Path | None = None,
    lexicon_path: str | Path | None = None,
    alias_table_path: str | Path | None = None,
    knowledge_base_path: str | Path | None = None,
    relation_rules_path: str | Path | None = None,
    normalization_path: str | Path | None = None,
) -> ResourceConfig:
    base_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    paths = {
        "entity_patterns": _resolve_path(base_dir, entity_patterns_path, "entity_patterns"),
        "lexicon": _resolve_path(base_dir, lexicon_path, "lexicon"),
        "alias_table": _resolve_path(base_dir, alias_table_path, "alias_table"),
        "knowledge_base": _resolve_path(base_dir, knowledge_base_path, "knowledge_base"),
        "relation_rules": _resolve_path(base_dir, relation_rules_path, "relation_rules"),
        "normalization": _resolve_path(base_dir, normalization_path, "normalization"),
    }

    entity_patterns = _load_json(paths["entity_patterns"], default={})
    lexicon = _load_json(paths["lexicon"], default={})
    alias_table = _load_json(paths["alias_table"], default={})
    knowledge_base = _normalize_knowledge_base(
        _load_json(paths["knowledge_base"], default={})
    )
    relation_rules = _load_json(paths["relation_rules"], default=[])
    normalization = _load_json(paths["normalization"], default={})

    for canonical_name, meta in knowledge_base.items():
        aliases = meta.get("aliases", [])
        for alias in aliases:
            if alias:
                alias_table.setdefault(alias, canonical_name)

    return ResourceConfig(
        entity_patterns=entity_patterns,
        lexicon=lexicon,
        alias_table=alias_table,
        knowledge_base=knowledge_base,
        relation_rules=relation_rules,
        normalization=normalization,
        paths=paths,
    )


def _resolve_path(
    config_dir: Path,
    override_path: str | Path | None,
    key: str,
) -> Path:
    if override_path:
        return Path(override_path)
    return config_dir / CONFIG_FILE_NAMES[key]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_knowledge_base(raw_data: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw_data, dict):
        normalized: dict[str, dict[str, Any]] = {}
        for name, meta in raw_data.items():
            if isinstance(meta, dict):
                normalized[name] = dict(meta)
            else:
                normalized[name] = {"description": str(meta)}
        return normalized

    if isinstance(raw_data, list):
        normalized = {}
        for item in raw_data:
            if not isinstance(item, dict) or "name" not in item:
                continue
            name = str(item["name"])
            meta = {key: value for key, value in item.items() if key != "name"}
            normalized[name] = meta
        return normalized

    return {}
