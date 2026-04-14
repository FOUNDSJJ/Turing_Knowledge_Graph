from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EntityMention:
    text: str
    label: str
    start: int
    end: int
    sentence_id: int
    confidence: float = 0.0
    normalized: str | None = None
    source: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EntityNode:
    entity_id: str
    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    mentions: list[dict[str, Any]] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RelationEdge:
    head: str
    tail: str
    relation: str
    sentence_id: int
    evidence: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KnowledgeGraph:
    text: str
    sentences: list[str]
    entities: list[EntityNode]
    relations: list[RelationEdge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sentences": self.sentences,
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": self.metadata,
        }
