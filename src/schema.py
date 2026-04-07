from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any


@dataclass
class Entity:
    name: str
    entity_type: str
    normalized_name: str = ""
    kb_id: str = ""
    description: str = ""
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)