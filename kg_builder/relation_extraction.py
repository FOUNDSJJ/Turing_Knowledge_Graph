from __future__ import annotations

import re
from typing import Any

from .schema import EntityNode, RelationEdge


class RelationExtractor:
    def __init__(self, relation_rules: list[dict[str, Any]] | None = None) -> None:
        self.relation_rules = relation_rules or []

    def extract(
        self,
        sentences: list[str],
        entities: list[EntityNode],
    ) -> list[RelationEdge]:
        relations: list[RelationEdge] = []
        entity_index = self._build_sentence_entity_index(entities)
        for sentence_id, sentence in enumerate(sentences):
            local_entities = entity_index.get(sentence_id, [])
            for head in local_entities:
                for tail in local_entities:
                    if head.entity_id == tail.entity_id:
                        continue
                    relations.extend(
                        self._match_rules(head, tail, sentence, sentence_id)
                    )
        return self._deduplicate(relations)

    @staticmethod
    def _build_sentence_entity_index(
        entities: list[EntityNode],
    ) -> dict[int, list[EntityNode]]:
        index: dict[int, list[EntityNode]] = {}
        for entity in entities:
            for sentence_id in entity.attributes.get("sentence_ids", []):
                index.setdefault(sentence_id, []).append(entity)
        return index

    def _match_rules(
        self,
        head: EntityNode,
        tail: EntityNode,
        sentence: str,
        sentence_id: int,
    ) -> list[RelationEdge]:
        relations: list[RelationEdge] = []
        head_variants = [head.name, *head.aliases]
        tail_variants = [tail.name, *tail.aliases]
        for rule in self.relation_rules:
            relation = str(rule.get("relation", "")).strip()
            template = str(rule.get("pattern", "")).strip()
            if not relation or not template:
                continue
            if not self._type_matches(rule, head.entity_type, tail.entity_type):
                continue
            for head_variant in head_variants:
                for tail_variant in tail_variants:
                    pattern = template.format(
                        head=re.escape(head_variant),
                        tail=re.escape(tail_variant),
                    )
                    if re.search(pattern, sentence):
                        relations.append(
                            RelationEdge(
                                head=head.entity_id,
                                tail=tail.entity_id,
                                relation=relation,
                                sentence_id=sentence_id,
                                evidence=sentence.strip(),
                                confidence=float(rule.get("confidence", 0.82)),
                            )
                        )
        return relations

    @staticmethod
    def _type_matches(rule: dict[str, Any], head_type: str, tail_type: str) -> bool:
        head_allowed = set(rule.get("head_types", []))
        tail_allowed = set(rule.get("tail_types", []))
        if head_allowed and head_type not in head_allowed:
            return False
        if tail_allowed and tail_type not in tail_allowed:
            return False
        return True

    @staticmethod
    def _deduplicate(relations: list[RelationEdge]) -> list[RelationEdge]:
        unique: dict[tuple[str, str, str, int], RelationEdge] = {}
        for relation in relations:
            key = (
                relation.head,
                relation.tail,
                relation.relation,
                relation.sentence_id,
            )
            unique[key] = relation
        return list(unique.values())
