from __future__ import annotations

import re

from .schema import EntityNode, RelationEdge

RELATION_RULES: list[tuple[str, str]] = [
    (r"{head}[^，。；;\n]{{0,12}}(?:就读于|毕业于|考入){tail}", "studies_at"),
    (r"{head}[^，。；;\n]{{0,12}}(?:任职于|工作于|担任)[^，。；;\n]{{0,8}}{tail}", "works_for"),
    (r"{head}[^，。；;\n]{{0,8}}(?:创立|创建|创办)了?{tail}", "founded"),
    (r"{head}[^，。；;\n]{{0,6}}(?:位于|坐落于){tail}", "located_in"),
    (r"{head}[^，。；;\n]{{0,6}}(?:与|和){tail}[^，。；;\n]{{0,12}}(?:合作|联合)", "cooperates_with"),
    (r"{head}[^，。；;\n]{{0,6}}(?:属于|隶属于){tail}", "belongs_to"),
]

RELATION_CONSTRAINTS: dict[str, tuple[set[str], set[str]]] = {
    "studies_at": ({"PERSON"}, {"ORG"}),
    "works_for": ({"PERSON"}, {"ORG"}),
    "founded": ({"PERSON", "ORG"}, {"ORG"}),
    "located_in": ({"ORG"}, {"LOC"}),
    "cooperates_with": ({"ORG"}, {"ORG"}),
    "belongs_to": ({"ORG"}, {"ORG"}),
}


class RelationExtractor:
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
        for head_variant in head_variants:
            for tail_variant in tail_variants:
                for rule, relation in RELATION_RULES:
                    pattern = rule.format(
                        head=re.escape(head_variant),
                        tail=re.escape(tail_variant),
                    )
                    if re.search(pattern, sentence):
                        if not self._type_matches(relation, head.entity_type, tail.entity_type):
                            continue
                        relations.append(
                            RelationEdge(
                                head=head.entity_id,
                                tail=tail.entity_id,
                                relation=relation,
                                sentence_id=sentence_id,
                                evidence=sentence.strip(),
                                confidence=0.82,
                            )
                        )
        return relations

    @staticmethod
    def _type_matches(relation: str, head_type: str, tail_type: str) -> bool:
        head_allowed, tail_allowed = RELATION_CONSTRAINTS.get(relation, (set(), set()))
        if not head_allowed and not tail_allowed:
            return True
        return head_type in head_allowed and tail_type in tail_allowed

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
