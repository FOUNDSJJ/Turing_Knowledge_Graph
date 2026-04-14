from __future__ import annotations

import math
import re
from collections import defaultdict

from .schema import EntityMention, EntityNode

DEFAULT_ALIAS_TABLE: dict[str, str] = {
    "北大": "北京大学",
    "清华": "清华大学",
    "中科院": "中国科学院",
    "微软": "微软公司",
    "中国科学院计算技术研究所": "中国科学院计算技术研究所",
}

DEFAULT_ENTITY_KB: dict[str, dict[str, str]] = {
    "北京大学": {
        "entity_type": "ORG",
        "description": "中国北京市的一所综合性大学。",
    },
    "清华大学": {
        "entity_type": "ORG",
        "description": "中国北京市的一所研究型大学。",
    },
    "中国科学院": {
        "entity_type": "ORG",
        "description": "中国自然科学最高学术机构。",
    },
    "中国科学院计算技术研究所": {
        "entity_type": "ORG",
        "description": "中国科学院下属科研机构，长期从事计算技术研究。",
    },
    "微软公司": {
        "entity_type": "ORG",
        "description": "全球软件与云服务企业。",
    },
}


def normalize_entity_name(name: str, alias_table: dict[str, str] | None = None) -> str:
    alias_table = alias_table or DEFAULT_ALIAS_TABLE
    compact = re.sub(r"[（(].*?[）)]", "", name).strip()
    compact = re.sub(r"(教授|博士|主任|先生|女士|同学|老师|研究员)$", "", compact)
    return alias_table.get(compact, compact)


class EntityExtender:
    def expand(self, mentions: list[EntityMention], text: str) -> list[EntityMention]:
        enriched: list[EntityMention] = []
        for mention in mentions:
            normalized = normalize_entity_name(mention.text)
            mention.normalized = normalized
            enriched.append(mention)
            if mention.label == "PERSON":
                title_match = re.search(
                    re.escape(mention.text) + r"(教授|博士|主任|老师|研究员)",
                    text[mention.start : mention.start + 12],
                )
                if title_match:
                    mention.normalized = mention.text
        return enriched


class EntityDisambiguator:
    def __init__(
        self,
        knowledge_base: dict[str, dict[str, str]] | None = None,
        alias_table: dict[str, str] | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base or DEFAULT_ENTITY_KB
        self.alias_table = alias_table or DEFAULT_ALIAS_TABLE

    def link(self, mentions: list[EntityMention], sentences: list[str]) -> list[EntityNode]:
        grouped: dict[str, list[EntityMention]] = defaultdict(list)
        for mention in mentions:
            candidate = mention.normalized or normalize_entity_name(mention.text, self.alias_table)
            best_name, best_meta = self._resolve_candidate(candidate, mention, sentences)
            mention.normalized = best_name
            grouped[best_name].append(mention)

        entities: list[EntityNode] = []
        for index, (name, grouped_mentions) in enumerate(grouped.items(), start=1):
            meta = self.knowledge_base.get(name, {})
            entity_type = self._majority_label(grouped_mentions, meta.get("entity_type"))
            aliases = sorted({item.text for item in grouped_mentions if item.text != name})
            confidence = sum(item.confidence for item in grouped_mentions) / len(grouped_mentions)
            entities.append(
                EntityNode(
                    entity_id=f"E{index:03d}",
                    name=name,
                    entity_type=entity_type,
                    aliases=aliases,
                    mentions=[item.to_dict() for item in grouped_mentions],
                    attributes={
                        "mention_count": len(grouped_mentions),
                        "sentence_ids": sorted({item.sentence_id for item in grouped_mentions}),
                    },
                    description=meta.get("description", ""),
                    confidence=round(confidence, 3),
                )
            )
        return entities

    def _resolve_candidate(
        self,
        candidate: str,
        mention: EntityMention,
        sentences: list[str],
    ) -> tuple[str, dict[str, str]]:
        if candidate in self.knowledge_base:
            return candidate, self.knowledge_base[candidate]

        best_score = -math.inf
        best_name = candidate
        best_meta: dict[str, str] = {}
        context = sentences[mention.sentence_id]
        for kb_name, meta in self.knowledge_base.items():
            score = self._string_similarity(candidate, kb_name)
            score += self._context_overlap(context, meta.get("description", "")) * 0.3
            if mention.label == meta.get("entity_type"):
                score += 0.2
            if score > best_score and score >= 0.5:
                best_score = score
                best_name = kb_name
                best_meta = meta
        return best_name, best_meta

    @staticmethod
    def _majority_label(
        mentions: list[EntityMention],
        fallback: str | None = None,
    ) -> str:
        if fallback:
            return fallback
        counts: dict[str, int] = defaultdict(int)
        for mention in mentions:
            counts[mention.label] += 1
        return max(counts, key=counts.get)

    @staticmethod
    def _string_similarity(left: str, right: str) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    @staticmethod
    def _context_overlap(context: str, description: str) -> float:
        context_tokens = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", context))
        desc_tokens = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", description))
        if not context_tokens or not desc_tokens:
            return 0.0
        return len(context_tokens & desc_tokens) / len(context_tokens | desc_tokens)
