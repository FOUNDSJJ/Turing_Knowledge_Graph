from __future__ import annotations

import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from .schema import EntityMention, EntityNode

PUNCTUATION_TO_STRIP = "，。；;,.!?！？、:：()（）[]【】<>《》\"'“”‘’"


def normalize_entity_name(
    name: str,
    alias_table: dict[str, str] | None = None,
    normalization: dict[str, Any] | None = None,
) -> str:
    alias_table = alias_table or {}
    normalization = normalization or {}
    compact = name.strip().strip(PUNCTUATION_TO_STRIP)

    parenthetical_pattern = normalization.get("strip_parenthetical_pattern")
    if parenthetical_pattern:
        compact = re.sub(str(parenthetical_pattern), "", compact).strip()

    strip_suffix_tokens = sorted(
        normalization.get("strip_suffix_tokens", []),
        key=len,
        reverse=True,
    )
    for token in strip_suffix_tokens:
        if compact.endswith(token):
            compact = compact[: -len(token)].strip()
            break

    return alias_table.get(compact, compact)


class EntityExtender:
    def __init__(
        self,
        alias_table: dict[str, str] | None = None,
        normalization: dict[str, Any] | None = None,
    ) -> None:
        self.alias_table = alias_table or {}
        self.normalization = normalization or {}
        self.title_suffixes = tuple(self.normalization.get("title_suffixes", []))
        self.title_window = int(self.normalization.get("title_window", 16))

    def expand(self, mentions: list[EntityMention], text: str) -> list[EntityMention]:
        enriched: list[EntityMention] = []
        for mention in mentions:
            mention.normalized = normalize_entity_name(
                mention.text,
                self.alias_table,
                self.normalization,
            )
            if mention.label == "PERSON" and self.title_suffixes:
                context_window = text[mention.start : mention.start + self.title_window]
                title_pattern = "|".join(re.escape(token) for token in self.title_suffixes)
                if re.search(re.escape(mention.text) + rf"(?:{title_pattern})", context_window):
                    mention.normalized = normalize_entity_name(
                        mention.text,
                        self.alias_table,
                        self.normalization,
                    )
            enriched.append(mention)
        return enriched


class EntityDisambiguator:
    def __init__(
        self,
        knowledge_base: dict[str, dict[str, Any]] | None = None,
        alias_table: dict[str, str] | None = None,
        normalization: dict[str, Any] | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base or {}
        self.alias_table = alias_table or {}
        self.normalization = normalization or {}
        self.link_threshold = float(self.normalization.get("link_threshold", 0.55))

    def link(self, mentions: list[EntityMention], sentences: list[str]) -> list[EntityNode]:
        grouped: dict[str, list[EntityMention]] = defaultdict(list)
        for mention in mentions:
            candidate = mention.normalized or normalize_entity_name(
                mention.text,
                self.alias_table,
                self.normalization,
            )
            best_name, best_meta = self._resolve_candidate(candidate, mention, sentences)
            mention.normalized = best_name
            grouped[best_name].append(mention)

        entities: list[EntityNode] = []
        for index, (name, grouped_mentions) in enumerate(grouped.items(), start=1):
            meta = self.knowledge_base.get(name, {})
            entity_type = self._majority_label(grouped_mentions, meta.get("entity_type"))
            aliases = sorted(
                {
                    alias
                    for alias in meta.get("aliases", [])
                    if alias and alias != name
                }
                | {
                    item.text
                    for item in grouped_mentions
                    if item.text != name
                }
            )
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
                    description=str(meta.get("description", "")),
                    confidence=round(confidence, 3),
                )
            )
        return entities

    def _resolve_candidate(
        self,
        candidate: str,
        mention: EntityMention,
        sentences: list[str],
    ) -> tuple[str, dict[str, Any]]:
        candidate = self.alias_table.get(candidate, candidate)
        if candidate in self.knowledge_base:
            return candidate, self.knowledge_base[candidate]

        best_score = -math.inf
        best_name = candidate
        best_meta: dict[str, Any] = {}
        context = sentences[mention.sentence_id]
        for kb_name, meta in self.knowledge_base.items():
            score = self._string_similarity(candidate, kb_name)
            score += self._context_overlap(context, str(meta.get("description", ""))) * 0.3
            if mention.label == meta.get("entity_type"):
                score += 0.2
            if score > best_score and score >= self.link_threshold:
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
        left_lower = left.lower()
        right_lower = right.lower()

        left_chars = set(left_lower)
        right_chars = set(right_lower)
        char_score = 0.0
        if left_chars and right_chars:
            char_score = len(left_chars & right_chars) / len(left_chars | right_chars)

        left_tokens = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", left_lower))
        right_tokens = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", right_lower))
        token_score = 0.0
        if left_tokens and right_tokens:
            token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

        sequence_score = SequenceMatcher(None, left_lower, right_lower).ratio()
        return max(char_score, (token_score + sequence_score) / 2)

    @staticmethod
    def _context_overlap(context: str, description: str) -> float:
        context_tokens = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", context))
        desc_tokens = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", description))
        if not context_tokens or not desc_tokens:
            return 0.0
        return len(context_tokens & desc_tokens) / len(context_tokens | desc_tokens)
