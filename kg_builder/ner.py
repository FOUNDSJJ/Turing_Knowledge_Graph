from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schema import EntityMention

DEFAULT_SENTENCE_SPLIT_PATTERN = r"(?<=[。！？!?；;\n])"
DEFAULT_LABEL_PRIORITY = {
    "ORG": 6,
    "EVENT": 5,
    "PRODUCT": 4,
    "TIME": 3,
    "LOC": 2,
    "PERSON": 1,
}
DEFAULT_LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "FAC": "LOC",
    "DATE": "TIME",
    "TIME": "TIME",
    "PRODUCT": "PRODUCT",
    "PROD": "PRODUCT",
    "EVENT": "EVENT",
}
PUNCTUATION_TO_STRIP = "，。；;,.!?！？、:：()（）[]【】<>《》\"'“”‘’"


def split_sentences(text: str, split_pattern: str = DEFAULT_SENTENCE_SPLIT_PATTERN) -> list[str]:
    sentences = [
        segment.strip()
        for segment in re.split(split_pattern, text)
        if segment.strip()
    ]
    return sentences or [text.strip()]


class RuleEntityRecognizer:
    def __init__(
        self,
        entity_patterns: dict[str, list[str]] | None = None,
        lexicon: dict[str, str] | None = None,
        normalization: dict[str, object] | None = None,
    ) -> None:
        self.entity_patterns = entity_patterns or {}
        self.lexicon = dict(
            sorted((lexicon or {}).items(), key=lambda item: len(item[0]), reverse=True)
        )
        self.normalization = normalization or {}
        self.label_priority = self.normalization.get("label_priority", DEFAULT_LABEL_PRIORITY)
        self.invalid_substrings = {
            label: set(values)
            for label, values in self.normalization.get("invalid_substrings", {}).items()
        }
        self.label_constraints = self.normalization.get("label_constraints", {})
        self.org_suffixes = tuple(self.normalization.get("org_suffixes", []))
        self.org_trim_prefixes = tuple(self.normalization.get("org_trim_prefixes", []))
        self.org_connectors = tuple(self.normalization.get("org_connectors", []))
        self.loc_trim_prefixes = tuple(self.normalization.get("loc_trim_prefixes", []))
        self.sentence_split_pattern = self.normalization.get(
            "sentence_split_pattern",
            DEFAULT_SENTENCE_SPLIT_PATTERN,
        )

    def recognize(self, text: str) -> tuple[list[str], list[EntityMention]]:
        sentences = split_sentences(text, self.sentence_split_pattern)
        mentions: list[EntityMention] = []
        cursor = 0
        for sentence_id, sentence in enumerate(sentences):
            base_offset = text.find(sentence, cursor)
            cursor = base_offset + len(sentence)
            mentions.extend(self._match_patterns(sentence, sentence_id, base_offset))
            mentions.extend(self._match_lexicon(sentence, sentence_id, base_offset))
        return sentences, self.deduplicate_mentions(mentions)

    def deduplicate_mentions(self, mentions: Iterable[EntityMention]) -> list[EntityMention]:
        ordered = sorted(
            mentions,
            key=lambda item: (
                -int(self.label_priority.get(item.label, 0)),
                -(item.end - item.start),
                -item.confidence,
                item.start,
            ),
        )
        deduped: list[EntityMention] = []
        occupied: list[tuple[int, int]] = []
        for mention in ordered:
            span = (mention.start, mention.end)
            if span in occupied:
                continue
            if any(
                max(start, mention.start) < min(end, mention.end)
                for start, end in occupied
            ):
                continue
            occupied.append(span)
            deduped.append(mention)
        return sorted(deduped, key=lambda item: (item.start, item.end, item.label))

    def _match_patterns(
        self,
        sentence: str,
        sentence_id: int,
        base_offset: int,
    ) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        for label, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, sentence):
                    candidates = self._expand_match_candidates(match.group(), label)
                    for text, local_start, local_end in candidates:
                        if len(text) < 2 or not self._is_valid_mention(text, label):
                            continue
                        score = min(0.96, 0.5 + len(text) / 30)
                        mentions.append(
                            EntityMention(
                                text=text,
                                label=label,
                                start=base_offset + match.start() + local_start,
                                end=base_offset + match.start() + local_end,
                                sentence_id=sentence_id,
                                confidence=round(score, 3),
                                source="rule",
                            )
                        )
        return mentions

    def _match_lexicon(
        self,
        sentence: str,
        sentence_id: int,
        base_offset: int,
    ) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        for term, label in self.lexicon.items():
            for match in re.finditer(re.escape(term), sentence):
                mentions.append(
                    EntityMention(
                        text=term,
                        label=label,
                        start=base_offset + match.start(),
                        end=base_offset + match.end(),
                        sentence_id=sentence_id,
                        confidence=0.92,
                        source="lexicon",
                    )
                )
        return mentions

    def _is_valid_mention(self, text: str, label: str) -> bool:
        stripped = self._clean_candidate(text)
        if not stripped:
            return False
        if any(token in stripped for token in self.invalid_substrings.get(label, set())):
            return False

        constraints = self.label_constraints.get(label, {})
        min_length = int(constraints.get("min_length", 2))
        max_length = int(constraints.get("max_length", 64))
        if not min_length <= len(stripped) <= max_length:
            return False

        if label == "ORG" and self.org_suffixes:
            return self._looks_like_org(stripped)
        return True

    def _expand_match_candidates(self, text: str, label: str) -> list[tuple[str, int, int]]:
        cleaned = self._clean_candidate(text)
        if not cleaned:
            return []
        if label == "ORG":
            return self._split_org_candidates(cleaned)
        if label == "LOC":
            return self._trim_location_candidate(cleaned)
        return [(cleaned, 0, len(cleaned))]

    def _split_org_candidates(self, text: str) -> list[tuple[str, int, int]]:
        for prefix in self.org_trim_prefixes:
            index = text.rfind(prefix)
            if index != -1:
                candidate = text[index + len(prefix) :].strip()
                offset = index + len(prefix)
                nested = self._split_org_candidates(candidate)
                if nested:
                    return [
                        (item_text, offset + start, offset + end)
                        for item_text, start, end in nested
                    ]
                return [(candidate, offset, offset + len(candidate))]

        for connector in self.org_connectors:
            if connector not in text:
                continue
            parts = [part.strip() for part in text.split(connector) if part.strip()]
            if len(parts) <= 1:
                continue
            results: list[tuple[str, int, int]] = []
            cursor = 0
            for part in parts:
                start = text.find(part, cursor)
                cursor = start + len(part)
                if self._looks_like_org(part):
                    results.append((part, start, start + len(part)))
            if results:
                return results

        if self._looks_like_org(text):
            return [(text, 0, len(text))]
        return []

    def _trim_location_candidate(self, text: str) -> list[tuple[str, int, int]]:
        for prefix in self.loc_trim_prefixes:
            index = text.rfind(prefix)
            if index != -1:
                candidate = text[index + len(prefix) :].strip()
                start = index + len(prefix)
                return [(candidate, start, start + len(candidate))]
        return [(text, 0, len(text))]

    def _looks_like_org(self, text: str) -> bool:
        if not self.org_suffixes:
            return True
        normalized = text.rstrip(PUNCTUATION_TO_STRIP)
        return normalized.endswith(self.org_suffixes)

    @staticmethod
    def _clean_candidate(text: str) -> str:
        return text.strip().strip(PUNCTUATION_TO_STRIP)


class CRFEntityRecognizer:
    """Optional char-level CRF recognizer."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model = None
        if self.model_path.exists():
            with self.model_path.open("rb") as file:
                self.model = pickle.load(file)

    def is_ready(self) -> bool:
        return self.model is not None

    def recognize(self, text: str) -> tuple[list[str], list[EntityMention]]:
        if self.model is None:
            raise RuntimeError("CRF model not found. Please train the model first.")
        sentences = split_sentences(text)
        mentions: list[EntityMention] = []
        cursor = 0
        for sentence_id, sentence in enumerate(sentences):
            base_offset = text.find(sentence, cursor)
            cursor = base_offset + len(sentence)
            features = [self._char_features(sentence, idx) for idx in range(len(sentence))]
            tags = self.model.predict_single(features)
            mentions.extend(self._decode_tags(sentence, tags, sentence_id, base_offset))
        return sentences, mentions

    @staticmethod
    def _char_features(sentence: str, index: int) -> dict[str, str]:
        char = sentence[index]
        prev_char = sentence[index - 1] if index > 0 else "<BOS>"
        next_char = sentence[index + 1] if index < len(sentence) - 1 else "<EOS>"
        return {
            "char": char,
            "prev_char": prev_char,
            "next_char": next_char,
            "char.isdigit": str(char.isdigit()),
            "char.istitle": str(char.istitle()),
            "bigram_left": prev_char + char,
            "bigram_right": char + next_char,
        }

    @staticmethod
    def _decode_tags(
        sentence: str,
        tags: list[str],
        sentence_id: int,
        base_offset: int,
    ) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        start = None
        label = None
        for idx, tag in enumerate(tags + ["O"]):
            if tag.startswith("B-"):
                if start is not None and label is not None:
                    mentions.append(
                        EntityMention(
                            text=sentence[start:idx],
                            label=label,
                            start=base_offset + start,
                            end=base_offset + idx,
                            sentence_id=sentence_id,
                            confidence=0.85,
                            source="crf",
                        )
                    )
                start = idx
                label = tag[2:]
            elif tag.startswith("I-"):
                continue
            else:
                if start is not None and label is not None:
                    mentions.append(
                        EntityMention(
                            text=sentence[start:idx],
                            label=label,
                            start=base_offset + start,
                            end=base_offset + idx,
                            sentence_id=sentence_id,
                            confidence=0.85,
                            source="crf",
                        )
                    )
                start = None
                label = None
        return mentions


class TransformerAssistant:
    """Optional helper using a Hugging Face token classification model."""

    def __init__(
        self,
        model_name: str | None = None,
        tokenizer_name: str | None = None,
        normalization: dict[str, object] | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.pipeline = None
        self.normalization = normalization or {}
        self.sentence_split_pattern = self.normalization.get(
            "sentence_split_pattern",
            DEFAULT_SENTENCE_SPLIT_PATTERN,
        )
        self.label_map = {
            **DEFAULT_LABEL_MAP,
            **self.normalization.get("transformer_label_map", {}),
        }
        if model_name is None:
            return
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "token-classification",
                model=model_name,
                tokenizer=self.tokenizer_name,
                aggregation_strategy="simple",
            )
        except Exception:
            self.pipeline = None

    def augment(self, text: str) -> list[EntityMention]:
        if self.pipeline is None:
            return []
        sentences = split_sentences(text, self.sentence_split_pattern)
        mentions: list[EntityMention] = []
        cursor = 0
        for sentence_id, sentence in enumerate(sentences):
            base_offset = text.find(sentence, cursor)
            cursor = base_offset + len(sentence)
            for item in self.pipeline(sentence):
                label = self._normalize_model_label(item)
                if label is None:
                    continue
                word = self._clean_predicted_text(item.get("word", ""))
                if len(word) < 2:
                    continue
                mentions.append(
                    EntityMention(
                        text=word,
                        label=label,
                        start=base_offset + int(item.get("start", 0)),
                        end=base_offset + int(item.get("end", len(word))),
                        sentence_id=sentence_id,
                        confidence=round(float(item.get("score", 0.8)), 3),
                        source="transformer",
                    )
                )
        return mentions

    def _normalize_model_label(self, item: dict[str, object]) -> str | None:
        raw_label = str(item.get("entity_group") or item.get("entity") or "")
        normalized = raw_label.replace("B-", "").replace("I-", "").upper()
        if normalized.startswith("LABEL_"):
            return None
        return self.label_map.get(normalized)

    @staticmethod
    def _clean_predicted_text(text: str) -> str:
        cleaned = text.replace("##", "").replace("▁", " ").strip()
        return re.sub(r"\s+", " ", cleaned)


def collect_label_statistics(mentions: Iterable[EntityMention]) -> dict[str, int]:
    counter = Counter(mention.label for mention in mentions)
    return dict(counter)
