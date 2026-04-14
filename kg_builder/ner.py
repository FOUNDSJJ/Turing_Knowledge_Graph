from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schema import EntityMention

COMMON_SURNAMES = (
    "张|王|李|赵|刘|陈|杨|黄|周|吴|徐|孙|胡|朱|高|林|何|郭|马|罗|"
    "梁|宋|郑|谢|韩|唐|冯|于|董|萧|程|曹|袁|邓|许|傅|沈|曾|彭|吕|"
    "苏|卢|蒋|蔡|贾|丁|魏|薛|叶|阎|余|潘|杜|戴|夏|钟|汪|田|任|姜"
)

INVALID_PERSON_SUBSTRINGS = {
    "工作",
    "合作",
    "研究",
    "位于",
    "加入",
    "开展",
    "信息",
    "技术",
    "学院",
    "大学",
    "公司",
    "研究所",
    "科学院",
}

DEFAULT_ENTITY_PATTERNS: dict[str, list[str]] = {
    "PERSON": [
        r"[\u4e00-\u9fa5]{2,4}(?:教授|博士|主任|同学|先生|女士)",
        rf"(?:{COMMON_SURNAMES})[\u4e00-\u9fa5]{{1,2}}(?=(?:在|于|与|和|任职|加入|毕业|提出|表示|指出))",
    ],
    "ORG": [
        r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}(?:大学|学院|研究院|研究所|实验室|公司|集团|医院|委员会|学校)",
    ],
    "LOC": [
        r"[\u4e00-\u9fa5]{2,12}(?:省|市|区|县|镇|乡|村|国|洲)",
    ],
    "TIME": [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{4}年\d{1,2}月",
        r"\d{4}年",
    ],
}


def split_sentences(text: str) -> list[str]:
    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[。！？!?；;\n])", text)
        if segment.strip()
    ]
    return sentences or [text.strip()]


class RuleEntityRecognizer:
    def __init__(
        self,
        entity_patterns: dict[str, list[str]] | None = None,
        lexicon: dict[str, str] | None = None,
    ) -> None:
        self.entity_patterns = entity_patterns or DEFAULT_ENTITY_PATTERNS
        self.lexicon = lexicon or {}

    def recognize(self, text: str) -> tuple[list[str], list[EntityMention]]:
        sentences = split_sentences(text)
        mentions: list[EntityMention] = []
        cursor = 0
        for sentence_id, sentence in enumerate(sentences):
            base_offset = text.find(sentence, cursor)
            cursor = base_offset + len(sentence)
            mentions.extend(self._match_patterns(sentence, sentence_id, base_offset))
            mentions.extend(self._match_lexicon(sentence, sentence_id, base_offset))
        return sentences, self._deduplicate_mentions(mentions)

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
                        score = min(0.95, 0.45 + len(text) / 20)
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
            start = sentence.find(term)
            while start != -1:
                mentions.append(
                    EntityMention(
                        text=term,
                        label=label,
                        start=base_offset + start,
                        end=base_offset + start + len(term),
                        sentence_id=sentence_id,
                        confidence=0.92,
                        source="lexicon",
                    )
                )
                start = sentence.find(term, start + len(term))
        return mentions

    @staticmethod
    def _deduplicate_mentions(mentions: Iterable[EntityMention]) -> list[EntityMention]:
        ordered = sorted(
            mentions,
            key=lambda item: (
                -RuleEntityRecognizer._label_priority(item.label),
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
        return deduped

    @staticmethod
    def _label_priority(label: str) -> int:
        priority = {"ORG": 4, "TIME": 3, "LOC": 2, "PERSON": 1}
        return priority.get(label, 0)

    @staticmethod
    def _is_valid_mention(text: str, label: str) -> bool:
        if label == "PERSON":
            if any(token in text for token in INVALID_PERSON_SUBSTRINGS):
                return False
            return 2 <= len(text) <= 6
        if label == "ORG":
            if any(token in text for token in {"在", "于", "并", "加入", "毕业", "合作", "开展"}):
                return False
            return len(text) <= 20
        if label == "LOC":
            if any(token in text for token in {"位于", "坐落", "合作", "开展"}):
                return False
        return True

    @staticmethod
    def _expand_match_candidates(text: str, label: str) -> list[tuple[str, int, int]]:
        cleaned = text.strip("，。；、 ")
        if label == "ORG":
            return RuleEntityRecognizer._split_org_candidates(cleaned)
        if label == "LOC":
            return RuleEntityRecognizer._trim_location_candidate(cleaned)
        return [(cleaned, 0, len(cleaned))]

    @staticmethod
    def _split_org_candidates(text: str) -> list[tuple[str, int, int]]:
        prefixes = ["毕业于", "就读于", "任职于", "工作于", "加入", "在", "于"]
        for prefix in prefixes:
            index = text.rfind(prefix)
            if index != -1:
                candidate = text[index + len(prefix) :]
                offset = index + len(prefix)
                nested = RuleEntityRecognizer._split_org_candidates(candidate)
                if nested:
                    return [
                        (item_text, offset + start, offset + end)
                        for item_text, start, end in nested
                    ]
                return [(candidate, offset, offset + len(candidate))]

        for connector in ["与", "和", "及"]:
            if connector in text:
                results: list[tuple[str, int, int]] = []
                cursor = 0
                for part in text.split(connector):
                    part = part.strip()
                    if not part:
                        cursor += 1
                        continue
                    start = text.find(part, cursor)
                    cursor = start + len(part) + 1
                    if RuleEntityRecognizer._looks_like_org(part):
                        results.append((part, start, start + len(part)))
                if results:
                    return results

        if RuleEntityRecognizer._looks_like_org(text):
            return [(text, 0, len(text))]
        return []

    @staticmethod
    def _trim_location_candidate(text: str) -> list[tuple[str, int, int]]:
        prefixes = ["位于", "坐落于", "在"]
        for prefix in prefixes:
            index = text.rfind(prefix)
            if index != -1:
                candidate = text[index + len(prefix) :]
                return [(candidate, index + len(prefix), index + len(prefix) + len(candidate))]
        return [(text, 0, len(text))]

    @staticmethod
    def _looks_like_org(text: str) -> bool:
        suffixes = ("大学", "学院", "研究院", "研究所", "实验室", "公司", "集团", "医院", "委员会", "学校")
        return text.endswith(suffixes)


class CRFEntityRecognizer:
    """Optional char-level CRF recognizer.

    Requires a pickled sklearn-crfsuite model. If the dependency or model does
    not exist, callers can fall back to RuleEntityRecognizer.
    """

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
    """Optional helper using Hugging Face pipeline when available locally."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name
        self.pipeline = None
        if model_name is None:
            return
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "token-classification",
                model=model_name,
                aggregation_strategy="simple",
            )
        except Exception:
            self.pipeline = None

    def augment(self, text: str) -> list[EntityMention]:
        if self.pipeline is None:
            return []
        sentences = split_sentences(text)
        mentions: list[EntityMention] = []
        cursor = 0
        for sentence_id, sentence in enumerate(sentences):
            base_offset = text.find(sentence, cursor)
            cursor = base_offset + len(sentence)
            for item in self.pipeline(sentence):
                label = item["entity_group"].upper()
                if label not in {"PER", "ORG", "LOC"}:
                    continue
                mentions.append(
                    EntityMention(
                        text=item["word"],
                        label={"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}[label],
                        start=base_offset + int(item["start"]),
                        end=base_offset + int(item["end"]),
                        sentence_id=sentence_id,
                        confidence=round(float(item["score"]), 3),
                        source="transformer",
                    )
                )
        return mentions


def collect_label_statistics(mentions: Iterable[EntityMention]) -> dict[str, int]:
    counter = Counter(mention.label for mention in mentions)
    return dict(counter)
