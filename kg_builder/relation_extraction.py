from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .schema import EntityNode, RelationEdge


DEFAULT_MODEL_CONFIDENCE = 0.86


@dataclass(slots=True)
class ExtractedTriplet:
    head: str
    relation: str
    tail: str
    confidence: float = DEFAULT_MODEL_CONFIDENCE


class RelationExtractor:
    """Relation extraction facade.

    The rule backend is kept for compatibility. The transformer backend is an
    open triplet generator: it reads a sentence and emits (head, relation, tail)
    candidates without project-specific trigger words.
    """

    def __init__(
        self,
        relation_rules: list[dict[str, Any]] | None = None,
        mode: str = "hybrid",
        model_name: str | None = None,
        tokenizer_name: str | None = None,
        device: int | None = None,
        max_length: int = 256,
        num_beams: int = 3,
        min_confidence: float = 0.0,
        source_lang: str | None = None,
        decoder_start_token: str | None = None,
    ) -> None:
        self.mode = self._normalize_mode(mode)
        if self.mode == "transformer" and not model_name:
            raise ValueError(
                "relation_model is required when relation_extractor='transformer'"
            )
        self.rule_extractor = RuleRelationExtractor(relation_rules)
        self.model_extractor = TransformerRelationExtractor(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            device=device,
            max_length=max_length,
            num_beams=num_beams,
            min_confidence=min_confidence,
            source_lang=source_lang,
            decoder_start_token=decoder_start_token,
        )
        if self.mode == "transformer" and not self.model_extractor.is_ready():
            detail = self.model_extractor.load_error or "unknown model loading error"
            raise RuntimeError(
                "Transformer relation extractor is not ready. "
                f"Please install dependencies and check the model path/name. Detail: {detail}"
            )

    def extract(
        self,
        sentences: list[str],
        entities: list[EntityNode],
    ) -> list[RelationEdge]:
        relations: list[RelationEdge] = []
        if self.mode in {"transformer", "hybrid"}:
            relations.extend(self.model_extractor.extract(sentences, entities))
        if self.mode in {"rules", "hybrid"}:
            relations.extend(self.rule_extractor.extract(sentences, entities))
        return deduplicate_relations(relations)

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "transformer_model": self.model_extractor.model_name,
            "transformer_ready": self.model_extractor.is_ready(),
            "transformer_error": self.model_extractor.load_error,
            "source_lang": self.model_extractor.source_lang,
            "decoder_start_token": self.model_extractor.decoder_start_token,
            "rule_backend_enabled": self.mode in {"rules", "hybrid"},
        }

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = (mode or "hybrid").strip().lower()
        if normalized not in {"rules", "transformer", "hybrid"}:
            raise ValueError(
                "relation extractor mode must be one of: rules, transformer, hybrid"
            )
        return normalized


class RuleRelationExtractor:
    def __init__(self, relation_rules: list[dict[str, Any]] | None = None) -> None:
        self.relation_rules = relation_rules or []

    def extract(
        self,
        sentences: list[str],
        entities: list[EntityNode],
    ) -> list[RelationEdge]:
        relations: list[RelationEdge] = []
        entity_index = build_sentence_entity_index(entities)
        for sentence_id, sentence in enumerate(sentences):
            local_entities = entity_index.get(sentence_id, [])
            for head in local_entities:
                for tail in local_entities:
                    if head.entity_id == tail.entity_id:
                        continue
                    relations.extend(
                        self._match_rules(head, tail, sentence, sentence_id)
                    )
        return deduplicate_relations(relations)

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


class TransformerRelationExtractor:
    """Generic pretrained-model relation extraction backend.

    This backend is designed for seq2seq relation extraction models such as
    REBEL/mREBEL. It also accepts JSON-like outputs from instruction-tuned
    text2text models, as long as they contain head/relation/tail fields.
    """

    def __init__(
        self,
        model_name: str | None = None,
        tokenizer_name: str | None = None,
        device: int | None = None,
        max_length: int = 256,
        num_beams: int = 3,
        min_confidence: float = 0.0,
        source_lang: str | None = None,
        decoder_start_token: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
        self.min_confidence = min_confidence
        self.source_lang = source_lang
        self.decoder_start_token = decoder_start_token
        self.model = None
        self.tokenizer = None
        self.load_error: str | None = None
        if not model_name:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer_kwargs = {}
            if self.source_lang:
                tokenizer_kwargs["src_lang"] = self.source_lang
            if self.decoder_start_token:
                tokenizer_kwargs["tgt_lang"] = self.decoder_start_token
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                **tokenizer_kwargs,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            if self.device is not None and self.device >= 0:
                self.model.to(f"cuda:{self.device}")
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.load_error = str(exc)

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def extract(
        self,
        sentences: list[str],
        entities: list[EntityNode],
    ) -> list[RelationEdge]:
        if not self.is_ready():
            return []

        relations: list[RelationEdge] = []
        entity_index = build_sentence_entity_index(entities)
        for sentence_id, sentence in enumerate(sentences):
            local_entities = entity_index.get(sentence_id, [])
            if len(local_entities) < 2:
                continue
            generated_texts = self._generate(sentence)
            for generated_text in generated_texts:
                triplets = self._parse_triplets(generated_text)
                for triplet in triplets:
                    edge = self._triplet_to_edge(
                        triplet=triplet,
                        sentence=sentence,
                        sentence_id=sentence_id,
                        local_entities=local_entities,
                    )
                    if edge is not None:
                        relations.append(edge)
        return deduplicate_relations(relations)

    def _generate(self, sentence: str) -> list[str]:
        if not self.is_ready():
            return []
        model_inputs = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device) for key, value in model_inputs.items()
        }
        gen_kwargs: dict[str, Any] = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "length_penalty": 0,
            "forced_bos_token_id": None,
        }
        decoder_start_token_id = self._decoder_start_token_id()
        if decoder_start_token_id is not None:
            gen_kwargs["decoder_start_token_id"] = decoder_start_token_id
        generated_tokens = self.model.generate(**model_inputs, **gen_kwargs)
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=False,
            )
            if text.strip()
        ]

    def _decoder_start_token_id(self) -> int | None:
        if self.tokenizer is None:
            return None
        tokens = [
            self.decoder_start_token,
            "tp_XX" if self.model_name and "mrebel" in self.model_name.lower() else None,
        ]
        for token in tokens:
            if not token:
                continue
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                return int(token_id)
        return None

    def _triplet_to_edge(
        self,
        triplet: ExtractedTriplet,
        sentence: str,
        sentence_id: int,
        local_entities: list[EntityNode],
    ) -> RelationEdge | None:
        if triplet.confidence < self.min_confidence:
            return None
        head = match_entity(triplet.head, local_entities)
        tail = match_entity(triplet.tail, local_entities)
        relation = normalize_relation_name(triplet.relation)
        if head is None or tail is None or head.entity_id == tail.entity_id or not relation:
            return None
        confidence = min(0.99, max(0.0, triplet.confidence))
        return RelationEdge(
            head=head.entity_id,
            tail=tail.entity_id,
            relation=relation,
            sentence_id=sentence_id,
            evidence=sentence.strip(),
            confidence=round(confidence, 3),
        )

    def _parse_triplets(self, generated_text: str) -> list[ExtractedTriplet]:
        triplets = parse_rebel_triplets(generated_text)
        if triplets:
            return triplets

        triplets = parse_mrebel_triplets(generated_text)
        if triplets:
            return triplets

        triplets = parse_json_triplets(generated_text)
        if triplets:
            return triplets

        return parse_delimited_triplets(generated_text)


def parse_rebel_triplets(text: str) -> list[ExtractedTriplet]:
    triplets: list[ExtractedTriplet] = []
    relation = ""
    subject = ""
    object_ = ""
    current = ""
    cleaned = strip_generation_markers(text)
    for token in cleaned.split():
        if token == "<triplet>":
            if subject and relation and object_:
                triplets.append(
                    ExtractedTriplet(
                        head=subject.strip(),
                        relation=relation.strip(),
                        tail=object_.strip(),
                    )
                )
            current = "subject"
            subject = ""
            relation = ""
            object_ = ""
        elif token == "<subj>":
            current = "object"
            object_ = ""
        elif token == "<obj>":
            current = "relation"
            relation = ""
        elif current == "subject":
            subject = append_token(subject, token)
        elif current == "object":
            object_ = append_token(object_, token)
        elif current == "relation":
            relation = append_token(relation, token)

    if subject and relation and object_:
        triplets.append(
            ExtractedTriplet(
                head=subject.strip(),
                relation=relation.strip(),
                tail=object_.strip(),
            )
        )
    return triplets


def parse_mrebel_triplets(text: str) -> list[ExtractedTriplet]:
    triplets: list[ExtractedTriplet] = []
    relation = ""
    subject = ""
    object_ = ""
    current = ""
    cleaned = strip_generation_markers(text)

    for token in cleaned.split():
        if token in {"<triplet>", "<relation>"}:
            if subject and relation and object_:
                triplets.append(
                    ExtractedTriplet(
                        head=subject.strip(),
                        relation=relation.strip(),
                        tail=object_.strip(),
                    )
                )
            current = "subject"
            subject = ""
            relation = ""
            object_ = ""
        elif is_angle_token(token):
            if current in {"subject", "relation"}:
                current = "object"
                object_ = ""
            elif current == "object":
                current = "relation"
                relation = ""
        elif current == "subject":
            subject = append_token(subject, token)
        elif current == "object":
            object_ = append_token(object_, token)
        elif current == "relation":
            relation = append_token(relation, token)

    if subject and relation and object_:
        triplets.append(
            ExtractedTriplet(
                head=subject.strip(),
                relation=relation.strip(),
                tail=object_.strip(),
            )
        )
    return triplets


def parse_json_triplets(text: str) -> list[ExtractedTriplet]:
    payload = extract_json_payload(text)
    if payload is None:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        candidates = data.get("relations") or data.get("triples") or data.get("triplets")
        if candidates is None:
            candidates = [data]
    else:
        candidates = data

    triplets: list[ExtractedTriplet] = []
    if not isinstance(candidates, list):
        return triplets
    for item in candidates:
        if not isinstance(item, dict):
            continue
        head = first_present(item, ("head", "subject", "source", "h"))
        relation = first_present(item, ("relation", "predicate", "type", "label", "r"))
        tail = first_present(item, ("tail", "object", "target", "t"))
        if not head or not relation or not tail:
            continue
        confidence = item.get("confidence", DEFAULT_MODEL_CONFIDENCE)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = DEFAULT_MODEL_CONFIDENCE
        triplets.append(
            ExtractedTriplet(
                head=str(head).strip(),
                relation=str(relation).strip(),
                tail=str(tail).strip(),
                confidence=confidence_value,
            )
        )
    return triplets


def parse_delimited_triplets(text: str) -> list[ExtractedTriplet]:
    triplets: list[ExtractedTriplet] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        normalized = line.strip(" -")
        match = re.match(
            r"^\(?\s*(?P<head>[^|,\t;]+?)\s*(?:\||,|\t|;)\s*"
            r"(?P<relation>[^|,\t;]+?)\s*(?:\||,|\t|;)\s*"
            r"(?P<tail>[^|,\t;]+?)\s*\)?$",
            normalized,
        )
        if not match:
            match = re.match(
                r"^(?P<head>.+?)\s*->\s*(?P<relation>.+?)\s*->\s*(?P<tail>.+?)$",
                normalized,
            )
        if match:
            triplets.append(
                ExtractedTriplet(
                    head=match.group("head").strip(),
                    relation=match.group("relation").strip(),
                    tail=match.group("tail").strip(),
                )
            )
    return triplets


def build_sentence_entity_index(
    entities: list[EntityNode],
) -> dict[int, list[EntityNode]]:
    index: dict[int, list[EntityNode]] = {}
    for entity in entities:
        for sentence_id in entity.attributes.get("sentence_ids", []):
            index.setdefault(sentence_id, []).append(entity)
    return index


def match_entity(candidate: str, entities: list[EntityNode]) -> EntityNode | None:
    normalized_candidate = normalize_entity_text(candidate)
    if not normalized_candidate:
        return None

    best_entity: EntityNode | None = None
    best_score = 0.0
    for entity in entities:
        variants = [entity.name, *entity.aliases]
        for variant in variants:
            score = entity_match_score(normalized_candidate, normalize_entity_text(variant))
            if score > best_score:
                best_entity = entity
                best_score = score
    return best_entity if best_score >= 0.72 else None


def entity_match_score(candidate: str, variant: str) -> float:
    if not candidate or not variant:
        return 0.0
    if candidate == variant:
        return 1.0
    if len(candidate) < 2 or len(variant) < 2:
        return 0.0
    if candidate in variant or variant in candidate:
        return min(len(candidate), len(variant)) / max(len(candidate), len(variant))
    return 0.0


def normalize_entity_text(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"[\s'\"`.,;:!?()\[\]{}<>]+", "", lowered)


def normalize_relation_name(relation: str) -> str:
    cleaned = re.sub(r"\s+", " ", relation.strip())
    cleaned = cleaned.strip("'\"`.,;:!?()[]{}")
    if not cleaned:
        return ""
    if re.search(r"[A-Za-z]", cleaned):
        cleaned = re.sub(r"[^0-9A-Za-z_ -]+", "", cleaned)
        cleaned = re.sub(r"[\s-]+", "_", cleaned.strip().lower())
    return cleaned


def extract_json_payload(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    object_start = stripped.find("{")
    array_start = stripped.find("[")
    starts = [idx for idx in (object_start, array_start) if idx != -1]
    if not starts:
        return None
    start = min(starts)
    end_char = "}" if stripped[start] == "{" else "]"
    end = stripped.rfind(end_char)
    if end <= start:
        return None
    return stripped[start : end + 1]


def first_present(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def append_token(text: str, token: str) -> str:
    if not text:
        return token
    if is_cjk(text[-1]) or is_cjk(token[0]):
        return text + token
    return f"{text} {token}"


def strip_generation_markers(text: str) -> str:
    cleaned = (
        text.strip()
        .replace("<s>", "")
        .replace("<pad>", "")
        .replace("</s>", "")
        .replace("tp_XX", "")
    )
    return re.sub(r"__[\w-]+__", "", cleaned).strip()


def is_angle_token(token: str) -> bool:
    if token in {"<triplet>", "<relation>", "<subj>", "<obj>"}:
        return False
    return token.startswith("<") and token.endswith(">")


def is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def deduplicate_relations(relations: list[RelationEdge]) -> list[RelationEdge]:
    unique: dict[tuple[str, str, str, int], RelationEdge] = {}
    for relation in relations:
        key = (
            relation.head,
            relation.tail,
            relation.relation,
            relation.sentence_id,
        )
        current = unique.get(key)
        if current is None or relation.confidence >= current.confidence:
            unique[key] = relation
    return list(unique.values())
