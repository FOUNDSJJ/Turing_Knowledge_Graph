from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .entity_linking import EntityDisambiguator, EntityExtender
from .ner import CRFEntityRecognizer, RuleEntityRecognizer, TransformerAssistant, collect_label_statistics
from .relation_extraction import RelationExtractor
from .schema import KnowledgeGraph


class KnowledgeGraphPipeline:
    def __init__(
        self,
        use_crf: bool = False,
        crf_model_path: str | None = None,
        transformer_model: str | None = None,
    ) -> None:
        self.rule_ner = RuleEntityRecognizer()
        self.crf_ner = CRFEntityRecognizer(crf_model_path) if use_crf and crf_model_path else None
        self.transformer_assistant = TransformerAssistant(transformer_model)
        self.extender = EntityExtender()
        self.disambiguator = EntityDisambiguator()
        self.relation_extractor = RelationExtractor()

    def build_from_text(self, text: str) -> dict[str, Any]:
        sentences, mentions = self._recognize_entities(text)
        mentions = self.extender.expand(mentions, text)
        entities = self.disambiguator.link(mentions, sentences)
        relations = self.relation_extractor.extract(sentences, entities)
        graph = KnowledgeGraph(
            text=text,
            sentences=sentences,
            entities=entities,
            relations=relations,
            metadata={
                "entity_count": len(entities),
                "relation_count": len(relations),
                "mention_count": len(mentions),
                "label_distribution": collect_label_statistics(mentions),
                "pipeline": {
                    "rule_ner": True,
                    "crf_enabled": bool(self.crf_ner and self.crf_ner.is_ready()),
                    "transformer_enabled": bool(self.transformer_assistant.pipeline),
                },
            },
        )
        return graph.to_dict()

    def build_from_file(self, input_path: str | Path) -> dict[str, Any]:
        text = Path(input_path).read_text(encoding="utf-8")
        return self.build_from_text(text)

    def _recognize_entities(self, text: str):
        sentences, mentions = self.rule_ner.recognize(text)
        if self.crf_ner and self.crf_ner.is_ready():
            crf_sentences, crf_mentions = self.crf_ner.recognize(text)
            sentences = crf_sentences or sentences
            mentions.extend(crf_mentions)
        mentions.extend(self.transformer_assistant.augment(text))
        mentions = self.rule_ner._deduplicate_mentions(mentions)
        return sentences, mentions
