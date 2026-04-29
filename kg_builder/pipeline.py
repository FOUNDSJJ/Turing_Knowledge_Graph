from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_resource_config
from .entity_linking import EntityDisambiguator, EntityExtender
from .ner import (
    CRFEntityRecognizer,
    RuleEntityRecognizer,
    TransformerAssistant,
    collect_label_statistics,
)
from .relation_extraction import RelationExtractor
from .schema import KnowledgeGraph


class KnowledgeGraphPipeline:
    def __init__(
        self,
        use_crf: bool = False,
        crf_model_path: str | None = None,
        transformer_model: str | None = None,
        transformer_tokenizer: str | None = None,
        config_dir: str | None = None,
        entity_patterns_path: str | None = None,
        lexicon_path: str | None = None,
        alias_table_path: str | None = None,
        knowledge_base_path: str | None = None,
        relation_rules_path: str | None = None,
        normalization_path: str | None = None,
        relation_extractor: str = "hybrid",
        relation_model: str | None = None,
        relation_tokenizer: str | None = None,
        relation_device: int | None = None,
        relation_max_length: int = 256,
        relation_num_beams: int = 3,
        relation_min_confidence: float = 0.0,
        relation_source_lang: str | None = None,
        relation_decoder_start_token: str | None = None,
        relation_augment_entities: bool = True,
        relation_context_window: int = 1,
    ) -> None:
        self.resources = load_resource_config(
            config_dir=config_dir,
            entity_patterns_path=entity_patterns_path,
            lexicon_path=lexicon_path,
            alias_table_path=alias_table_path,
            knowledge_base_path=knowledge_base_path,
            relation_rules_path=relation_rules_path,
            normalization_path=normalization_path,
        )
        self.rule_ner = RuleEntityRecognizer(
            entity_patterns=self.resources.entity_patterns,
            lexicon=self.resources.lexicon,
            normalization=self.resources.normalization,
        )
        self.crf_ner = CRFEntityRecognizer(crf_model_path) if use_crf and crf_model_path else None
        self.transformer_assistant = TransformerAssistant(
            transformer_model,
            tokenizer_name=transformer_tokenizer,
            normalization=self.resources.normalization,
        )
        self.extender = EntityExtender(
            alias_table=self.resources.alias_table,
            normalization=self.resources.normalization,
        )
        self.disambiguator = EntityDisambiguator(
            knowledge_base=self.resources.knowledge_base,
            alias_table=self.resources.alias_table,
            normalization=self.resources.normalization,
        )
        self.relation_extractor = RelationExtractor(
            self.resources.relation_rules,
            mode=relation_extractor,
            model_name=relation_model,
            tokenizer_name=relation_tokenizer,
            device=relation_device,
            max_length=relation_max_length,
            num_beams=relation_num_beams,
            min_confidence=relation_min_confidence,
            source_lang=relation_source_lang,
            decoder_start_token=relation_decoder_start_token,
            augment_entities=relation_augment_entities,
            context_window=relation_context_window,
        )

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
                    "relation_extraction": self.relation_extractor.metadata(),
                    "config_paths": {
                        key: str(path) for key, path in self.resources.paths.items()
                    },
                },
            },
        )
        return graph.to_dict()

    def build_from_file(self, input_path: str | Path) -> dict[str, Any]:
        text = Path(input_path).read_text(encoding="utf-8")
        return self.build_from_text(text)

    def _recognize_entities(self, text: str) -> tuple[list[str], list[Any]]:
        sentences, mentions = self.rule_ner.recognize(text)
        if self.crf_ner and self.crf_ner.is_ready():
            print("Using CRF model for auxiliary entity recognition...")
            crf_sentences, crf_mentions = self.crf_ner.recognize(text)
            sentences = crf_sentences or sentences
            mentions.extend(crf_mentions)
        mentions.extend(self.transformer_assistant.augment(text))
        mentions = self.rule_ner.deduplicate_mentions(mentions)
        return sentences, mentions
