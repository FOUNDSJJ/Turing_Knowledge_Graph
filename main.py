from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_builder import KnowledgeGraphPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from plain text with configurable NER and linking resources."
    )
    parser.add_argument("--input", required=True, help="Input txt file path")
    parser.add_argument("--output", default="output/kg.json", help="Output json path")
    parser.add_argument("--use-crf", action="store_true", help="Enable CRF NER if model exists")
    parser.add_argument("--crf-model", default="models/crf_ner.pkl", help="CRF model path")
    parser.add_argument(
        "--transformer-model",
        default=None,
        help="Optional local Hugging Face token classification model path or model name",
    )
    parser.add_argument(
        "--transformer-tokenizer",
        default=None,
        help="Optional tokenizer path or name when it differs from the transformer model",
    )
    parser.add_argument(
        "--config-dir",
        default="data/config",
        help="Directory containing configurable dictionaries and extraction rules",
    )
    parser.add_argument("--entity-patterns", default=None, help="Override entity pattern JSON path")
    parser.add_argument("--lexicon", default=None, help="Override lexicon JSON path")
    parser.add_argument("--alias-table", default=None, help="Override alias table JSON path")
    parser.add_argument("--knowledge-base", default=None, help="Override knowledge base JSON path")
    parser.add_argument("--relation-rules", default=None, help="Override relation rules JSON path")
    parser.add_argument(
        "--relation-extractor",
        choices=["rules", "transformer", "hybrid"],
        default="hybrid",
        help=(
            "Relation extraction backend. Use transformer to avoid rule/keyword "
            "relations; hybrid runs the pretrained model first and then rules."
        ),
    )
    parser.add_argument(
        "--relation-model",
        default=None,
        help=(
            "Optional local or Hugging Face seq2seq relation extraction model, "
            "for example Babelscape/mrebel-large or Babelscape/rebel-large."
        ),
    )
    parser.add_argument(
        "--relation-tokenizer",
        default=None,
        help="Optional relation extraction tokenizer path or name",
    )
    parser.add_argument(
        "--relation-device",
        type=int,
        default=None,
        help="Device for relation model: -1 for CPU, 0 for first GPU",
    )
    parser.add_argument(
        "--relation-max-length",
        type=int,
        default=256,
        help="Maximum generated tokens for relation extraction",
    )
    parser.add_argument(
        "--relation-num-beams",
        type=int,
        default=3,
        help="Beam size for seq2seq relation extraction",
    )
    parser.add_argument(
        "--relation-min-confidence",
        type=float,
        default=0.0,
        help="Discard model-generated relations below this confidence",
    )
    parser.add_argument(
        "--relation-source-lang",
        default=None,
        help="Optional source language token for multilingual models, e.g. zh_CN or en_XX",
    )
    parser.add_argument(
        "--relation-decoder-start-token",
        default=None,
        help="Optional decoder start token for multilingual triplet models, e.g. tp_XX",
    )
    parser.add_argument(
        "--normalization-config",
        default=None,
        help="Override normalization JSON path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = KnowledgeGraphPipeline(
        use_crf=args.use_crf,
        crf_model_path=args.crf_model,
        transformer_model=args.transformer_model,
        transformer_tokenizer=args.transformer_tokenizer,
        config_dir=args.config_dir,
        entity_patterns_path=args.entity_patterns,
        lexicon_path=args.lexicon,
        alias_table_path=args.alias_table,
        knowledge_base_path=args.knowledge_base,
        relation_rules_path=args.relation_rules,
        normalization_path=args.normalization_config,
        relation_extractor=args.relation_extractor,
        relation_model=args.relation_model,
        relation_tokenizer=args.relation_tokenizer,
        relation_device=args.relation_device,
        relation_max_length=args.relation_max_length,
        relation_num_beams=args.relation_num_beams,
        relation_min_confidence=args.relation_min_confidence,
        relation_source_lang=args.relation_source_lang,
        relation_decoder_start_token=args.relation_decoder_start_token,
    )
    graph = pipeline.build_from_file(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Knowledge graph saved to {output_path}")


if __name__ == "__main__":
    main()
