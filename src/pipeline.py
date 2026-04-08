from typing import List, Dict
from .ner import RuleNER
from .entity_expansion import SimpleEntityExpander
from .entity_disambiguation import SimpleEntityDisambiguator
from .relation_extraction import RuleRelationExtractor
from .schema import Entity


class KnowledgeGraphPipeline:
    def __init__(self, alias_map: Dict, entity_kb: Dict):
        self.ner = RuleNER()
        self.expander = SimpleEntityExpander()
        self.disambiguator = SimpleEntityDisambiguator(alias_map, entity_kb)
        self.relation_extractor = RuleRelationExtractor()
        self.entity_kb = entity_kb
        self.alias_map = alias_map

    def _maybe_infer_entity_type(self, name: str) -> str:
        if name in self.entity_kb and self.entity_kb[name]:
            return self.entity_kb[name][0].get("type", "TERM")
        return "TERM"

    def _get_entity_from_relation_name(self, name: str) -> Entity:
        candidates = self.entity_kb.get(name, [])
        if candidates:
            cand = candidates[0]
            return Entity(
                name=name,
                entity_type=cand.get("type", "TERM"),
                normalized_name=cand.get("name", name),
                kb_id=cand.get("kb_id", ""),
                description=cand.get("description", "")
            )
        return Entity(
            name=name,
            entity_type=self._maybe_infer_entity_type(name),
            normalized_name=name
        )

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        merged = {}

        for e in entities:
            key = e.normalized_name or e.name
            if key not in merged:
                merged[key] = Entity(
                    name=key,
                    entity_type=e.entity_type,
                    normalized_name=key,
                    kb_id=e.kb_id,
                    description=e.description,
                    aliases=[]
                )
            base = merged[key]

            if e.name != key and e.name not in base.aliases:
                base.aliases.append(e.name)

            if not base.kb_id and e.kb_id:
                base.kb_id = e.kb_id

            if not base.description and e.description:
                base.description = e.description

            # 若已有实体类型过泛，尝试用更具体类型覆盖
            priority = {
                "PER": 6, "ORG": 5, "LOC": 5, "WORK": 5,
                "AWARD": 5, "FIELD": 4, "EVENT": 4, "TIME": 3, "TERM": 1
            }
            if priority.get(e.entity_type, 0) > priority.get(base.entity_type, 0):
                base.entity_type = e.entity_type

        return list(merged.values())

    def _ensure_relation_entities_exist(self, entities: List[Entity], relations) -> List[Entity]:
        existing = {e.normalized_name or e.name for e in entities}
        additions = []

        for r in relations:
            for node_name in [r.subject, r.object]:
                if node_name not in existing:
                    additions.append(self._get_entity_from_relation_name(node_name))
                    existing.add(node_name)

        return entities + additions

    def _filter_expanded_entities(self, expanded_entities: List, existing_names: set) -> List[Entity]:
        results = []
        for name, _score in expanded_entities:
            if name in existing_names:
                continue
            if name in self.alias_map:
                norm = self.alias_map[name]
            else:
                norm = name

            entity_type = self._maybe_infer_entity_type(norm)

            # 不再把 TIME 和明显噪声扩展进来
            if entity_type == "TIME":
                continue

            results.append(Entity(name=name, entity_type=entity_type))
        return results

    def build(self, text: str, seed_entities: List[str] = None) -> Dict:
        seed_entities = seed_entities or []

        # 1. 实体识别
        entities = self.ner.predict(text)
        print(f"NER识别到的实体: {[e.name for e in entities]}\n")

        # 2. 实体扩展
        if seed_entities:
            expanded = self.expander.expand(text, seed_entities, top_k=5)
            print(f"实体扩展得到的候选: {expanded}\n")

            existing_names = {e.name for e in entities}
            print(f"已有实体名称: {existing_names}\n")

            entities.extend(self._filter_expanded_entities(expanded, existing_names))
            print(f"扩展后实体列表: {[e.name for e in entities]}\n")

        # 3. 实体消歧 / 归一化
        entities = self.disambiguator.batch_disambiguate(entities, text)
        entities = self._merge_entities(entities)
        print(f"消歧后实体列表: {[e.normalized_name for e in entities]}\n")

        # 4. 关系抽取
        relations = self.relation_extractor.extract(text, entities)

        # 5. 保证关系中的节点在实体表中存在
        entities = self._ensure_relation_entities_exist(entities, relations)

        # 6. 再做一次消歧，补齐新增节点信息
        entities = self.disambiguator.batch_disambiguate(entities, text)

        # 7. 合并同实体
        entities = self._merge_entities(entities)

        return {
            "entities": [e.to_dict() for e in entities],
            "relations": [r.to_dict() for r in relations]
        }