from typing import List, Dict
from .schema import Entity
from .utils import text_similarity


class SimpleEntityDisambiguator:
    def __init__(self, alias_map: Dict[str, str], entity_kb: Dict):
        self.alias_map = alias_map
        self.entity_kb = entity_kb

    def normalize_mention(self, mention: str) -> str:
        return self.alias_map.get(mention, mention)

    def get_candidates(self, mention: str) -> List[Dict]:
        return self.entity_kb.get(mention, [])

    def score_candidate(self, context: str, candidate: Dict) -> float:
        prior = candidate.get("prior", 0.0)
        keywords = candidate.get("keywords", [])
        context_score = text_similarity(context, keywords)
        return 0.6 * prior + 0.4 * context_score

    def disambiguate(self, entity: Entity, context: str) -> Entity:
        mention = self.normalize_mention(entity.name)
        candidates = self.get_candidates(mention)

        # 没有候选时只做别名归一化
        if not candidates:
            entity.normalized_name = mention
            return entity

        best = None
        best_score = -1.0
        for cand in candidates:
            score = self.score_candidate(context, cand)
            if score > best_score:
                best_score = score
                best = cand

        if best:
            entity.normalized_name = best["name"]
            entity.kb_id = best["kb_id"]
            entity.description = best["description"]

            # 若原类型明显不对，用 KB 类型覆盖
            kb_type = best.get("type", "")
            if kb_type:
                entity.entity_type = kb_type
        else:
            entity.normalized_name = mention

        return entity

    def batch_disambiguate(self, entities: List[Entity], context: str) -> List[Entity]:
        return [self.disambiguate(e, context) for e in entities]