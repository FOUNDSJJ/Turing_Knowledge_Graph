import re
from typing import List
from .schema import Entity
from .utils import contains_whole_word


class RuleNER:
    def __init__(self):
        self.person_dict = [
            "Alan Turing",
            "Gordon Welchman"
        ]

        self.org_dict = [
            "Bletchley Park",
            "University of Manchester",
            "ACM"
        ]

        self.loc_dict = [
            "London",
            "Wilmslow",
            "Manchester",
            "England",
            "Cheshire"
        ]

        self.work_dict = [
            "Turing machine",
            "Enigma",
            "Bombe"
        ]

        self.field_dict = [
            "Artificial Intelligence",
            "计算理论",
            "密码分析",
            "逻辑学",
            "计算机科学"
        ]

        self.event_dict = [
            "World War II",
            "第二次世界大战"
        ]

        self.award_dict = [
            "A.M. Turing Award",
            "Turing Award",
            "图灵奖",
            "ACM图灵奖"
        ]

    def _match_dictionary(self, text: str, dictionary: List[str], entity_type: str) -> List[Entity]:
        entities = []
        for item in dictionary:
            if contains_whole_word(text, item):
                entities.append(Entity(name=item, entity_type=entity_type))
        return entities

    def extract_awards(self, text: str) -> List[Entity]:
        return self._match_dictionary(text, self.award_dict, "AWARD")

    def extract_orgs(self, text: str) -> List[Entity]:
        entities = self._match_dictionary(text, self.org_dict, "ORG")
        pattern_cn = r"([\u4e00-\u9fffA-Za-z]{2,30}(?:大学|学院|实验室|研究院|研究所|公司|集团))"
        for m in re.finditer(pattern_cn, text):
            name = m.group(1)
            entities.append(Entity(name=name, entity_type="ORG"))
        return entities

    def extract_locs(self, text: str) -> List[Entity]:
        return self._match_dictionary(text, self.loc_dict, "LOC")

    def extract_works(self, text: str) -> List[Entity]:
        return self._match_dictionary(text, self.work_dict, "WORK")

    def extract_fields(self, text: str) -> List[Entity]:
        return self._match_dictionary(text, self.field_dict, "FIELD")

    def extract_events(self, text: str) -> List[Entity]:
        return self._match_dictionary(text, self.event_dict, "EVENT")

    def extract_persons(self, text: str) -> List[Entity]:
        entities = self._match_dictionary(text, self.person_dict, "PER")

        # 中文人物称谓
        cn_patterns = [
            r"([\u4e00-\u9fff]{2,4})(?:教授|博士|先生|女士)"
        ]
        for pattern in cn_patterns:
            for m in re.finditer(pattern, text):
                name = m.group(1)
                entities.append(Entity(name=name, entity_type="PER"))

        # 单独的 Turing / 图灵 作为别名，后面交给消歧归一化
        if contains_whole_word(text, "Turing"):
            entities.append(Entity(name="Turing", entity_type="PER"))
        if "图灵" in text:
            entities.append(Entity(name="图灵", entity_type="PER"))

        return entities

    def extract_time(self, text: str) -> List[Entity]:
        entities = []
        for m in re.finditer(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text):
            entities.append(Entity(name=m.group(1), entity_type="TIME"))
        for m in re.finditer(r"(\d{4}年)", text):
            entities.append(Entity(name=m.group(1), entity_type="TIME"))
        return entities

    def deduplicate(self, entities: List[Entity]) -> List[Entity]:
        seen = set()
        results = []
        for e in entities:
            key = (e.name, e.entity_type)
            if key not in seen:
                seen.add(key)
                results.append(e)
        return results

    def predict(self, text: str) -> List[Entity]:
        entities = []

        # 顺序很重要：先稳定专名，再识别人名
        entities.extend(self.extract_awards(text))
        entities.extend(self.extract_orgs(text))
        entities.extend(self.extract_locs(text))
        entities.extend(self.extract_works(text))
        entities.extend(self.extract_fields(text))
        entities.extend(self.extract_events(text))
        entities.extend(self.extract_persons(text))
        entities.extend(self.extract_time(text))

        return self.deduplicate(entities)