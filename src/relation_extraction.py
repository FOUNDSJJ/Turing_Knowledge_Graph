import re
from typing import List
from .schema import Relation


class RuleRelationExtractor:
    def extract(self, text: str, entities: List) -> List[Relation]:
        relations = []
        sentences = re.split(r"[。！？!?；;\n]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sent in sentences:
            # 身份
            if ("Alan Turing" in sent or "Turing" in sent or "图灵" in sent) and ("数学家" in sent or "逻辑学家" in sent or "密码分析学家" in sent):
                if "数学家" in sent:
                    relations.append(Relation("Alan Turing", "is_a", "数学家", sent))
                if "逻辑学家" in sent:
                    relations.append(Relation("Alan Turing", "is_a", "逻辑学家", sent))
                if "密码分析学家" in sent:
                    relations.append(Relation("Alan Turing", "is_a", "密码分析学家", sent))

            # 出生地 / 出生时间
            if ("Alan Turing" in sent or "Turing" in sent or "图灵" in sent):
                if re.search(r"出生于|出生在", sent):
                    if "London" in sent:
                        relations.append(Relation("Alan Turing", "born_in", "London", sent))
                    if "England" in sent:
                        relations.append(Relation("Alan Turing", "born_in", "England", sent))

                m = re.search(r"(18\d{2}|19\d{2}|20\d{2})年", sent)
                if m and ("出生" in sent):
                    relations.append(Relation("Alan Turing", "born_on", m.group(1), sent))

            # 提出图灵机
            if ("Alan Turing" in sent or "Turing" in sent or "图灵" in sent) and ("Turing machine" in sent or "图灵机" in sent):
                if re.search(r"提出|提出了|proposed", sent):
                    relations.append(Relation("Alan Turing", "proposed", "Turing machine", sent))
                    relations.append(Relation("Alan Turing", "contributed_to", "计算理论", sent))

            # 二战 / Bletchley Park / Enigma / Bombe
            if ("Alan Turing" in sent or "Turing" in sent or "图灵" in sent):
                if "Bletchley Park" in sent and re.search(r"工作|加入|在.*工作", sent):
                    relations.append(Relation("Alan Turing", "worked_at", "Bletchley Park", sent))

                if ("Enigma" in sent or "恩尼格玛" in sent) and re.search(r"破译|破解|解密", sent):
                    relations.append(Relation("Alan Turing", "helped_break", "Enigma machine", sent))
                    relations.append(Relation("Alan Turing", "contributed_to", "密码分析", sent))

                if "Bombe" in sent and re.search(r"设计|推动|改进|develop", sent):
                    relations.append(Relation("Alan Turing", "designed", "Bombe", sent))

                if "第二次世界大战" in sent or "World War II" in sent:
                    relations.append(Relation("Alan Turing", "related_to", "World War II", sent))

            # Gordon Welchman
            if "Gordon Welchman" in sent:
                if "Bletchley Park" in sent:
                    relations.append(Relation("Gordon Welchman", "worked_at", "Bletchley Park", sent))
                if "Bombe" in sent and re.search(r"改进|improved|贡献", sent):
                    relations.append(Relation("Gordon Welchman", "improved", "Bombe", sent))
                    relations.append(Relation("Gordon Welchman", "collaborated_with", "Alan Turing", sent))

            # 曼彻斯特大学
            if ("Alan Turing" in sent or "Turing" in sent or "图灵" in sent) and "University of Manchester" in sent:
                if re.search(r"加入|joined|工作|研究", sent):
                    relations.append(Relation("Alan Turing", "worked_at", "University of Manchester", sent))

            # AI
            if ("Alan Turing" in sent or "Turing" in sent or "图灵" in sent) and ("Artificial Intelligence" in sent or "人工智能" in sent):
                if re.search(r"奠基|先驱|基础|早期", sent):
                    relations.append(Relation("Alan Turing", "contributed_to", "Artificial Intelligence", sent))

            # 图灵奖
            if ("A.M. Turing Award" in sent or "Turing Award" in sent or "图灵奖" in sent):
                relations.append(Relation("A.M. Turing Award", "named_after", "Alan Turing", sent))
                if "ACM" in sent:
                    relations.append(Relation("Association for Computing Machinery", "established", "A.M. Turing Award", sent))

            # 去世
            if ("Alan Turing" in sent or "Turing" in sent or "图灵" in sent) and re.search(r"去世|逝世|died", sent):
                if "Wilmslow" in sent:
                    relations.append(Relation("Alan Turing", "died_in", "Wilmslow", sent))
                m = re.search(r"(18\d{2}|19\d{2}|20\d{2})年", sent)
                if m:
                    relations.append(Relation("Alan Turing", "died_on", m.group(1), sent))

        return self._deduplicate(relations)

    def _deduplicate(self, relations: List[Relation]) -> List[Relation]:
        seen = set()
        results = []
        for r in relations:
            key = (r.subject, r.predicate, r.object, r.evidence)
            if key not in seen:
                seen.add(key)
                results.append(r)
        return results