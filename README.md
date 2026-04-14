# 知识图谱构建示例（Python）

这个项目实现了一条从 `txt` 文本到 `json` 知识图谱的完整流水线，覆盖：

- 实体识别：默认使用规则 + 词典方法，支持接入 `CRF` 模型
- 实体扩展：别名归一化、称谓补全
- 实体消歧：字符串相似度 + 上下文重叠度 + 简单知识库匹配
- 关系提取：基于模板规则的关系抽取
- 图谱导出：输出统一的 `JSON` 结构

## 目录结构

```text
Knowledge_Graph/
├── kg_builder/
│   ├── __init__.py
│   ├── entity_linking.py
│   ├── ner.py
│   ├── pipeline.py
│   ├── relation_extraction.py
│   └── schema.py
├── data/
│   └── sample_bio_train.jsonl
├── sample_text.txt
├── train_crf.py
├── main.py
└── requirements.txt
```

## 输入输出格式

输入：一段 `.txt` 文本。

输出：`JSON` 知识图谱，核心格式如下：

```json
{
  "text": "原始文本",
  "sentences": ["句子1", "句子2"],
  "entities": [
    {
      "entity_id": "E001",
      "name": "北京大学",
      "entity_type": "ORG",
      "aliases": ["北大"],
      "mentions": [],
      "attributes": {
        "mention_count": 2,
        "sentence_ids": [0, 2]
      },
      "description": "中国北京市的一所综合性大学。",
      "confidence": 0.91
    }
  ],
  "relations": [
    {
      "head": "E002",
      "tail": "E001",
      "relation": "works_for",
      "sentence_id": 0,
      "evidence": "张三教授在北京大学信息科学技术学院工作。",
      "confidence": 0.82
    }
  ],
  "metadata": {}
}
```

## 运行方式

### 1. 直接运行默认流水线

```bash
python main.py --input sample_text.txt --output output/kg.json
```

### 2. 使用 CRF 模型

先训练：

```bash
python train_crf.py --train data/sample_bio_train.jsonl --output models/crf_ner.pkl
```

再执行：

```bash
python main.py --input sample_text.txt --output output/kg.json --use-crf --crf-model models/crf_ner.pkl
```

### 3. 使用预训练模型辅助实体识别

如果本地已经有可用的 Hugging Face 中文命名实体识别模型，可以这样接入：

```bash
python main.py --input sample_text.txt --output output/kg.json --transformer-model your_local_model
```

## 方法说明

### 1. 实体识别

- 默认实现：规则模板 + 词典匹配
- 可选增强：字符级 `CRF`
- 再增强：预训练模型辅助补召回

这里的 `CRF` 使用的是字符级特征，包括：

- 当前字
- 前后字
- 左右双字组合
- 是否数字
- 简单大小写特征

这种设计比较适合课程作业里说明传统序列标注方法。

### 2. 实体扩展

主要做两件事：

- 别名归一化，例如 `北大 -> 北京大学`
- 实体提及扩展，例如保留人物称谓上下文

### 3. 实体消歧

使用以下组合分数：

- 名称字符串相似度
- 当前句子与知识库描述的上下文重叠度
- 提及类别与候选实体类别是否一致

### 4. 关系提取

当前实现是规则模板法，内置了以下关系：

- `studies_at`
- `works_for`
- `founded`
- `located_in`
- `cooperates_with`
- `belongs_to`

如果你后续想升级成监督学习关系分类器，可以在这个模块上继续扩展。

## 适合课程报告怎么写

你可以把系统描述成一个“混合式知识图谱构建框架”：

1. 使用传统机器学习中的 `CRF` 进行序列标注式实体识别
2. 使用规则和词典提升小样本场景下的可用性
3. 使用轻量知识库做实体归一和消歧
4. 使用模板规则做关系提取
5. 最终导出结构化 `JSON` 图谱

这样既符合“传统方法为主”的要求，也保留了“预训练模型辅助”的扩展空间。
