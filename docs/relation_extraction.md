# 预训练模型关系抽取

项目现在支持两类关系抽取后端：

- `rules`：沿用 `data/config/relation_rules.json` 的规则抽取。
- `transformer`：使用 Hugging Face seq2seq 关系抽取模型直接生成三元组，不依赖项目内预设触发词。
- `hybrid`：先运行模型抽取，再运行规则抽取并去重。未配置模型时会自动退回规则抽取。

## 推荐模型

通用关系抽取建议优先使用 REBEL/mREBEL 这类“文本到三元组”的预训练模型：

- 多语言文本：`Babelscape/mrebel-large`
- 英文文本：`Babelscape/rebel-large`

这类模型会从句子中生成 `<triplet> head <subj> tail <obj> relation` 形式的开放三元组。本项目会把模型生成的 head/tail 映射回已有实体节点，然后写入知识图谱关系边。

## 安装依赖

`requirements.txt` 已包含所需依赖：

```bash
pip install -r requirements.txt
```

如果想离线运行，先把模型下载到本地目录：

```bash
huggingface-cli download Babelscape/mrebel-large --local-dir models/mrebel-large
```

## 使用纯模型关系抽取

纯模型模式不会读取 `relation_rules.json` 来产生关系：

```bash
python main.py ^
  --input sample_text.txt ^
  --output output/kg_re.json ^
  --relation-extractor transformer ^
  --relation-model models/mrebel-large ^
  --relation-source-lang en_XX ^
  --relation-decoder-start-token tp_XX
```

如果希望直接从 Hugging Face 拉取模型，也可以把模型名作为参数：

```bash
python main.py --input sample_text.txt --output output/kg_re.json --relation-extractor transformer --relation-model Babelscape/mrebel-large --relation-source-lang en_XX --relation-decoder-start-token tp_XX
```

处理中文文本时，可以把 `--relation-source-lang` 改为 `zh_CN`。

## 和 Transformer NER 一起使用

关系抽取依赖实体节点。为了减少规则实体识别带来的限制，可以同时启用预训练 NER：

```bash
python main.py ^
  --input your_text.txt ^
  --output output/kg_transformer.json ^
  --transformer-model models/bert-base-ner ^
  --relation-extractor transformer ^
  --relation-model models/mrebel-large
```

## 输出元数据

输出 JSON 的 `metadata.pipeline.relation_extraction` 会记录：

- `mode`：当前关系抽取模式。
- `transformer_model`：使用的模型名或本地路径。
- `transformer_ready`：模型是否成功加载。
- `transformer_error`：模型加载失败时的错误信息。
- `source_lang`：多语言模型的源语言 token。
- `decoder_start_token`：多语言模型的解码起始 token。
- `rule_backend_enabled`：本次是否启用了规则后端。
