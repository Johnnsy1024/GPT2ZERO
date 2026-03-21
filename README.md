# GPT2ZERO

一个从零实现的 PyTorch GPT-2 练手项目，包含模型定义、训练入口、文本生成入口和少量工程辅助脚本。这个仓库的目标不是复刻完整 Hugging Face 生态，而是提供一套清楚、可运行、方便继续改造的最小工程骨架。

## 这个项目现在有什么

- 支持 `gpt2-mini`、`gpt2`、`gpt2-medium`、`gpt2-large`、`gpt2-xl` 五种配置
- `gpt2-mini` 可直接在本地 CPU 上做 smoke test
- `train.py` / `generate.py` 保持为薄入口，训练与生成逻辑拆到 pipeline 文件
- `gpt2-mini` 训练时会保存字符级词表，生成阶段可直接复用
- 支持单条生成、交互式生成、文件批量生成
- 内置示例数据、自检脚本和模型信息导出工具
- 支持导出 ONNX

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你只是想一键准备环境，也可以运行：

```bash
python setup.py
```

### 2. 创建示例数据

```bash
python utils.py
```

这会在 `data/` 下创建：

- `training_data.txt`
- `test_prompts.txt`

### 3. 跑一轮最小训练

```bash
python train.py --model_type gpt2-mini --epochs 1 --batch_size 2 --max_length 64
```

### 4. 生成文本

```bash
python generate.py --model_type gpt2-mini --prompt "人工智能的未来"
```

批量生成默认输出到 `results/generated_texts.txt`：

```bash
python generate.py --input_file ./data/test_prompts.txt
```

## 项目结构

```text
GPT2ZERO/
├── config.py               # GPT-2 配置定义
├── model.py                # Transformer 与 LM Head 实现
├── tokenizer.py            # GPT-2 / simple / char 三套分词器
├── tokenizer_helpers.py    # 训练与生成共用的编码/解码兼容层
├── training_pipeline.py    # 数据集、切分、训练、评估、检查点逻辑
├── generation_pipeline.py  # 模型加载、批量/交互生成逻辑
├── train.py                # 训练 CLI 入口
├── generate.py             # 生成 CLI 入口
├── utils.py                # 示例数据、数据分析、模型信息导出
├── test_imports.py         # 环境与模块导入检查
├── setup.py                # 一键 bootstrap 脚本
├── data/                   # 示例训练数据
├── checkpoints/            # 训练产物（已忽略）
└── results/                # 生成结果（已忽略）
```

## 训练说明

### 常用命令

```bash
# 默认用 gpt2-mini 训练
python train.py

# 指定数据文件
python train.py --data_path ./data/training_data.txt

# 指定更大的模型
python train.py --model_type gpt2-medium --batch_size 2 --epochs 5

# 从检查点继续训练
python train.py --checkpoint ./checkpoints/checkpoint_epoch_3.pt
```

### 数据格式

- 传入单个 `.txt` 文件时，按行读取，每行视为一个样本
- 传入目录时，会读取目录下所有 `.txt` 文件，并按空行分段

### 词表策略

- `gpt2-mini`：从训练语料构建字符级词表，并保存到 `checkpoints/tokenizer.json`
- 其他模型：优先尝试本地 `transformers` GPT-2 分词器
- 如果本地标准分词器不可用，则回退到简化分词器

## 生成说明

### 单条生成

```bash
python generate.py \
  --model_type gpt2-mini \
  --prompt "Transformer 架构的核心思想是" \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_length 120
```

### 交互式生成

```bash
python generate.py --interactive
```

### 批量生成

```bash
python generate.py \
  --input_file ./data/test_prompts.txt \
  --output_file ./results/demo.txt
```

常用参数：

- `--temperature`：控制随机性
- `--top_k`：限制候选 token 数量
- `--top_p`：使用 nucleus sampling
- `--no_sample`：关闭采样，使用贪婪解码
- `--num_return_sequences`：一次生成多条结果

## 工具脚本

`utils.py` 现在是一个明确的工具入口，不再在默认情况下生成额外的大模型信息文件。

```bash
# 默认行为：创建示例数据
python utils.py

# 显式创建示例数据
python utils.py create-sample-data --num-samples 200 --overwrite

# 分析训练文本
python utils.py analyze-data ./data/training_data.txt

# 导出模型参数信息
python utils.py model-info --model-type gpt2-mini --output ./results/model_info.json
```

环境检查：

```bash
python test_imports.py
python test_imports.py --create-sample-data
```

## 已做的工程约束

- `checkpoints/`、`results/`、`model_info.json` 等产物默认不进 git
- 批量生成默认写入 `results/`，避免根目录堆积临时文件
- 自检脚本默认只检查，不再擅自修改仓库
- 训练与生成入口脚本只保留参数解析和流程编排

## 常见问题

### 1. `transformers` 分词器加载失败

项目使用 `local_files_only=True` 加载 GPT-2 分词器。如果你本机没有缓存，标准 GPT-2 分词器不可用，但 `gpt2-mini` 路径仍可正常使用。

### 2. `max_length` 太大

`train.py` 会自动把 `max_length` 限制到模型的上下文长度上限，并打印提示。

### 3. CPU 训练太慢

先用 `gpt2-mini` 做验证：

```bash
python train.py --model_type gpt2-mini --epochs 1 --batch_size 2 --max_length 64
```

## 下一步建议

如果你准备继续把它当成长期项目，可以优先考虑这几件事：

- 增加单元测试，而不只是导入检查
- 给训练和生成补充更稳定的配置管理
- 把日志与实验记录独立到专门目录
- 为更大模型补充显存占用与训练建议
