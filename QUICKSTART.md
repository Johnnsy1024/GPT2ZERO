# Quick Start

5 分钟内跑通这个项目，按下面顺序即可。

## 1. 准备环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果想一步完成初始化，也可以：

```bash
python setup.py
```

## 2. 检查环境

```bash
python test_imports.py
```

## 3. 生成示例数据

```bash
python utils.py
```

这一步会在 `data/` 下创建训练样本和测试提示。

## 4. 训练一个最小模型

```bash
python train.py --model_type gpt2-mini --epochs 1 --batch_size 2 --max_length 64
```

如果你只是想确认链路没问题，这是最稳妥的起点。

## 5. 生成文本

```bash
python generate.py --model_type gpt2-mini --prompt "人工智能的未来"
```

批量生成：

```bash
python generate.py --input_file ./data/test_prompts.txt
```

默认结果会写到 `results/generated_texts.txt`。

## 常用命令

```bash
# 继续训练
python train.py --checkpoint ./checkpoints/checkpoint_epoch_1.pt

# 交互式生成
python generate.py --interactive

# 分析数据
python utils.py analyze-data ./data/training_data.txt

# 导出模型参数信息
python utils.py model-info --model-type gpt2-mini --output ./results/model_info.json

# 覆盖现有示例数据
python utils.py create-sample-data --num-samples 200 --overwrite
```

## 两个注意点

### `gpt2-mini` 和其他模型不一样

- `gpt2-mini` 会根据当前训练语料构建字符级词表
- 其他模型优先尝试读取本地 `transformers` GPT-2 分词器

所以最适合本地快速验证的就是 `gpt2-mini`。

### 结果文件默认不进 git

仓库已经忽略了这些目录和文件：

- `checkpoints/`
- `results/`
- `model_info.json`

这意味着你可以反复训练和生成，而不会把产物堆进版本控制。

## 如果出问题

### 导入失败

```bash
pip install -r requirements.txt
python test_imports.py
```

### 训练太慢

```bash
python train.py --model_type gpt2-mini --epochs 1 --batch_size 2 --max_length 32
```

### 生成效果差

- 增加训练轮数
- 增加训练数据量
- 调整 `--temperature`、`--top_k`、`--top_p`

更完整的说明见 `README.md`。
