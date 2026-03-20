# GPT-2 快速开始指南

本指南将帮助您在5分钟内开始使用GPT-2实现。

## 1. 环境准备

### 选项A：使用安装脚本（推荐）
```bash
# 运行安装脚本
python setup.py

# 或直接安装依赖
pip install -r requirements.txt
```

### 选项B：手动安装
```bash
# 安装核心依赖
pip install torch transformers numpy tqdm regex

# 创建必要目录
mkdir -p data checkpoints results
```

## 2. 快速测试

### 测试环境
```bash
# 运行环境检查
python test_imports.py
```

### 测试模型
```bash
# 测试模型创建
python -c "
from config import GPT2Config
from model import GPT2LMHeadModel
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)
print(f'✅ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}')
"
```

## 3. 一分钟体验

### 创建示例数据
```bash
# 创建100个示例文本
python -c "
from utils import create_sample_data
create_sample_data(num_samples=100)
"
```

### 快速训练（小规模）
```bash
# 快速训练（1个epoch，小批量）
python train.py --model_type gpt2-mini --epochs 1 --batch_size 2 --max_length 64
```

### 快速生成
```bash
# 使用预训练模型生成文本
python generate.py --model_type gpt2-mini --prompt "人工智能" --max_length 50
```

`gpt2-mini` 会自动在 `checkpoints/tokenizer.json` 保存训练时构建的字符词表，生成阶段默认会读取它。

## 4. 完整使用流程

### 步骤1：准备数据
```bash
# 方法A：使用示例数据
python utils.py

# 方法B：使用自己的数据
# 将文本文件放入 data/ 目录，每行一个样本
```

### 步骤2：训练模型
```bash
# 基础训练（GPT-2 small）
python train.py --model_type gpt2-mini --epochs 1 --batch_size 2 --max_length 64

# 进阶训练（GPT-2 medium，GPU加速）
python train.py --model_type gpt2-medium --device cuda --epochs 5

# 自定义训练
python train.py \
  --data_path ./data \
  --model_type gpt2 \
  --batch_size 4 \
  --epochs 10 \
  --learning_rate 5e-5 \
  --max_length 512 \
  --save_dir ./checkpoints
```

### 步骤3：生成文本
```bash
# 单次生成
python generate.py --model_type gpt2-mini --prompt "今天的天气很好，" --max_length 100

# 交互式生成
python generate.py --interactive

# 批量生成
python generate.py --input_file ./data/test_prompts.txt --output_file ./results.txt
```

## 5. 常用命令示例

### 查看模型信息
```bash
# 查看模型大小
python -c "
from config import GPT2Config
from model import GPT2LMHeadModel
from utils import print_model_size
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)
print_model_size(model)
"
```

### 分析数据
```bash
# 分析训练数据
python -c "
from utils import analyze_text_data
analyze_text_data('./data/training_data.txt')
"
```

### 继续训练
```bash
# 从检查点继续训练
python train.py --checkpoint ./checkpoints/checkpoint_epoch_5.pt
```

## 6. 故障排除

### 问题1：内存不足
```bash
# 减小批次大小和序列长度
python train.py --batch_size 2 --max_length 256
```

### 问题2：训练速度慢
```bash
# 使用GPU加速
python train.py --device cuda

# 或减小模型大小
python train.py --model_type gpt2  # 使用最小的模型
```

### 问题3：生成质量差
```bash
# 调整生成参数
python generate.py --prompt "你的提示" --temperature 0.8 --top_p 0.9

# 增加训练数据量和轮数
python train.py --epochs 20  # 增加训练轮数
```

## 7. 高级功能

### 模型导出
```python
# 将模型导出为ONNX格式（在Python中运行）
from utils import convert_model_to_onnx
from config import GPT2Config
from model import GPT2LMHeadModel
import torch

config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)
dummy_input = (torch.randint(0, config.vocab_size, (1, 10)), 
               torch.ones(1, 10, dtype=torch.long))
convert_model_to_onnx(model, dummy_input, 'gpt2.onnx')
```

### 自定义配置
```python
# 创建自定义配置
from config import GPT2Config

# 小型模型（用于测试）
small_config = GPT2Config(
    vocab_size=10000,
    n_positions=512,
    n_embd=256,
    n_layer=6,
    n_head=8,
    n_inner=1024
)
```

## 8. 下一步

1. **阅读完整文档**: 查看 [README.md](README.md) 获取详细说明
2. **探索代码**: 查看各个Python文件了解实现细节
3. **调整参数**: 尝试不同的模型配置和训练参数
4. **使用自己的数据**: 用您的数据训练定制化模型
5. **集成到项目**: 将GPT-2模型集成到您的应用中

## 9. 获取帮助

- 查看代码注释和文档字符串
- 运行 `python train.py --help` 查看所有选项
- 检查 `test_imports.py` 的输出解决环境问题
- 参考原始论文和Transformer相关资料

---

**提示**: `gpt2-mini` 是项目内置的快速验证配置，适合本机 CPU 做 smoke test。首次运行标准 GPT-2 分词器时可能需要联网下载分词器数据。

**开始您的GPT-2之旅吧！** 🚀
