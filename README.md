# GPT-2 实现

这是一个完整的GPT-2（Generative Pre-trained Transformer 2）实现，使用PyTorch从头开始构建。该项目包含了GPT-2模型架构、训练脚本、文本生成工具以及相关的实用函数。

## 功能特性

- ✅ 完整的GPT-2模型架构实现（支持small、medium、large、xl版本）
- ✅ 基于Transformer的解码器架构
- ✅ 多头自注意力机制
- ✅ 位置编码和前馈网络
- ✅ 文本生成功能（支持温度采样、top-k、top-p）
- ✅ 训练脚本和数据处理工具
- ✅ 交互式文本生成界面
- ✅ 模型评估和可视化工具
- ✅ 支持ONNX模型导出

## 安装要求

### 系统要求
- Python 3.8+
- PyTorch 1.9+
- CUDA（可选，用于GPU加速）

### 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 或者手动安装
pip install torch transformers numpy tqdm datasets regex
```

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd gpt2-implementation
```

### 2. 创建示例数据
```bash
python utils.py
```

### 3. 训练模型
```bash
# 使用默认参数训练GPT-2 small
python train.py

# 训练GPT-2 medium
python train.py --model_type gpt2-medium --batch_size 2 --epochs 5

# 使用GPU训练
python train.py --device cuda
```

### 4. 生成文本
```bash
# 使用提示生成文本
python generate.py --prompt "人工智能的未来"

# 交互式生成
python generate.py --interactive

# 批量生成
python generate.py --input_file ./data/test_prompts.txt --output_file ./results.txt
```

## 项目结构

```
gpt2-implementation/
├── config.py              # 模型配置类
├── model.py              # GPT-2模型实现
├── tokenizer.py          # 分词器实现
├── train.py              # 训练脚本
├── generate.py           # 文本生成脚本
├── utils.py              # 工具函数
├── requirements.txt      # 依赖列表
├── README.md            # 项目说明
├── data/                # 数据目录
│   ├── training_data.txt
│   └── test_prompts.txt
└── checkpoints/         # 模型检查点
    ├── config.json
    └── final_model.pt
```

## 详细使用指南

### 模型配置

GPT-2有四个预定义配置：

| 模型类型 | 参数量 | 层数 | 隐藏维度 | 注意力头数 |
|---------|--------|------|----------|------------|
| gpt2 | 124M | 12 | 768 | 12 |
| gpt2-medium | 355M | 24 | 1024 | 16 |
| gpt2-large | 774M | 36 | 1280 | 20 |
| gpt2-xl | 1.5B | 48 | 1600 | 25 |

### 训练模型

#### 基本训练
```bash
python train.py \
  --data_path ./data \
  --model_type gpt2 \
  --batch_size 4 \
  --epochs 10 \
  --learning_rate 5e-5 \
  --max_length 512 \
  --save_dir ./checkpoints
```

#### 高级选项
```bash
# 继续训练
python train.py --checkpoint ./checkpoints/checkpoint_epoch_5.pt

# 使用自定义数据
python train.py --data_path /path/to/your/data.txt

# 调整训练参数
python train.py --batch_size 8 --learning_rate 3e-5 --max_length 1024
```

### 文本生成

#### 生成参数说明
- `--temperature`: 温度参数（0.1-2.0），值越高越随机
- `--top_k`: top-k采样参数，限制候选token数量
- `--top_p`: top-p（核）采样参数，限制累积概率
- `--do_sample`: 是否使用采样（否则使用贪婪解码）
- `--max_length`: 最大生成长度

#### 示例
```bash
# 创意写作
python generate.py \
  --prompt "在一个遥远的未来，人类发现了时间旅行的秘密。" \
  --temperature 0.9 \
  --top_p 0.95 \
  --max_length 200

# 技术文档
python generate.py \
  --prompt "Transformer架构的核心思想是" \
  --temperature 0.7 \
  --top_k 30 \
  --max_length 150

# 代码生成
python generate.py \
  --prompt "def calculate_fibonacci(n):" \
  --temperature 0.8 \
  --max_length 100
```

### 交互式模式

启动交互式生成界面：
```bash
python generate.py --interactive
```

在交互式模式中，您可以：
- 输入提示文本生成内容
- 实时调整生成参数
- 查看生成历史
- 导出生成结果

## 模型架构

### 核心组件

1. **词嵌入层**：将token ID转换为向量表示
2. **位置编码**：添加位置信息到词嵌入
3. **Transformer块**：
   - 层归一化
   - 多头自注意力机制
   - 前馈网络（MLP）
   - 残差连接
4. **语言模型头部**：将隐藏状态转换为词汇表概率分布

### 注意力机制

GPT-2使用因果自注意力机制，确保每个位置只能关注之前的位置：
```python
# 因果掩码
mask = torch.tril(torch.ones(seq_len, seq_len))
```

### 前馈网络

每个Transformer块包含一个两层MLP：
```python
MLP(x) = GELU(xW₁ + b₁)W₂ + b₂
```

## API参考

### 模型类

```python
from config import GPT2Config
from model import GPT2LMHeadModel

# 加载配置
config = GPT2Config.from_pretrained("gpt2")

# 创建模型
model = GPT2LMHeadModel(config)

# 前向传播
outputs = model(input_ids, attention_mask, labels)
loss, logits = outputs[0], outputs[1]

# 生成文本
generated = model.generate(input_ids, max_length=100, temperature=0.8)
```

### 分词器

```python
from tokenizer import GPT2Tokenizer

# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 编码文本
input_ids = tokenizer.encode("Hello, world!", add_special_tokens=True)

# 解码文本
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

## 示例代码

### 完整训练流程
```python
import torch
from config import GPT2Config
from model import GPT2LMHeadModel
from tokenizer import GPT2Tokenizer

# 1. 准备配置和模型
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 2. 准备数据
texts = ["示例文本1", "示例文本2", "示例文本3"]
input_ids = [tokenizer.encode(text) for text in texts]

# 3. 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for epoch in range(10):
    for batch in data_loader:
        loss = model(input_ids, labels=input_ids)[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 4. 生成文本
prompt = "人工智能的未来是"
input_ids = tokenizer.encode(prompt)
generated = model.generate(input_ids, max_length=100)
output_text = tokenizer.decode(generated[0])
print(output_text)
```

### 自定义训练数据
```python
from utils import create_sample_data, analyze_text_data

# 创建训练数据
data_file, test_file = create_sample_data(num_samples=10000)

# 分析数据
stats = analyze_text_data(data_file)

# 使用自定义数据训练
# python train.py --data_path /path/to/your/data.txt
```

## 性能优化

### GPU加速
```bash
# 使用CUDA
python train.py --device cuda

# 使用特定GPU
CUDA_VISIBLE_DEVICES=0 python train.py --device cuda
```

### 混合精度训练
```python
# 在train.py中添加
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(input_ids, labels=input_ids)[0]
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 模型量化
```python
# 动态量化
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## 常见问题

### 1. 内存不足
- 减小`batch_size`
- 减小`max_length`
- 使用梯度累积
- 启用混合精度训练

### 2. 训练速度慢
- 使用GPU加速
- 增大`batch_size`（在内存允许的情况下）
- 使用数据并行
- 优化数据加载

### 3. 生成质量差
- 调整温度参数（通常0.7-0.9效果较好）
- 使用top-p采样（0.9-0.95）
- 增加训练数据量
- 增加训练轮数

### 4. 分词器问题
- 确保安装了`transformers`库
- 检查网络连接（首次使用会下载预训练分词器）
- 使用`SimpleTokenizer`作为备选

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献指南

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 致谢

- OpenAI发布了原始的GPT-2模型和论文
- Hugging Face的Transformers库提供了参考实现
- PyTorch团队提供了优秀的深度学习框架

## 参考文献

1. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"
2. Vaswani, A., et al. (2017). "Attention Is All You Need"
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners"

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 完整的GPT-2模型实现
- 训练和生成脚本
- 交互式界面
- 工具函数和示例数据

---

**注意**: 这是一个教育目的的实现，可能与原始GPT-2实现存在差异。对于生产环境，建议使用Hugging Face的Transformers库。
```

## 下一步

1. 运行 `python utils.py` 创建示例数据
2. 运行 `python train.py` 开始训练
3. 运行 `python generate.py --interactive` 体验文本生成

祝您使用愉快！
