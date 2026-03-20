import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import GPT2Config
from model import GPT2LMHeadModel
from tokenizer import CharTokenizer, GPT2Tokenizer, SimpleTokenizer


class TextDataset(Dataset):
    """文本数据集"""

    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 编码文本
        if isinstance(self.tokenizer, GPT2Tokenizer):
            try:
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
            except:
                # 如果GPT2Tokenizer失败，使用简单编码
                encoded = (
                    [self.tokenizer.bos_token_id]
                    + [ord(c) % 1000 + 10 for c in text]
                    + [self.tokenizer.eos_token_id]
                )
        else:
            encoded = self.tokenizer.encode(text, add_special_tokens=True)

        # 截断或填充到最大长度
        pad_token_id = (
            self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        )
        if len(encoded) > self.max_length:
            encoded = encoded[: self.max_length]
        else:
            encoded = encoded + [pad_token_id] * (self.max_length - len(encoded))

        # 创建注意力掩码
        attention_mask = [1 if token_id != pad_token_id else 0 for token_id in encoded]
        labels = [
            token_id if mask == 1 else -100
            for token_id, mask in zip(encoded, attention_mask)
        ]

        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_dataset(data_path, max_samples=10000):
    """加载数据集"""
    texts = []

    if os.path.isfile(data_path):
        # 如果是文件，按行读取
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                texts.append(line.strip())
    elif os.path.isdir(data_path):
        # 如果是目录，读取所有txt文件
        for filename in os.listdir(data_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(data_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 按段落分割
                    paragraphs = content.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            texts.append(para.strip())
                            if len(texts) >= max_samples:
                                return texts

    return texts


def split_dataset(texts, val_ratio=0.1, seed=42):
    """划分训练集和验证集"""
    texts = list(texts)
    if len(texts) < 2 or val_ratio <= 0:
        return texts, []

    rng = random.Random(seed)
    rng.shuffle(texts)
    val_size = max(1, int(len(texts) * val_ratio))
    val_size = min(val_size, len(texts) - 1)
    return texts[val_size:], texts[:val_size]


def train_epoch(model, dataloader, optimizer, device, epoch, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # 移动到设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # 更新进度条
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs[0]
            total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": (
            model.config.to_dict()
            if hasattr(model.config, "to_dict")
            else model.config.__dict__
        ),
    }

    torch.save(checkpoint, save_path)
    print(f"检查点已保存到: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"从epoch {epoch}加载检查点，损失: {loss}")
    return epoch, loss


def main():
    parser = argparse.ArgumentParser(description="训练GPT-2模型")
    parser.add_argument(
        "--data_path", type=str, default="./data", help="数据路径（文件或目录）"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2-mini",
        choices=["gpt2-mini", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="模型类型",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="保存目录")
    parser.add_argument("--checkpoint", type=str, help="继续训练的检查点路径")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载配置
    config = GPT2Config.from_pretrained(args.model_type)
    print(f"加载 {args.model_type} 配置")

    # 加载数据集
    print(f"从 {args.data_path} 加载数据集...")
    texts = load_dataset(args.data_path)
    print(f"加载了 {len(texts)} 个文本样本")
    if not texts:
        raise ValueError(
            f"在 {args.data_path} 中没有找到可训练文本，请先准备 .txt 数据文件。"
        )

    train_texts, val_texts = split_dataset(texts, val_ratio=args.val_ratio, seed=args.seed)
    print(f"训练集样本数: {len(train_texts)}")
    print(f"验证集样本数: {len(val_texts)}")

    if args.model_type == "gpt2-mini":
        tokenizer = CharTokenizer.from_texts(train_texts)
        config.vocab_size = tokenizer.vocab_size
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        print(f"使用CharTokenizer，词表大小: {tokenizer.vocab_size}")
    else:
        if args.model_type == "gpt2-mini":
            tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
            print("使用SimpleTokenizer")
        else:
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
                print("使用GPT2Tokenizer")
            except Exception:
                tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
                print("使用SimpleTokenizer")

    # 初始化模型
    model = GPT2LMHeadModel(config)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 创建数据集和数据加载器
    effective_max_length = min(args.max_length, config.n_positions)
    if effective_max_length != args.max_length:
        print(
            f"警告: max_length={args.max_length} 超过模型上下文长度 {config.n_positions}，"
            f"已自动调整为 {effective_max_length}"
        )

    train_dataset = TextDataset(train_texts, tokenizer, max_length=effective_max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataloader = None
    if val_texts:
        val_dataset = TextDataset(val_texts, tokenizer, max_length=effective_max_length)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 3e-4 if args.model_type == "gpt2-mini" else 5e-5
    print(f"学习率: {learning_rate}")

    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    tokenizer_path = os.path.join(args.save_dir, "tokenizer.json")
    if hasattr(tokenizer, "save"):
        tokenizer.save(tokenizer_path)
        print(f"分词器已保存到: {tokenizer_path}")

    # 加载检查点（如果提供）
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"从 {args.checkpoint} 加载检查点...")
        start_epoch, _ = load_checkpoint(args.checkpoint, model, optimizer, device)

    # 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch + 1)
        print(f"训练损失: {train_loss:.4f}")

        # 保存检查点
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_path)

        # 保存模型配置
        config_path = os.path.join(args.save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        if val_dataloader is not None:
            eval_loss = evaluate(model, val_dataloader, device)
            print(f"验证损失: {eval_loss:.4f}")

    print("\n训练完成!")

    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")


if __name__ == "__main__":
    # 创建示例数据目录（如果不存在）
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    # 创建示例数据文件（如果不存在）
    example_file = os.path.join(data_dir, "example.txt")
    if not os.path.exists(example_file):
        with open(example_file, "w", encoding="utf-8") as f:
            f.write(
                """GPT-2是一个基于Transformer架构的语言模型。
它由OpenAI开发，可以生成连贯的文本。
这个模型有多个版本，包括small、medium、large和xl。
每个版本都有不同的参数数量。
GPT-2在多种自然语言处理任务上表现出色。
它可以用于文本生成、翻译、摘要等任务。
"""
            )
        print(f"已创建示例数据文件: {example_file}")

    # 运行训练
    main()
