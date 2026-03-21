import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from config import GPT2Config
from model import GPT2LMHeadModel
from training_pipeline import (
    TextDataset,
    ensure_example_training_data,
    evaluate,
    load_checkpoint,
    load_dataset,
    save_checkpoint,
    select_training_tokenizer,
    split_dataset,
    train_epoch,
)


def build_arg_parser():
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
    return parser


def resolve_learning_rate(model_type: str, learning_rate):
    if learning_rate is not None:
        return learning_rate
    return 3e-4 if model_type == "gpt2-mini" else 5e-5


def resolve_effective_max_length(requested_max_length: int, config) -> int:
    effective_max_length = min(requested_max_length, config.n_positions)
    if effective_max_length != requested_max_length:
        print(
            f"警告: max_length={requested_max_length} 超过模型上下文长度 {config.n_positions}，"
            f"已自动调整为 {effective_max_length}"
        )
    return effective_max_length


def create_dataloaders(train_texts, val_texts, tokenizer, max_length, batch_size):
    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = None
    if val_texts:
        val_dataset = TextDataset(val_texts, tokenizer, max_length=max_length)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def save_training_state(save_dir: str, model, optimizer, epoch: int, loss: float):
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    save_checkpoint(model, optimizer, epoch, loss, checkpoint_path)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(model.config.to_dict(), file, indent=2)


def save_tokenizer_if_possible(tokenizer, save_dir: str):
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    if hasattr(tokenizer, "save"):
        tokenizer.save(tokenizer_path)
        print(f"分词器已保存到: {tokenizer_path}")


def main():
    args = build_arg_parser().parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    config = GPT2Config.from_pretrained(args.model_type)
    print(f"加载 {args.model_type} 配置")

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

    tokenizer, tokenizer_message = select_training_tokenizer(
        config, args.model_type, train_texts
    )
    print(tokenizer_message)

    model = GPT2LMHeadModel(config)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    effective_max_length = resolve_effective_max_length(args.max_length, config)
    train_dataloader, val_dataloader = create_dataloaders(
        train_texts,
        val_texts,
        tokenizer,
        effective_max_length,
        args.batch_size,
    )

    learning_rate = resolve_learning_rate(args.model_type, args.learning_rate)
    print(f"学习率: {learning_rate}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    save_tokenizer_if_possible(tokenizer, args.save_dir)

    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"从 {args.checkpoint} 加载检查点...")
        start_epoch, _ = load_checkpoint(args.checkpoint, model, optimizer, device)

    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch + 1)
        print(f"训练损失: {train_loss:.4f}")

        save_training_state(args.save_dir, model, optimizer, epoch + 1, train_loss)

        if val_dataloader is not None:
            eval_loss = evaluate(model, val_dataloader, device)
            print(f"验证损失: {eval_loss:.4f}")

    print("\n训练完成!")

    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")


if __name__ == "__main__":
    example_file, created = ensure_example_training_data("./data")
    if created:
        print(f"已创建示例数据文件: {example_file}")

    main()
