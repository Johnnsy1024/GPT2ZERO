import os
import random
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenizer import CharTokenizer, GPT2Tokenizer, SimpleTokenizer
from tokenizer_helpers import encode_training_text


class TextDataset(Dataset):
    """Fixed-length language-modeling samples built from raw text."""

    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = encode_training_text(self.tokenizer, self.texts[idx])
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

        if len(encoded) > self.max_length:
            encoded = encoded[: self.max_length]
        else:
            encoded = encoded + [pad_token_id] * (self.max_length - len(encoded))

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
    """Load line-based or paragraph-based text samples from a file or directory."""
    texts = []

    if os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index >= max_samples:
                    break
                texts.append(line.strip())
    elif os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(data_path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                for paragraph in file.read().split("\n\n"):
                    if paragraph.strip():
                        texts.append(paragraph.strip())
                    if len(texts) >= max_samples:
                        return texts

    return texts


def split_dataset(texts, val_ratio=0.1, seed=42):
    """Split texts into train and validation subsets."""
    texts = list(texts)
    if len(texts) < 2 or val_ratio <= 0:
        return texts, []

    rng = random.Random(seed)
    rng.shuffle(texts)

    val_size = max(1, int(len(texts) * val_ratio))
    val_size = min(val_size, len(texts) - 1)
    return texts[val_size:], texts[:val_size]


def select_training_tokenizer(config, model_type: str, train_texts: Iterable[str]):
    """Create the tokenizer used by train.py without changing current behavior."""
    if model_type == "gpt2-mini":
        tokenizer = CharTokenizer.from_texts(list(train_texts))
        config.vocab_size = tokenizer.vocab_size
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        return tokenizer, f"使用CharTokenizer，词表大小: {tokenizer.vocab_size}"

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        return tokenizer, "使用GPT2Tokenizer"
    except Exception:
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        return tokenizer, "使用SimpleTokenizer"


def train_epoch(model, dataloader, optimizer, device, epoch, grad_clip=1.0):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, device):
    """Evaluate the model on a validation dataloader."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs[0].item()

    return total_loss / max(len(dataloader), 1)


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model, optimizer, and metadata to a checkpoint file."""
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
    """Restore model and optional optimizer state from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"从epoch {epoch}加载检查点，损失: {loss}")
    return epoch, loss


def ensure_example_training_data(data_dir="./data"):
    """Create the sample training file used by train.py if it is missing."""
    os.makedirs(data_dir, exist_ok=True)
    example_file = os.path.join(data_dir, "example.txt")

    if os.path.exists(example_file):
        return example_file, False

    with open(example_file, "w", encoding="utf-8") as file:
        file.write(
            """GPT-2是一个基于Transformer架构的语言模型。
它由OpenAI开发，可以生成连贯的文本。
这个模型有多个版本，包括small、medium、large和xl。
每个版本都有不同的参数数量。
GPT-2在多种自然语言处理任务上表现出色。
它可以用于文本生成、翻译、摘要等任务。
"""
        )

    return example_file, True
