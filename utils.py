import argparse
import json
import os
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    import torch


MODEL_CHOICES = ["gpt2-mini", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def count_parameters(model: "torch.nn.Module") -> Tuple[int, int]:
    """计算模型参数数量。"""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    return total_params, trainable_params


def print_model_summary(model: "torch.nn.Module"):
    """打印逐层参数摘要。"""
    print("=" * 80)
    print("模型摘要")
    print("=" * 80)

    total_params, trainable_params = count_parameters(model)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")

    print("\n各层参数:")
    print("-" * 80)
    for name, param in model.named_parameters():
        print(
            f"{name:50} {str(param.shape):30} {param.numel():10,} "
            f"{'可训练' if param.requires_grad else '冻结'}"
        )

    print("=" * 80)


def save_model_info(model: "torch.nn.Module", save_path: str):
    """保存模型参数信息到 JSON 文件。"""
    model_info = {
        "total_params": sum(param.numel() for param in model.parameters()),
        "trainable_params": sum(
            param.numel() for param in model.parameters() if param.requires_grad
        ),
        "layers": [],
    }

    for name, param in model.named_parameters():
        model_info["layers"].append(
            {
                "name": name,
                "shape": list(param.shape),
                "num_params": param.numel(),
                "trainable": param.requires_grad,
            }
        )

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2, ensure_ascii=False)

    print(f"模型信息已保存到: {save_path}")


def calculate_model_size(
    model: "torch.nn.Module", precision_bits: int = 32
) -> Dict[str, float]:
    """计算不同精度下的模型占用。"""
    total_params, _ = count_parameters(model)

    if precision_bits == 32:
        bytes_per_param = 4
    elif precision_bits == 16:
        bytes_per_param = 2
    elif precision_bits == 8:
        bytes_per_param = 1
    else:
        raise ValueError(f"不支持的精度位数: {precision_bits}")

    total_bytes = total_params * bytes_per_param
    return {
        "parameters": total_params,
        "bytes": total_bytes,
        "kilobytes": total_bytes / 1024,
        "megabytes": total_bytes / (1024**2),
        "gigabytes": total_bytes / (1024**3),
        "precision_bits": precision_bits,
    }


def print_model_size(model: "torch.nn.Module", model_name: str = "模型"):
    """打印不同精度下的模型大小。"""
    size_info_32 = calculate_model_size(model, precision_bits=32)
    size_info_16 = calculate_model_size(model, precision_bits=16)
    size_info_8 = calculate_model_size(model, precision_bits=8)

    print(f"\n{model_name} 大小信息:")
    print("-" * 60)
    print(f"参数量: {size_info_32['parameters']:,}")
    print("\n不同精度下的模型大小:")
    print(f"  float32: {size_info_32['megabytes']:.2f} MB")
    print(f"  float16: {size_info_16['megabytes']:.2f} MB")
    print(f"  int8:    {size_info_8['megabytes']:.2f} MB")
    print("-" * 60)


def create_sample_data(
    output_dir: str = "./data",
    num_samples: int = 1000,
    overwrite: bool = False,
):
    """创建示例训练数据与测试提示。

    默认在目标文件已存在时不覆盖，避免无意修改仓库内的示例数据。
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "training_data.txt")
    test_file = os.path.join(output_dir, "test_prompts.txt")

    if not overwrite and os.path.exists(output_file) and os.path.exists(test_file):
        print(f"示例数据已存在，跳过覆盖: {output_file}")
        print(f"测试提示已存在，跳过覆盖: {test_file}")
        return output_file, test_file

    sample_texts = [
        "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。",
        "机器学习是人工智能的一个子领域，使计算机能够从数据中学习而无需明确编程。",
        "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的复杂模式。",
        "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
        "计算机视觉是人工智能的一个领域，使计算机能够从数字图像或视频中获取高级理解。",
        "强化学习是机器学习的一个领域，智能体通过与环境交互来学习如何实现目标。",
        "Transformer是一种神经网络架构，广泛应用于自然语言处理任务。",
        "GPT（生成式预训练Transformer）是由OpenAI开发的一系列语言模型。",
        "BERT（双向编码器表示来自Transformer）是由Google开发的语言表示模型。",
        "神经网络是由相互连接的节点组成的计算系统，灵感来自人脑的生物神经网络。",
    ]

    all_texts = []
    for index in range(num_samples):
        base_text = sample_texts[index % len(sample_texts)]
        if index % 3 == 0:
            text = f"第{index + 1}个示例: {base_text} 这是关于人工智能的一个重要概念。"
        elif index % 3 == 1:
            text = f"让我们讨论一下: {base_text} 这个领域近年来取得了显著进展。"
        else:
            text = f"重要知识点: {base_text} 理解这个概念对于掌握人工智能至关重要。"
        all_texts.append(text)

    with open(output_file, "w", encoding="utf-8") as file:
        for text in all_texts:
            file.write(text + "\n")

    print(f"已创建 {len(all_texts)} 个示例文本到: {output_file}")

    test_prompts = [
        "人工智能的未来",
        "机器学习如何工作",
        "深度学习的应用",
        "自然语言处理的挑战",
        "计算机视觉的发展",
        "强化学习的原理",
        "Transformer架构的优势",
        "GPT模型的特点",
        "BERT模型的创新",
        "神经网络的基础",
    ]

    with open(test_file, "w", encoding="utf-8") as file:
        for prompt in test_prompts:
            file.write(prompt + "\n")

    print(f"已创建测试提示到: {test_file}")
    return output_file, test_file


def analyze_text_data(file_path: str, max_samples: int = 1000):
    """分析文本文件的样本规模与长度分布。"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        print("文件为空")
        return None

    lines = lines[:max_samples]
    num_samples = len(lines)
    total_chars = sum(len(line) for line in lines)
    total_words = sum(len(line.split()) for line in lines)
    avg_chars = total_chars / num_samples
    avg_words = total_words / num_samples
    char_lengths = [len(line) for line in lines]
    word_lengths = [len(line.split()) for line in lines]

    stats = {
        "num_samples": num_samples,
        "total_chars": total_chars,
        "total_words": total_words,
        "avg_chars": avg_chars,
        "avg_words": avg_words,
        "max_chars": max(char_lengths),
        "min_chars": min(char_lengths),
        "max_words": max(word_lengths),
        "min_words": min(word_lengths),
        "char_lengths": char_lengths,
        "word_lengths": word_lengths,
    }

    print(f"\n文本数据分析 ({file_path}):")
    print("-" * 60)
    print(f"样本数量: {stats['num_samples']:,}")
    print(f"总字符数: {stats['total_chars']:,}")
    print(f"总词数: {stats['total_words']:,}")
    print(f"平均字符数: {stats['avg_chars']:.1f}")
    print(f"平均词数: {stats['avg_words']:.1f}")
    print(f"最大字符数: {stats['max_chars']}")
    print(f"最小字符数: {stats['min_chars']}")
    print(f"最大词数: {stats['max_words']}")
    print(f"最小词数: {stats['min_words']}")
    print("-" * 60)

    return stats


def convert_model_to_onnx(
    model,
    dummy_input,
    onnx_path,
    input_names=None,
    output_names=None,
):
    """将 PyTorch 模型导出为 ONNX。"""
    if input_names is None:
        input_names = ["input_ids", "attention_mask"]
    if output_names is None:
        output_names = ["logits"]

    try:
        import torch

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"ONNX模型已保存到: {onnx_path}")
        return True
    except Exception as error:
        print(f"转换到ONNX失败: {error}")
        return False


def build_arg_parser():
    """构建 utils.py 的命令行入口。"""
    parser = argparse.ArgumentParser(description="GPT2ZERO 工具脚本")
    subparsers = parser.add_subparsers(dest="command")

    sample_parser = subparsers.add_parser(
        "create-sample-data", help="创建示例训练数据与测试提示"
    )
    sample_parser.add_argument(
        "--output-dir", type=str, default="./data", help="输出目录"
    )
    sample_parser.add_argument(
        "--num-samples", type=int, default=1000, help="训练样本数量"
    )
    sample_parser.add_argument(
        "--overwrite", action="store_true", help="覆盖现有示例数据文件"
    )

    analyze_parser = subparsers.add_parser("analyze-data", help="分析文本数据文件")
    analyze_parser.add_argument(
        "file_path",
        nargs="?",
        default="./data/training_data.txt",
        help="待分析的文本文件",
    )
    analyze_parser.add_argument(
        "--max-samples", type=int, default=1000, help="最多读取的样本数"
    )

    model_info_parser = subparsers.add_parser(
        "model-info", help="导出模型参数信息到 JSON 文件"
    )
    model_info_parser.add_argument(
        "--model-type",
        choices=MODEL_CHOICES,
        default="gpt2",
        help="模型配置类型",
    )
    model_info_parser.add_argument(
        "--output", type=str, default="model_info.json", help="输出文件路径"
    )
    model_info_parser.add_argument(
        "--print-summary", action="store_true", help="同时打印逐层参数摘要"
    )

    return parser


def export_model_info(model_type: str, output_path: str, print_summary: bool = False):
    """按需构建模型并导出参数统计。"""
    from config import GPT2Config
    from model import GPT2LMHeadModel

    config = GPT2Config.from_pretrained(model_type)
    model = GPT2LMHeadModel(config)

    if print_summary:
        print_model_summary(model)

    print_model_size(model, model_type)
    save_model_info(model, output_path)
    return output_path


def main():
    """命令行入口。

    默认行为保持为“创建示例数据”，与 README/QUICKSTART 的快速体验一致。
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command in (None, "create-sample-data"):
        output_dir = getattr(args, "output_dir", "./data")
        num_samples = getattr(args, "num_samples", 1000)
        overwrite = getattr(args, "overwrite", False)
        create_sample_data(
            output_dir=output_dir,
            num_samples=num_samples,
            overwrite=overwrite,
        )
        return 0

    if args.command == "analyze-data":
        stats = analyze_text_data(args.file_path, max_samples=args.max_samples)
        return 0 if stats is not None else 1

    if args.command == "model-info":
        export_model_info(
            model_type=args.model_type,
            output_path=args.output,
            print_summary=args.print_summary,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
