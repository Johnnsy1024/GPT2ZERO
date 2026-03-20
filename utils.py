import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch


def count_parameters(model: "torch.nn.Module") -> Tuple[int, int]:
    """计算模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def print_model_summary(model: "torch.nn.Module"):
    """打印模型摘要"""
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
            f"{name:50} {str(param.shape):30} {param.numel():10,} {'可训练' if param.requires_grad else '冻结'}"
        )

    print("=" * 80)


def save_model_info(model: "torch.nn.Module", save_path: str):
    """保存模型信息到文件"""
    model_info = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": [],
    }

    for name, param in model.named_parameters():
        layer_info = {
            "name": name,
            "shape": list(param.shape),
            "num_params": param.numel(),
            "trainable": param.requires_grad,
        }
        model_info["layers"].append(layer_info)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"模型信息已保存到: {save_path}")


def calculate_model_size(
    model: "torch.nn.Module", precision_bits: int = 32
) -> Dict[str, float]:
    """计算模型大小

    Args:
        model: PyTorch模型
        precision_bits: 精度位数（32 for float32, 16 for float16, 8 for int8）

    Returns:
        包含不同单位模型大小的字典
    """
    total_params, _ = count_parameters(model)

    # 计算大小（字节）
    if precision_bits == 32:
        bytes_per_param = 4
    elif precision_bits == 16:
        bytes_per_param = 2
    elif precision_bits == 8:
        bytes_per_param = 1
    else:
        raise ValueError(f"不支持的精度位数: {precision_bits}")

    total_bytes = total_params * bytes_per_param

    # 转换为不同单位
    size_info = {
        "parameters": total_params,
        "bytes": total_bytes,
        "kilobytes": total_bytes / 1024,
        "megabytes": total_bytes / (1024**2),
        "gigabytes": total_bytes / (1024**3),
        "precision_bits": precision_bits,
    }

    return size_info


def print_model_size(model: "torch.nn.Module", model_name: str = "模型"):
    """打印模型大小信息"""
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


def create_sample_data(output_dir: str = "./data", num_samples: int = 1000):
    """创建示例训练数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 示例文本
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

    # 生成更多样本
    all_texts = []
    for i in range(num_samples):
        base_text = sample_texts[i % len(sample_texts)]
        # 添加一些变化
        if i % 3 == 0:
            text = f"第{i+1}个示例: {base_text} 这是关于人工智能的一个重要概念。"
        elif i % 3 == 1:
            text = f"让我们讨论一下: {base_text} 这个领域近年来取得了显著进展。"
        else:
            text = f"重要知识点: {base_text} 理解这个概念对于掌握人工智能至关重要。"
        all_texts.append(text)

    # 保存到文件
    output_file = os.path.join(output_dir, "training_data.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")

    print(f"已创建 {len(all_texts)} 个示例文本到: {output_file}")

    # 创建测试数据
    test_file = os.path.join(output_dir, "test_prompts.txt")
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

    with open(test_file, "w", encoding="utf-8") as f:
        for prompt in test_prompts:
            f.write(prompt + "\n")

    print(f"已创建测试提示到: {test_file}")

    return output_file, test_file


def analyze_text_data(file_path: str, max_samples: int = 1000):
    """分析文本数据"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("文件为空")
        return None

    # 限制样本数量
    lines = lines[:max_samples]

    # 计算统计信息
    num_samples = len(lines)
    total_chars = sum(len(line) for line in lines)
    total_words = sum(len(line.split()) for line in lines)

    # 计算平均长度
    avg_chars = total_chars / num_samples
    avg_words = total_words / num_samples

    # 计算长度分布
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
    model, dummy_input, onnx_path, input_names=None, output_names=None
):
    """将PyTorch模型转换为ONNX格式

    Args:
        model: PyTorch模型
        dummy_input: 示例输入
        onnx_path: ONNX文件保存路径
        input_names: 输入名称列表
        output_names: 输出名称列表
    """
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
    except Exception as e:
        print(f"转换到ONNX失败: {e}")
        return False


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")

    # 创建示例数据
    data_file, test_file = create_sample_data(num_samples=100)

    # 分析数据
    stats = analyze_text_data(data_file)

    # 测试模型大小计算
    from config import GPT2Config
    from model import GPT2LMHeadModel

    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)

    print_model_summary(model)
    print_model_size(model, "GPT-2 Small")

    # 保存模型信息
    save_model_info(model, "model_info.json")

    print("\n工具函数测试完成!")
    print("\n工具函数测试完成!")
