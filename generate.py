import argparse
import os

import torch

from generation_pipeline import (
    batch_mode,
    generate_text,
    interactive_mode,
    load_generation_tokenizer,
    load_model,
    save_single_prompt_results,
    strip_prompt_prefix,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="GPT-2 文本生成")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/final_model.pt",
        help="模型路径",
    )
    parser.add_argument(
        "--config_path", type=str, default="./checkpoints/config.json", help="配置路径"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="./checkpoints/tokenizer.json", help="分词器路径"
    )
    parser.add_argument(
        "--model_type", type=str, default="gpt2-mini", help="模型类型（用于分词器）"
    )
    parser.add_argument("--prompt", type=str, help="提示文本")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="top-k参数")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p参数")
    parser.add_argument("--do_sample", action="store_true", default=True, help="使用采样")
    parser.add_argument(
        "--no_sample",
        action="store_false",
        dest="do_sample",
        help="不使用采样（贪婪解码）",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="生成数量")
    parser.add_argument("--input_file", type=str, help="输入文件（批量模式）")
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/generated_texts.txt",
        help="输出文件（单条/批量模式）",
    )
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )
    return parser


def build_generation_params(args):
    return {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "num_return_sequences": args.num_return_sequences,
    }


def print_single_prompt_results(prompt: str, generated_texts, num_return_sequences: int):
    print("\n" + "=" * 60)
    for index, text in enumerate(generated_texts):
        if num_return_sequences > 1:
            print(f"\n生成结果 {index+1}:")
            print("-" * 40)

        print(strip_prompt_prefix(prompt, text))

    print("=" * 60)


def main():
    args = build_arg_parser().parse_args()

    if not os.path.exists(args.model_path):
        print(f"模型文件不存在: {args.model_path}")
        print("请先运行 train.py 训练模型，或指定正确的模型路径")
        return

    if not os.path.exists(args.config_path):
        print(f"配置文件不存在: {args.config_path}")
        print("请确保配置文件存在")
        return

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    print(f"加载模型: {args.model_path}")
    model, config = load_model(args.model_path, args.config_path, device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    tokenizer = load_generation_tokenizer(config, args.model_type, args.tokenizer_path)
    generation_params = build_generation_params(args)

    if args.interactive:
        interactive_mode(model, tokenizer, device, generation_params)

    elif args.input_file:
        batch_mode(
            model,
            tokenizer,
            device,
            args.input_file,
            args.output_file,
            generation_params,
        )

    elif args.prompt:
        print(f"提示: {args.prompt}")
        print("生成中...")

        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            device=device,
            num_return_sequences=args.num_return_sequences,
        )
        print_single_prompt_results(args.prompt, generated_texts, args.num_return_sequences)
        save_single_prompt_results(args.prompt, generated_texts, args.output_file)
        print(f"结果已写入: {args.output_file}")

    else:
        print("未指定生成模式")
        print("使用 --prompt 指定提示文本")
        print("使用 --interactive 进入交互模式")
        print("使用 --input_file 进行批量生成")


if __name__ == "__main__":
    # 检查是否有训练好的模型
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"创建检查点目录: {checkpoint_dir}")
        print("请先运行 train.py 训练模型")

    # 运行生成
    main()
