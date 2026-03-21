#!/usr/bin/env python3
"""项目环境与导入检查脚本。"""

import argparse
import importlib
import os

from utils import create_sample_data


CORE_THIRD_PARTY_LIBS = ["torch", "numpy", "tqdm", "regex"]
OPTIONAL_THIRD_PARTY_LIBS = ["transformers"]
PROJECT_FILES = [
    "requirements.txt",
    "config.py",
    "model.py",
    "tokenizer.py",
    "tokenizer_helpers.py",
    "training_pipeline.py",
    "generation_pipeline.py",
    "train.py",
    "generate.py",
    "utils.py",
    "README.md",
    "QUICKSTART.md",
]
PROJECT_IMPORTS = [
    ("config", "GPT2Config"),
    ("model", "GPT2LMHeadModel"),
    ("tokenizer", "CharTokenizer"),
    ("tokenizer_helpers", "encode_prompt_text"),
    ("training_pipeline", "TextDataset"),
    ("generation_pipeline", "generate_text"),
    ("utils", "create_sample_data"),
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="检查 GPT2ZERO 项目环境")
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="检查通过后顺手创建一份小型示例数据",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="示例数据条数")
    return parser


def check_file_structure():
    print("\n" + "=" * 60)
    print("检查文件结构")
    print("=" * 60)

    all_present = True
    for path in PROJECT_FILES:
        if os.path.exists(path):
            print(f"  OK   {path}")
        else:
            print(f"  MISS {path}")
            all_present = False

    for directory in ["data", "checkpoints", "results"]:
        if os.path.exists(directory):
            print(f"  DIR  {directory}/")
        else:
            print(f"  NOTE {directory}/ 不存在，按需创建")

    return all_present


def test_imports():
    print("\n" + "=" * 60)
    print("检查模块导入")
    print("=" * 60)

    passed = 0
    failed = 0

    print("\n第三方依赖:")
    for lib in CORE_THIRD_PARTY_LIBS:
        try:
            importlib.import_module(lib)
            print(f"  OK   {lib}")
            passed += 1
        except ImportError as error:
            print(f"  FAIL {lib}: {error}")
            failed += 1

    print("\n可选依赖:")
    for lib in OPTIONAL_THIRD_PARTY_LIBS:
        try:
            importlib.import_module(lib)
            print(f"  OK   {lib}")
        except ImportError:
            print(f"  NOTE {lib} 未安装，标准 GPT-2 分词器将不可用")

    print("\n项目模块:")
    for module_name, symbol_name in PROJECT_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            getattr(module, symbol_name)
            print(f"  OK   {module_name}.{symbol_name}")
            passed += 1
        except Exception as error:
            print(f"  FAIL {module_name}.{symbol_name}: {error}")
            failed += 1

    print("\n结果汇总:")
    print(f"  通过: {passed}")
    print(f"  失败: {failed}")
    return failed == 0


def maybe_create_sample_data(enabled: bool, num_samples: int):
    if not enabled:
        return

    print("\n" + "=" * 60)
    print("创建示例数据")
    print("=" * 60)
    data_file, test_file = create_sample_data(num_samples=num_samples)
    print(f"  训练数据: {data_file}")
    print(f"  测试提示: {test_file}")


def print_next_steps(success: bool):
    print("\n" + "=" * 60)
    print("下一步")
    print("=" * 60)

    if not success:
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 重新检查: python test_imports.py")
        return

    print("1. 创建样例数据: python utils.py")
    print(
        "2. 快速训练: python train.py --model_type gpt2-mini --epochs 1 --batch_size 2 --max_length 64"
    )
    print(
        "3. 生成文本: python generate.py --model_type gpt2-mini --prompt '人工智能的未来'"
    )
    print("4. 详细说明: cat README.md")


def main():
    args = build_arg_parser().parse_args()

    print("GPT2ZERO 环境检查")
    print("=" * 60)

    files_ok = check_file_structure()
    imports_ok = test_imports()
    success = files_ok and imports_ok

    if success:
        maybe_create_sample_data(args.create_sample_data, args.num_samples)

    print_next_steps(success)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
