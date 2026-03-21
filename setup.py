#!/usr/bin/env python3
"""项目引导脚本。

保留 `setup.py` 入口，但行为改为显式、可重复执行的 bootstrap 流程：
1. 检查 Python 版本
2. 可选安装依赖
3. 创建基础目录
4. 生成示例数据
5. 运行环境检查
"""

import argparse
import os
import platform
import subprocess
import sys

from utils import create_sample_data


def print_header():
    print("=" * 70)
    print("GPT2ZERO 项目引导")
    print("=" * 70)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="初始化 GPT2ZERO 项目环境")
    parser.add_argument(
        "--skip-install", action="store_true", help="跳过依赖安装"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="跳过 test_imports.py 自检"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="示例训练数据条数"
    )
    return parser


def check_python_version():
    print("\n1. 检查 Python 版本")
    version = sys.version_info
    print(f"   当前版本: {sys.version.split()[0]}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(
            f"   不满足要求: 需要 Python 3.8+，当前为 "
            f"{version.major}.{version.minor}.{version.micro}"
        )
        return False

    print("   版本满足要求")
    return True


def check_system():
    print("\n2. 检查运行环境")
    print(f"   操作系统: {platform.system()}")
    print(f"   架构: {platform.machine()}")


def install_dependencies(python_executable: str):
    print("\n3. 安装依赖")
    try:
        result = subprocess.run(
            [python_executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
        )
    except Exception as error:
        print(f"   执行失败: {error}")
        return False

    if result.returncode != 0:
        print(f"   安装失败: {result.stderr[:400]}")
        return False

    print("   依赖安装完成")
    return True


def create_directories():
    print("\n4. 创建目录")
    for directory in ["data", "checkpoints", "results"]:
        os.makedirs(directory, exist_ok=True)
        print(f"   已就绪: {directory}/")


def seed_sample_data(num_samples: int):
    print("\n5. 创建示例数据")
    data_file, test_file = create_sample_data(num_samples=num_samples)
    print(f"   训练数据: {data_file}")
    print(f"   测试提示: {test_file}")


def run_checks(python_executable: str):
    print("\n6. 运行环境检查")
    try:
        result = subprocess.run(
            [python_executable, "test_imports.py"],
            capture_output=True,
            text=True,
        )
    except Exception as error:
        print(f"   执行失败: {error}")
        return False

    if result.returncode != 0:
        print("   自检未通过，关键输出如下:")
        print(result.stdout[-800:])
        print(result.stderr[-400:])
        return False

    print("   自检通过")
    return True


def print_next_steps():
    print("\n" + "=" * 70)
    print("下一步")
    print("=" * 70)
    print("1. 环境检查: python test_imports.py")
    print("2. 创建示例数据: python utils.py")
    print(
        "3. 快速训练: python train.py --model_type gpt2-mini --epochs 1 --batch_size 2 --max_length 64"
    )
    print(
        "4. 生成文本: python generate.py --model_type gpt2-mini --prompt '人工智能的未来' --max_length 40"
    )
    print("5. 查看文档: cat README.md")


def main():
    args = build_arg_parser().parse_args()
    print_header()

    if not check_python_version():
        return 1

    check_system()

    success = True
    if args.skip_install:
        print("\n3. 跳过依赖安装")
    else:
        success = install_dependencies(sys.executable) and success

    create_directories()
    seed_sample_data(args.num_samples)

    if args.skip_check:
        print("\n6. 跳过环境检查")
    else:
        success = run_checks(sys.executable) and success

    print_next_steps()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
