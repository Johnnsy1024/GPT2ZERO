#!/usr/bin/env python3
"""
GPT2实现导入测试脚本
这个脚本测试所有必要的导入，而不需要实际运行模型
"""

import os
import sys


def test_imports():
    """测试所有必要的导入"""
    print("=" * 60)
    print("GPT2实现导入测试")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # 测试标准库导入
    print("\n1. 测试标准库导入...")
    try:
        import json
        import math
        import os
        import sys
        from typing import Dict, List, Optional, Tuple

        print("  ✅ 标准库导入成功")
        tests_passed += 1
    except ImportError as e:
        print(f"  ❌ 标准库导入失败: {e}")
        tests_failed += 1

    # 测试第三方库导入
    print("\n2. 测试第三方库导入...")
    third_party_libs = ["torch", "numpy", "tqdm", "regex"]

    for lib in third_party_libs:
        try:
            __import__(lib)
            print(f"  ✅ {lib} 导入成功")
            tests_passed += 1
        except ImportError:
            print(f"  ⚠️  {lib} 未安装（运行 'pip install {lib}' 安装）")
            tests_failed += 1

    # 测试项目模块导入
    print("\n3. 测试项目模块导入...")
    project_modules = ["config", "model", "tokenizer", "utils"]

    for module in project_modules:
        try:
            if module == "config":
                from config import GPT2Config

                print(f"  ✅ {module}.GPT2Config 导入成功")
            elif module == "model":
                from model import GPT2LMHeadModel

                print(f"  ✅ {module}.GPT2LMHeadModel 导入成功")
            elif module == "tokenizer":
                from tokenizer import GPT2Tokenizer, SimpleTokenizer

                print(f"  ✅ {module}.GPT2Tokenizer 导入成功")
            elif module == "utils":
                from utils import count_parameters, print_model_summary

                print(f"  ✅ {module}.count_parameters 导入成功")
            tests_passed += 1
        except ImportError as e:
            print(f"  ❌ {module} 导入失败: {e}")
            tests_failed += 1
        except Exception as e:
            print(f"  ⚠️  {module} 导入时出错（可能是依赖问题）: {e}")
            tests_failed += 1

    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  通过: {tests_passed}")
    print(f"  失败: {tests_failed}")
    print(f"  总计: {tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\n✅ 所有导入测试通过！")
        print("   现在可以运行 'python train.py' 开始训练模型")
    else:
        print("\n⚠️  有些导入测试失败")
        print("   请安装缺失的依赖:")
        print("   pip install torch transformers numpy tqdm datasets regex")

    print("=" * 60)

    return tests_failed == 0


def check_file_structure():
    """检查文件结构"""
    print("\n" + "=" * 60)
    print("检查文件结构...")
    print("=" * 60)

    required_files = [
        "requirements.txt",
        "config.py",
        "model.py",
        "tokenizer.py",
        "train.py",
        "generate.py",
        "utils.py",
        "README.md",
    ]

    optional_dirs = ["data", "checkpoints"]

    all_present = True

    print("\n必需文件:")
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (缺失)")
            all_present = False

    print("\n可选目录:")
    for directory in optional_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}/")
        else:
            print(f"  ⚠️  {directory}/ (不存在，将在需要时创建)")

    return all_present


def create_sample_structure():
    """创建示例目录结构"""
    print("\n" + "=" * 60)
    print("创建示例目录结构...")
    print("=" * 60)

    directories = ["data", "checkpoints"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"  创建目录: {directory}/")

    # 创建示例数据文件
    data_file = os.path.join("data", "example.txt")
    if not os.path.exists(data_file):
        with open(data_file, "w", encoding="utf-8") as f:
            f.write("这是一个示例文本文件，用于测试GPT2模型。\n")
            f.write("您可以将自己的训练数据放在data目录中。\n")
            f.write("每行应该是一个独立的文本样本。\n")
        print(f"  创建示例文件: {data_file}")

    print("\n目录结构创建完成！")


def main():
    """主函数"""
    print("GPT2实现环境检查")
    print("=" * 60)

    # 检查文件结构
    files_ok = check_file_structure()

    # 测试导入
    imports_ok = test_imports()

    # 创建目录结构（如果需要）
    if files_ok and imports_ok:
        create_sample_structure()

    # 最终建议
    print("\n" + "=" * 60)
    print("下一步建议:")
    print("=" * 60)

    if not imports_ok:
        print("1. 安装缺失的依赖:")
        print("   pip install -r requirements.txt")
        print("   或")
        print("   pip install torch transformers numpy tqdm datasets regex")

    print("2. 创建训练数据:")
    print("   python utils.py  # 创建示例数据")
    print("   或将自己的数据放入 data/ 目录")

    print("3. 开始训练:")
    print("   python train.py  # 默认使用 gpt2-mini 训练")

    print("4. 生成文本:")
    print("   python generate.py --prompt '你的提示文本'")
    print("   或")
    print("   python generate.py --interactive  # 交互式模式")

    print("\n详细说明请参阅 README.md")
    print("=" * 60)

    return files_ok and imports_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
