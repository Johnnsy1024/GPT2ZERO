#!/usr/bin/env python3
"""
GPT2实现安装脚本
这个脚本帮助用户安装必要的依赖并设置环境
"""

import os
import sys
import subprocess
import platform

def print_header():
    """打印标题"""
    print("=" * 70)
    print("GPT-2 实现安装脚本")
    print("=" * 70)
    print()

def check_python_version():
    """检查Python版本"""
    print("1. 检查Python版本...")
    version = sys.version_info
    print(f"   当前Python版本: {sys.version.split()[0]}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ 需要Python 3.8+，当前版本: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"   ✅ Python版本满足要求")
        return True

def check_system():
    """检查系统信息"""
    print("\n2. 检查系统信息...")
    system = platform.system()
    machine = platform.machine()
    print(f"   操作系统: {system}")
    print(f"   架构: {machine}")
    return system, machine

def install_dependencies():
    """安装依赖"""
    print("\n3. 安装依赖...")
    
    # 依赖列表
    dependencies = [
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "numpy>=1.19.0",
        "tqdm>=4.0.0",
        "datasets>=1.0.0",
        "regex>=2020.0.0"
    ]
    
    print("   将安装以下依赖:")
    for dep in dependencies:
        print(f"     - {dep}")
    
    print("\n   开始安装...")
    
    # 尝试使用pip安装
    try:
        for dep in dependencies:
            print(f"   安装 {dep}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"     ✅ {dep} 安装成功")
            else:
                print(f"     ❌ {dep} 安装失败: {result.stderr[:100]}")
                # 尝试继续安装其他依赖
                continue
    
    except Exception as e:
        print(f"   安装过程中出错: {e}")
        return False
    
    print("\n   ✅ 所有依赖安装完成")
    return True

def create_directories():
    """创建必要的目录"""
    print("\n4. 创建目录结构...")
    
    directories = ['data', 'checkpoints', 'results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"   创建目录: {directory}/")
        else:
            print(f"   目录已存在: {directory}/")
    
    return True

def create_sample_data():
    """创建示例数据"""
    print("\n5. 创建示例数据...")
    
    try:
        # 导入utils模块创建示例数据
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import create_sample_data as create_data
        
        data_file, test_file = create_data(num_samples=100)
        print(f"   创建示例训练数据: {data_file}")
        print(f"   创建测试提示: {test_file}")
        
        return True
    except ImportError as e:
        print(f"   ⚠️  无法创建示例数据（依赖未完全安装）: {e}")
        print(f"   稍后可以运行 'python utils.py' 创建示例数据")
        return False
    except Exception as e:
        print(f"   ⚠️  创建示例数据时出错: {e}")
        return False

def run_tests():
    """运行测试"""
    print("\n6. 运行测试...")
    
    try:
        # 运行导入测试
        result = subprocess.run(
            [sys.executable, "test_imports.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✅ 导入测试通过")
            return True
        else:
            print("   ⚠️  导入测试失败，输出:")
            print(result.stdout[-500:])  # 显示最后500字符
            return False
            
    except Exception as e:
        print(f"   ❌ 运行测试时出错: {e}")
        return False

def print_next_steps():
    """打印下一步建议"""
    print("\n" + "=" * 70)
    print("安装完成！下一步：")
    print("=" * 70)
    
    print("\n1. 验证安装:")
    print("   python test_imports.py")
    
    print("\n2. 查看模型信息:")
    print("   python -c \"from config import GPT2Config; from model import GPT2LMHeadModel;")
    print("   config = GPT2Config.from_pretrained('gpt2');")
    print("   model = GPT2LMHeadModel(config);")
    print("   print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')\"")
    
    print("\n3. 开始训练:")
    print("   python train.py")
    print("   或使用自定义参数:")
    print("   python train.py --model_type gpt2-medium --batch_size 2 --epochs 5")
    
    print("\n4. 生成文本:")
    print("   python generate.py --prompt '人工智能的未来'")
    print("   或使用交互式模式:")
    print("   python generate.py --interactive")
    
    print("\n5. 更多选项:")
    print("   - 查看完整文档: cat README.md")
    print("   - 创建更多数据: python utils.py")
    print("   - 分析数据: python -c \"from utils import analyze_text_data; analyze_text_data('./data/training_data.txt')\"")
    
    print("\n" + "=" * 70)
    print("祝您使用愉快！")
    print("=" * 70)

def main():
    """主函数"""
    print_header()
    
    # 检查Python版本
    if not check_python_version():
        print("\n❌ Python版本不满足要求，请安装Python 3.8+")
        return False
    
    # 检查系统
    system, machine = check_system()
    
    # 询问用户是否继续
    print("\n" + "-" * 70)
    response = input("是否继续安装？(y/n): ").strip().lower()
    
    if response not in ['y', 'yes', '是']:
        print("安装已取消")
        return False
    
    # 安装依赖
    if not install_dependencies():
        print("\n⚠️  依赖安装可能有问题，请手动检查")
    
    # 创建目录
    create_directories()
    
    # 创建示例数据
    create_sample_data()
    
    # 运行测试
    run_tests()
    
    # 打印下一步建议
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n安装被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n安装过程中发生错误: {e}")
        sys.exit(1)
