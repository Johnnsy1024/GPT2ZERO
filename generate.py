import torch
import argparse
import json
import os
from typing import List, Optional

from config import GPT2Config
from model import GPT2LMHeadModel
from tokenizer import GPT2Tokenizer, SimpleTokenizer


def load_model(model_path: str, config_path: str, device: str = "cpu"):
    """加载模型"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    config = GPT2Config(**config_dict)
    
    # 初始化模型
    model = GPT2LMHeadModel(config)
    
    # 加载模型权重
    if model_path.endswith('.pt'):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # 假设是检查点文件
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config


def load_tokenizer(config, model_type: str = "gpt2"):
    """加载分词器"""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        print("使用GPT2Tokenizer")
    except:
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        print("使用SimpleTokenizer")
    
    return tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
    device: str = "cpu",
    num_return_sequences: int = 1,
) -> List[str]:
    """生成文本"""
    # 编码提示
    if isinstance(tokenizer, GPT2Tokenizer):
        try:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        except:
            input_ids = [tokenizer.bos_token_id] + [ord(c) % 1000 + 10 for c in prompt] + [tokenizer.eos_token_id]
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(input_ids) == 0:
        bos_token_id = getattr(tokenizer, "bos_token_id", 1)
        input_ids = [bos_token_id]
    
    # 转换为张量
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 生成文本
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length + len(input_ids[0]),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        eos_token_id=getattr(tokenizer, "eos_token_id", model.config.eos_token_id),
    )
    
    # 解码生成的文本
    generated_texts = []
    for i in range(num_return_sequences):
        if num_return_sequences > 1:
            gen_ids = generated_ids[i].tolist()
        else:
            gen_ids = generated_ids[0].tolist()
        
        # 解码
        if isinstance(tokenizer, GPT2Tokenizer):
            try:
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            except:
                generated_text = "".join([chr((tid - 10) % 256) for tid in gen_ids if tid >= 10])
        else:
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        generated_texts.append(generated_text)
    
    return generated_texts


def interactive_mode(model, tokenizer, device, generation_params):
    """交互式生成模式"""
    print("\n" + "="*60)
    print("GPT-2 交互式文本生成")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'params' 查看/修改生成参数")
    print("="*60)
    
    while True:
        try:
            prompt = input("\n请输入提示文本: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式")
                break
            
            if prompt.lower() == 'params':
                print("\n当前生成参数:")
                print(f"  最大长度: {generation_params['max_length']}")
                print(f"  温度: {generation_params['temperature']}")
                print(f"  top-k: {generation_params['top_k']}")
                print(f"  top-p: {generation_params['top_p']}")
                print(f"  采样: {generation_params['do_sample']}")
                print(f"  生成数量: {generation_params['num_return_sequences']}")
                
                change = input("\n是否修改参数? (y/n): ").lower()
                if change == 'y':
                    try:
                        generation_params['max_length'] = int(input(f"最大长度 [{generation_params['max_length']}]: ") or generation_params['max_length'])
                        generation_params['temperature'] = float(input(f"温度 [{generation_params['temperature']}]: ") or generation_params['temperature'])
                        generation_params['top_k'] = int(input(f"top-k [{generation_params['top_k']}]: ") or generation_params['top_k'])
                        generation_params['top_p'] = float(input(f"top-p [{generation_params['top_p']}]: ") or generation_params['top_p'])
                        generation_params['do_sample'] = input(f"采样 (true/false) [{generation_params['do_sample']}]: ").lower() == 'true'
                        generation_params['num_return_sequences'] = int(input(f"生成数量 [{generation_params['num_return_sequences']}]: ") or generation_params['num_return_sequences'])
                    except ValueError:
                        print("输入无效，保持原参数")
                continue
            
            if not prompt:
                print("提示文本不能为空")
                continue
            
            print("\n生成中...")
            
            # 生成文本
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=generation_params['max_length'],
                temperature=generation_params['temperature'],
                top_k=generation_params['top_k'],
                top_p=generation_params['top_p'],
                do_sample=generation_params['do_sample'],
                device=device,
                num_return_sequences=generation_params['num_return_sequences'],
            )
            
            # 输出结果
            print("\n" + "-"*60)
            print(f"提示: {prompt}")
            print("-"*60)
            
            for i, text in enumerate(generated_texts):
                if generation_params['num_return_sequences'] > 1:
                    print(f"\n生成结果 {i+1}:")
                    print("-"*40)
                
                # 只显示新生成的部分（去除提示）
                if text.startswith(prompt):
                    new_text = text[len(prompt):].strip()
                else:
                    new_text = text
                
                print(new_text)
            
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\n退出交互模式")
            break
        except Exception as e:
            print(f"生成错误: {e}")


def batch_mode(model, tokenizer, device, input_file, output_file, generation_params):
    """批量生成模式"""
    print(f"从 {input_file} 读取提示...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"文件不存在: {input_file}")
        return
    
    print(f"找到 {len(prompts)} 个提示")
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"处理提示 {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        try:
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=generation_params['max_length'],
                temperature=generation_params['temperature'],
                top_k=generation_params['top_k'],
                top_p=generation_params['top_p'],
                do_sample=generation_params['do_sample'],
                device=device,
                num_return_sequences=generation_params['num_return_sequences'],
            )
            
            for text in generated_texts:
                results.append({
                    'prompt': prompt,
                    'generated_text': text,
                    'new_text': text[len(prompt):].strip() if text.startswith(prompt) else text
                })
        
        except Exception as e:
            print(f"处理提示 '{prompt[:50]}...' 时出错: {e}")
            results.append({
                'prompt': prompt,
                'generated_text': f"ERROR: {e}",
                'new_text': f"ERROR: {e}"
            })
    
    # 保存结果
    print(f"保存结果到 {output_file}...")
    
    if output_file.endswith('.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"提示: {result['prompt']}\n")
                f.write(f"生成文本: {result['generated_text']}\n")
                f.write(f"新内容: {result['new_text']}\n")
                f.write("-"*80 + "\n")
    
    print(f"完成! 生成了 {len(results)} 个结果")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 文本生成")
    parser.add_argument("--model_path", type=str, default="./checkpoints/final_model.pt", help="模型路径")
    parser.add_argument("--config_path", type=str, default="./checkpoints/config.json", help="配置路径")
    parser.add_argument("--model_type", type=str, default="gpt2", help="模型类型（用于分词器）")
    parser.add_argument("--prompt", type=str, help="提示文本")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="top-k参数")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p参数")
    parser.add_argument("--do_sample", action="store_true", default=True, help="使用采样")
    parser.add_argument("--no_sample", action="store_false", dest="do_sample", help="不使用采样（贪婪解码）")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="生成数量")
    parser.add_argument("--input_file", type=str, help="输入文件（批量模式）")
    parser.add_argument("--output_file", type=str, default="./generated_texts.txt", help="输出文件（批量模式）")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"模型文件不存在: {args.model_path}")
        print("请先运行 train.py 训练模型，或指定正确的模型路径")
        return
    
    if not os.path.exists(args.config_path):
        print(f"配置文件不存在: {args.config_path}")
        print("请确保配置文件存在")
        return
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model, config = load_model(args.model_path, args.config_path, device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 加载分词器
    tokenizer = load_tokenizer(config, args.model_type)
    
    # 生成参数
    generation_params = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'do_sample': args.do_sample,
        'num_return_sequences': args.num_return_sequences,
    }
    
    # 运行模式
    if args.interactive:
        interactive_mode(model, tokenizer, device, generation_params)
    
    elif args.input_file:
        batch_mode(model, tokenizer, device, args.input_file, args.output_file, generation_params)
    
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
        
        print("\n" + "="*60)
        for i, text in enumerate(generated_texts):
            if args.num_return_sequences > 1:
                print(f"\n生成结果 {i+1}:")
                print("-"*40)
            
            # 只显示新生成的部分
            if text.startswith(args.prompt):
                new_text = text[len(args.prompt):].strip()
            else:
                new_text = text
            
            print(new_text)
        
        print("="*60)
    
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
