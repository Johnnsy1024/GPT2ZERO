import json
import os
from typing import List, Optional

import torch

from config import GPT2Config
from model import GPT2LMHeadModel
from tokenizer import CharTokenizer, GPT2Tokenizer, SimpleTokenizer
from tokenizer_helpers import decode_generated_ids, encode_prompt_text


def load_model(model_path: str, config_path: str, device: str = "cpu"):
    """Load a trained model and config from disk."""
    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = json.load(file)

    config = GPT2Config(**config_dict)
    model = GPT2LMHeadModel(config)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, config


def load_generation_tokenizer(
    config, model_type: str = "gpt2", tokenizer_path: Optional[str] = None
):
    """Load the tokenizer used by generate.py without changing current behavior."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = CharTokenizer.from_file(tokenizer_path)
        print(f"使用CharTokenizer: {tokenizer_path}")
        return tokenizer

    if model_type == "gpt2-mini":
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        print("使用SimpleTokenizer")
        return tokenizer

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        print("使用GPT2Tokenizer")
        return tokenizer
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
    """Generate one or more continuations for a single prompt."""
    prompt_ids = encode_prompt_text(tokenizer, prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

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

    generated_texts = []
    for index in range(num_return_sequences):
        row = generated_ids[index] if num_return_sequences > 1 else generated_ids[0]
        generated_texts.append(decode_generated_ids(tokenizer, row.tolist()))

    return generated_texts


def strip_prompt_prefix(prompt: str, text: str) -> str:
    """Return only the newly generated suffix when the decoded text keeps the prompt."""
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text


def save_generation_results(results, output_file: str):
    """Persist generation results in txt or json format."""
    print(f"保存结果到 {output_file}...")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if output_file.endswith(".json"):
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
        return

    with open(output_file, "w", encoding="utf-8") as file:
        for result in results:
            file.write(f"提示: {result['prompt']}\n")
            file.write(f"生成文本: {result['generated_text']}\n")
            file.write(f"新内容: {result['new_text']}\n")
            file.write("-" * 80 + "\n")


def save_single_prompt_results(prompt: str, generated_texts, output_file: str):
    """Save one prompt's generation outputs using the same format as batch mode."""
    results = [
        {
            "prompt": prompt,
            "generated_text": text,
            "new_text": strip_prompt_prefix(prompt, text),
        }
        for text in generated_texts
    ]
    save_generation_results(results, output_file)


def interactive_mode(model, tokenizer, device, generation_params):
    """Run the interactive generation loop used by generate.py."""
    print("\n" + "=" * 60)
    print("GPT-2 交互式文本生成")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'params' 查看/修改生成参数")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n请输入提示文本: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("退出交互模式")
                break

            if prompt.lower() == "params":
                print("\n当前生成参数:")
                print(f"  最大长度: {generation_params['max_length']}")
                print(f"  温度: {generation_params['temperature']}")
                print(f"  top-k: {generation_params['top_k']}")
                print(f"  top-p: {generation_params['top_p']}")
                print(f"  采样: {generation_params['do_sample']}")
                print(f"  生成数量: {generation_params['num_return_sequences']}")

                change = input("\n是否修改参数? (y/n): ").lower()
                if change == "y":
                    try:
                        generation_params["max_length"] = int(
                            input(f"最大长度 [{generation_params['max_length']}]: ")
                            or generation_params["max_length"]
                        )
                        generation_params["temperature"] = float(
                            input(f"温度 [{generation_params['temperature']}]: ")
                            or generation_params["temperature"]
                        )
                        generation_params["top_k"] = int(
                            input(f"top-k [{generation_params['top_k']}]: ")
                            or generation_params["top_k"]
                        )
                        generation_params["top_p"] = float(
                            input(f"top-p [{generation_params['top_p']}]: ")
                            or generation_params["top_p"]
                        )
                        generation_params["do_sample"] = (
                            input(
                                f"采样 (true/false) [{generation_params['do_sample']}]: "
                            ).lower()
                            == "true"
                        )
                        generation_params["num_return_sequences"] = int(
                            input(
                                f"生成数量 [{generation_params['num_return_sequences']}]: "
                            )
                            or generation_params["num_return_sequences"]
                        )
                    except ValueError:
                        print("输入无效，保持原参数")
                continue

            if not prompt:
                print("提示文本不能为空")
                continue

            print("\n生成中...")
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=generation_params["max_length"],
                temperature=generation_params["temperature"],
                top_k=generation_params["top_k"],
                top_p=generation_params["top_p"],
                do_sample=generation_params["do_sample"],
                device=device,
                num_return_sequences=generation_params["num_return_sequences"],
            )

            print("\n" + "-" * 60)
            print(f"提示: {prompt}")
            print("-" * 60)

            for index, text in enumerate(generated_texts):
                if generation_params["num_return_sequences"] > 1:
                    print(f"\n生成结果 {index + 1}:")
                    print("-" * 40)

                print(strip_prompt_prefix(prompt, text))

            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n退出交互模式")
            break
        except Exception as error:
            print(f"生成错误: {error}")


def batch_mode(model, tokenizer, device, input_file, output_file, generation_params):
    """Run file-based generation in the same prompt-by-prompt style as before."""
    print(f"从 {input_file} 读取提示...")

    try:
        with open(input_file, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"文件不存在: {input_file}")
        return

    print(f"找到 {len(prompts)} 个提示")
    results = []

    for index, prompt in enumerate(prompts):
        print(f"处理提示 {index + 1}/{len(prompts)}: {prompt[:50]}...")

        try:
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=generation_params["max_length"],
                temperature=generation_params["temperature"],
                top_k=generation_params["top_k"],
                top_p=generation_params["top_p"],
                do_sample=generation_params["do_sample"],
                device=device,
                num_return_sequences=generation_params["num_return_sequences"],
            )

            for text in generated_texts:
                results.append(
                    {
                        "prompt": prompt,
                        "generated_text": text,
                        "new_text": strip_prompt_prefix(prompt, text),
                    }
                )

        except Exception as error:
            print(f"处理提示 '{prompt[:50]}...' 时出错: {error}")
            results.append(
                {
                    "prompt": prompt,
                    "generated_text": f"ERROR: {error}",
                    "new_text": f"ERROR: {error}",
                }
            )

    save_generation_results(results, output_file)
    print(f"完成! 生成了 {len(results)} 个结果")
