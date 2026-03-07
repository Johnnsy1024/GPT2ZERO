import json
import os
from typing import List, Dict, Optional
import regex as re


class GPT2Tokenizer:
    """GPT-2分词器（简化版）"""
    
    def __init__(self, vocab_file: Optional[str] = None, merges_file: Optional[str] = None):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.hf_tokenizer = None
        
        # 特殊token
        self.special_tokens = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        
        # 词汇表
        self.vocab = {}
        self.inverse_vocab = {}
        
        # BPE合并规则
        self.bpe_ranks = {}
        
        # 初始化词汇表
        self._init_vocab()
        
        # 如果提供了文件，从文件加载
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file, merges_file)
    
    def _init_vocab(self):
        """初始化基础词汇表"""
        # 添加字节级token
        for i in range(256):
            token = chr(i)
            token_id = len(self.special_tokens) + i
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
        
        # 添加特殊token到词汇表
        for token, token_id in self.special_tokens.items():
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
    
    def load_vocab(self, vocab_file: str, merges_file: Optional[str] = None):
        """从文件加载词汇表"""
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # 更新词汇表
        self.vocab.update(vocab_data)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 加载BPE合并规则
        if merges_file and os.path.exists(merges_file):
            with open(merges_file, 'r', encoding='utf-8') as f:
                merges = f.read().strip().split('\n')[1:]  # 跳过第一行
                merges = [tuple(merge.split()) for merge in merges]
            
            self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}
    
    def save_vocab(self, save_directory: str):
        """保存词汇表到文件"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存词汇表
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # 保存合并规则
        merges_path = os.path.join(save_directory, "merges.txt")
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for merge in sorted(self.bpe_ranks.keys(), key=lambda x: self.bpe_ranks[x]):
                f.write(f"{merge[0]} {merge[1]}\n")
    
    def bpe(self, token: str) -> List[str]:
        """对token进行BPE编码"""
        if token in self.vocab:
            return [token]
        
        # 将token拆分为字符
        word = list(token)
        
        # 应用BPE合并规则
        while True:
            pairs = []
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs.append(pair)
            
            if not pairs:
                break
            
            # 找到优先级最高的合并
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
            
            # 执行合并
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
        
        return word
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本编码为token IDs"""
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)

        # 清理文本
        text = text.strip()
        
        # 分割为token
        tokens = re.findall(r"\S+|\n", text)
        
        # 对每个token进行BPE编码
        bpe_tokens = []
        for token in tokens:
            if token == "\n":
                bpe_tokens.append("Ċ")  # GPT-2使用Ċ表示换行
            else:
                bpe_tokens.extend(self.bpe(token))
        
        # 转换为IDs
        token_ids = []
        for token in bpe_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.unk_token])
        
        # 添加特殊token
        if add_special_tokens:
            token_ids = [self.vocab[self.bos_token]] + token_ids + [self.vocab[self.eos_token]]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """将token IDs解码为文本"""
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        special_token_ids = {
            getattr(self, "pad_token_id", 0),
            getattr(self, "bos_token_id", 1),
            getattr(self, "eos_token_id", 2),
            getattr(self, "unk_token_id", 3),
        }
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if skip_special_tokens and token_id in special_token_ids:
                    continue
                tokens.append(token)
        
        # 合并BPE token
        text = "".join(tokens)
        
        # 还原特殊字符
        text = text.replace("Ċ", "\n")
        
        return text
    
    def __call__(self, text: str, **kwargs) -> Dict:
        """调用分词器"""
        input_ids = self.encode(text, **kwargs)
        return {"input_ids": input_ids}
    
    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2"):
        """从预训练模型加载分词器"""
        try:
            from transformers import GPT2Tokenizer as HFGPT2Tokenizer
            hf_tokenizer = HFGPT2Tokenizer.from_pretrained(model_name)
            
            # 创建自定义分词器实例
            tokenizer = cls()
            tokenizer.hf_tokenizer = hf_tokenizer
            
            # 复制词汇表
            tokenizer.vocab = hf_tokenizer.get_vocab()
            tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
            
            # 复制特殊token
            tokenizer.pad_token = hf_tokenizer.pad_token
            tokenizer.bos_token = hf_tokenizer.bos_token
            tokenizer.eos_token = hf_tokenizer.eos_token
            tokenizer.unk_token = hf_tokenizer.unk_token

            # GPT-2默认没有pad token，这里回退到eos token，便于训练时padding
            if hf_tokenizer.pad_token is None:
                tokenizer.pad_token = hf_tokenizer.eos_token
                tokenizer.pad_token_id = hf_tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = hf_tokenizer.pad_token_id
            tokenizer.bos_token_id = hf_tokenizer.bos_token_id
            tokenizer.eos_token_id = hf_tokenizer.eos_token_id
            tokenizer.unk_token_id = hf_tokenizer.unk_token_id if hf_tokenizer.unk_token_id is not None else 3

            tokenizer.special_tokens = {
                tokenizer.pad_token: tokenizer.pad_token_id,
                tokenizer.bos_token: tokenizer.bos_token_id,
                tokenizer.eos_token: tokenizer.eos_token_id,
                tokenizer.unk_token: tokenizer.unk_token_id,
            }
            
            # 复制BPE合并规则
            if hasattr(hf_tokenizer, "bpe_ranks"):
                tokenizer.bpe_ranks = hf_tokenizer.bpe_ranks
            
            return tokenizer
        except ImportError:
            print("警告: 未安装transformers库，使用简化分词器")
            return cls()


class SimpleTokenizer:
    """简单分词器（用于测试）"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """简单编码：将字符转换为ASCII码"""
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for char in text:
            token_id = ord(char) % (self.vocab_size - 10) + 10  # 保留前10个给特殊token
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """简单解码：将token IDs转换为字符"""
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id < 10:
                continue
            if token_id >= 10:
                char = chr((token_id - 10) % 256)
                chars.append(char)
        
        return "".join(chars)


if __name__ == "__main__":
    # 测试分词器
    print("测试GPT2Tokenizer:")
    
    # 使用transformers库的分词器（如果可用）
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        text = "Hello, world! This is a test."
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        print(f"原始文本: {text}")
        print(f"编码: {encoded}")
        print(f"解码: {decoded}")
        print(f"词汇表大小: {len(tokenizer.vocab)}")
    except Exception as e:
        print(f"无法加载transformers分词器: {e}")
        print("使用简单分词器进行测试...")
        
        tokenizer = SimpleTokenizer()
        text = "Hello, world!"
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        print(f"原始文本: {text}")
        print(f"编码: {encoded}")
        print(f"解码: {decoded}")
