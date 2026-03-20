import json


class GPT2Config:
    """GPT-2 模型配置类"""

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 处理额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        """从预训练模型类型加载配置"""
        configs = {
            "gpt2-mini": cls(
                n_positions=256,
                n_embd=256,
                n_layer=4,
                n_head=4,
                n_inner=1024,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            ),
            "gpt2": cls(),
            "gpt2-medium": cls(n_embd=1024, n_layer=24, n_head=16, n_inner=4096),
            "gpt2-large": cls(n_embd=1280, n_layer=36, n_head=20, n_inner=5120),
            "gpt2-xl": cls(n_embd=1600, n_layer=48, n_head=25, n_inner=6400),
        }

        if model_type not in configs:
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {list(configs.keys())}"
            )

        return configs[model_type]

    def to_dict(self):
        """将配置转换为字典"""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def to_json_string(self):
        """将配置转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)

    def save_pretrained(self, save_directory):
        """保存配置到文件"""
        import os

        os.makedirs(save_directory, exist_ok=True)

        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json_file(cls, json_file):
        """从JSON文件加载配置"""
        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


if __name__ == "__main__":
    # 测试配置
    config = GPT2Config.from_pretrained("gpt2")
    print("GPT2 Small配置:")
    print(config.to_json_string())

    config_medium = GPT2Config.from_pretrained("gpt2-medium")
    print("\nGPT2 Medium配置:")
    print(
        f"n_embd: {config_medium.n_embd}, n_layer: {config_medium.n_layer}, n_head: {config_medium.n_head}"
    )
