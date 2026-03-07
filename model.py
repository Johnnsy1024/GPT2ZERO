import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from config import GPT2Config


class GPT2Attention(nn.Module):
    """GPT-2 多头注意力机制"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 注意力头数和每个头的维度
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        # 查询、键、值投影
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        # 注意力dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # 因果掩码
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
        
    def _split_heads(self, tensor, num_heads, head_dim):
        """将张量分割为多个注意力头"""
        batch_size, seq_length, hidden_size = tensor.shape
        tensor = tensor.view(batch_size, seq_length, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)  # (batch, heads, seq_length, head_dim)
    
    def _merge_heads(self, tensor, num_heads, head_dim):
        """合并多个注意力头"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_length, _, _ = tensor.shape
        return tensor.view(batch_size, seq_length, num_heads * head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # 计算查询、键、值
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.n_embd, dim=2)
        
        # 分割为多个头
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        if use_cache:
            present = (key, value)
        else:
            present = None
        
        # 缩放点积注意力
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # 应用因果掩码
        causal_mask = self.bias[:, :, :seq_length, :seq_length]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        
        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 计算注意力概率
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, value)
        
        # 合并头
        attn_output = self._merge_heads(attn_output, self.n_head, self.head_dim)
        
        # 输出投影
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present


class GPT2MLP(nn.Module):
    """GPT-2 多层感知机（前馈网络）"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    """GPT-2 Transformer块"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 自注意力
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        hidden_states = attn_output + residual
        
        # 前馈网络
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        
        return hidden_states, present


class GPT2Model(nn.Module):
    """完整的GPT-2模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词嵌入和位置嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer块
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]]]:
        
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("必须提供input_ids")
        
        # 生成位置ID
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # 嵌入层
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # 将 [batch, seq] 的0/1掩码转换为可加到注意力分数上的掩码
        # 目标形状: [batch, 1, 1, seq]
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -1e4
        
        # 准备past_key_values
        if past_key_values is None:
            past_key_values = [None] * len(self.h)
        
        presents = () if use_cache else None
        
        # 通过所有Transformer块
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            hidden_states, present = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            
            if use_cache:
                presents = presents + (present,)
        
        # 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states, presents


class GPT2LMHeadModel(nn.Module):
    """GPT-2语言模型头部"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 共享嵌入权重
        self.lm_head.weight = self.transformer.wte.weight
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[Tuple[torch.Tensor]]]]:
        
        # 前向传播
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 计算损失
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, lm_logits) + transformer_outputs[1:]
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.LongTensor:
        """生成文本"""
        self.eval()
        eos_token_id = kwargs.get("eos_token_id", getattr(self.config, "eos_token_id", None))
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 生成序列
        generated = input_ids
        for _ in range(max_length - input_ids.shape[1]):
            # 前向传播
            outputs = self(
                input_ids=generated[:, -self.config.n_positions:],
                use_cache=False,
            )
            
            # 获取下一个token的logits
            next_token_logits = outputs[1][:, -1, :] / temperature
            
            # 应用top-k过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 应用top-p过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 采样下一个token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到生成的序列
            generated = torch.cat((generated, next_token), dim=1)

            # 所有样本都生成eos后提前停止
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break
            
        return generated


if __name__ == "__main__":
    # 测试模型
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    
    print("模型结构:")
    print(model)
    
    # 测试前向传播
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    loss, logits, past_key_values = model(input_ids=input_ids, labels=input_ids)
    print(f"\n前向传播测试:")
    print(f"损失: {loss.item() if loss is not None else 'N/A'}")
    print(f"Logits形状: {logits.shape}")
    print(f"Past key values长度: {len(past_key_values) if past_key_values is not None else 0}")
