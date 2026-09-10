"""FirstLLM 64M：Dense GQA(8Q/4KV) + qk-norm + SwiGLU + tied embedding 的 Decoder-only 模型。

对应 configs/firstllm_64m_exp24.yaml 的 model 段；notebook 11 里的 TeachingLM 是它的
等比例紧凑教学版（demo），本文件是全量训练/评测共用的正式实现。

- transformers 可用时继承 PreTrainedModel：支持 lm-eval 加载与 generate（gsm8k 用）
- transformers 不可用时退回纯 torch：训练与 PPL 评测不依赖 transformers
- `python modeling_firstllm.py info --config ...` 打印参数量
- `python modeling_firstllm.py export ...` 导出 HF 格式，供 run_lm_eval.sh 使用
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# transformers 可用时挂上 HF 接口（lm-eval 的 --model hf 后端需要）；否则纯 torch
try:
    from transformers import PretrainedConfig as _ConfigBase
    from transformers import GenerationMixin as _GenerationMixin
    from transformers import PreTrainedModel as _ModelBase
    from transformers.modeling_outputs import CausalLMOutputWithPast as _Output

    _HAS_HF = True
except ImportError:  # 无 transformers 时保持可 import、可训练
    _ConfigBase = object
    _ModelBase = nn.Module
    _GenerationMixin = object
    _HAS_HF = False

    @dataclass
    class _Output:
        """极简回退输出：提供与 CausalLMOutputWithPast 相同的 .logits/.loss 属性。"""

        logits: torch.Tensor = None
        loss: torch.Tensor = None

        def __getitem__(self, key):
            """支持下标与键名两种取法，贴近 ModelOutput 的行为。"""
            if key == 0 or key == "logits":
                return self.logits
            if key == 1 or key == "loss":
                return self.loss
            raise KeyError(key)

        def keys(self):
            """返回可用字段名，供上层按 dict 风格读取。"""
            return ["logits", "loss"]


class FirstLLMConfig(_ConfigBase):
    """FirstLLM 模型配置。

    参数与 yaml 的 model 段一一对应；返回/保存时序列化成 json。
    """

    model_type = "firstllm"

    def __init__(
        self,
        vocab_size=6400,
        hidden_size=768,
        num_layers=8,
        num_query_heads=8,
        num_kv_heads=4,
        ffn_size=2304,
        rope_theta=10000.0,
        norm_eps=1.0e-6,
        qk_norm=True,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if _HAS_HF:
            super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        else:
            super().__init__()
            for key, value in kwargs.items():  # 保留额外字段，序列化不丢信息
                setattr(self, key, value)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # HF 生成管线（DynamicCache 等）读标准字段名，同步别名
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_query_heads
        self.num_key_value_heads = num_kv_heads
        self.intermediate_size = ffn_size
        self.num_experts_per_tok = None
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.ffn_size = ffn_size
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = False  # 64M 规模直接全量重算，省去 KV cache 维护
        self.head_dim = hidden_size // num_query_heads


class RMSNorm(nn.Module):
    """按最后一维的均方根归一化输入（带可学习缩放）。"""

    def __init__(self, size, eps=1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def apply_rope(x, theta=10000.0):
    """给 [batch, head, seq, head_dim] 的 Q 或 K 加 RoPE 相对位置信息。"""
    seq_len, head_dim = x.size(-2), x.size(-1)
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    dimensions = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    frequencies = 1.0 / (theta ** (dimensions / head_dim))
    angles = torch.outer(positions, frequencies)
    cos = angles.cos()[None, None, :, :].to(x.dtype)
    sin = angles.sin()[None, None, :, :].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
    return rotated.flatten(-2)


class GQAAttention(nn.Module):
    """带 RoPE 与 qk-norm 的 Grouped-Query Attention。

    num_query_heads 个 query head 共享 num_kv_heads 组 K/V（qk_norm=True 时，
    Q/K 在打分前各过一次 RMSNorm，稳定注意力分数的数值尺度）。
    """

    def __init__(self, hidden_size, num_query_heads, num_kv_heads, rope_theta, qk_norm):
        super().__init__()
        assert hidden_size % num_query_heads == 0
        assert num_query_heads % num_kv_heads == 0
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_query_heads
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        kv_size = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, attn_mask=None):
        """输入 [batch, seq, hidden]，返回相同 shape 的注意力输出。"""
        batch, seq_len, hidden_size = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_query_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, self.rope_theta)
        k = apply_rope(k, self.rope_theta)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        repeats = self.num_query_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    """门控前馈网络：down(silu(gate(x)) * up(x))，三层均无偏置。"""

    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Pre-Norm 残差结构：attention 与 SwiGLU 各带一路 RMSNorm。"""

    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = GQAAttention(
            config.hidden_size,
            config.num_query_heads,
            config.num_kv_heads,
            config.rope_theta,
            config.qk_norm,
        )
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.ffn_size)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), attn_mask)
        return x + self.ffn(self.ffn_norm(x))


class FirstLLMForCausalLM(_ModelBase, _GenerationMixin):
    """Decoder-only Causal LM：tied embedding、GQA(8Q/4KV)、qk-norm、SwiGLU。

    transformers 5.x 起 PreTrainedModel 不再自带 generate，需显式继承 GenerationMixin。
    """

    if _HAS_HF:
        config_class = FirstLLMConfig
        _tied_weights_keys = {"lm_head.weight": "token_embedding.weight"}

    def __init__(self, config):
        if _HAS_HF:
            super().__init__(config)
        else:
            super().__init__()
            self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if _HAS_HF:
            self.post_init()  # v5：权重初始化 + 注册 tied 权重映射（lm-eval 加载需要）
        else:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Linear 与 Embedding 用 std=0.02 的正态初始化（PreTrainedModel 钩子）。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        """返回 FirstLLMOutput：logits [batch, seq, vocab]；labels 非空时带平均 CE loss。

        labels 中 -100 的位置不参与 loss（SFT 的 assistant-only mask 用）。
        kwargs 吞掉 generate 传入的 cache_position 等参数（本模型不用 KV cache）。
        """
        del kwargs  # 显式忽略 use_cache/cache_position：全量重算，不需要缓存
        seq_len = input_ids.size(1)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
        attn_mask = causal.tril()[None, None]
        if attention_mask is not None:
            padding = attention_mask[:, None, None, :].to(torch.bool)
            attn_mask = attn_mask & padding

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        logits = self.lm_head(self.final_norm(hidden))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return _Output(logits=logits, loss=loss)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """generate 每步喂完整序列（无 KV cache）；64M 模型的生成开销可接受。"""
        return {"input_ids": input_ids, "use_cache": False}

    @torch.no_grad()
    def count_parameters(self):
        """返回去重后的真实参数量（tied embedding 只计一次）。"""
        return sum(p.numel() for p in self.parameters())

    # ---------- 训练管线用的 checkpoint 存取（.pt，含配置与优化器可选） ----------

    def save_checkpoint(self, path, extra=None):
        """保存 state_dict + 配置 + extra（如 step/loss），路径不存在则创建。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": config_to_dict(self.config), "model": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu"):
        """从 .pt checkpoint 恢复模型（含配置），返回 (model, extra)。"""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(FirstLLMConfig(**payload["config"]))
        model.load_state_dict(payload["model"])
        extra = {k: v for k, v in payload.items() if k not in ("config", "model")}
        return model, extra


def config_to_dict(config):
    """把 FirstLLMConfig 转成可 json 序列化的 dict（两种基类都适用）。"""
    fields = [
        "vocab_size", "hidden_size", "num_layers", "num_query_heads", "num_kv_heads",
        "ffn_size", "rope_theta", "norm_eps", "qk_norm", "tie_word_embeddings",
        "use_cache", "head_dim", "model_type",
    ]
    result = {}
    for field in fields:
        value = getattr(config, field, None)
        if value is not None and field != "head_dim":
            result[field] = value
    return result


def export_hf(checkpoint, out_dir, tokenizer_json=None):
    """把 checkpoint 导出成 lm-eval 可加载的 HF 目录（remote code 方式）。

    checkpoint 可以是 .pt 路径，也可以是已构建的 FirstLLMForCausalLM 实例。
    产物：config.json（含 auto_map）、model.safetensors、modeling_firstllm.py 副本，
    以及可选的 tokenizer（tokenizers 库的 json 可直接被 AutoTokenizer 读取）。
    """
    from safetensors.torch import save_file

    if isinstance(checkpoint, FirstLLMForCausalLM):
        model, extra = checkpoint, {}
    else:
        model, extra = FirstLLMForCausalLM.load_checkpoint(checkpoint)
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {k: v for k, v in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)  # tied 权重只存 embedding 一份
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    config_dict = config_to_dict(model.config)
    config_dict["architectures"] = ["FirstLLMForCausalLM"]
    config_dict["auto_map"] = {
        "AutoConfig": "modeling_firstllm.FirstLLMConfig",
        "AutoModelForCausalLM": "modeling_firstllm.FirstLLMForCausalLM",
    }
    if extra:
        config_dict["checkpoint_extra"] = extra
    (out_dir / "config.json").write_text(
        json.dumps(config_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy(Path(__file__).resolve(), out_dir / "modeling_firstllm.py")

    if tokenizer_json is not None:
        shutil.copy(tokenizer_json, out_dir / "tokenizer.json")
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "model_max_length": 4096,
        }
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out_dir


def config_from_yaml(yaml_path):
    """读取实验 yaml，返回 (FirstLLMConfig, 完整配置 dict)，供训练脚本复用。"""
    import yaml

    full = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    return FirstLLMConfig(**full["model"]), full


def main():
    """命令行入口：info 打印参数量；export 导出 HF 目录。"""
    parser = argparse.ArgumentParser(description="FirstLLM 64M 模型工具")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="从 yaml 打印结构与参数量")
    info.add_argument("--config", required=True)

    export = sub.add_parser("export", help="导出 HF 格式目录（lm-eval 用）")
    export.add_argument("--ckpt", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--tokenizer", default=None, help="tokenizers 库的 json 路径")

    args = parser.parse_args()
    if args.command == "info":
        config, full = config_from_yaml(args.config)
        model = FirstLLMForCausalLM(config)
        embedding = config.vocab_size * config.hidden_size
        transformer = model.count_parameters() - embedding
        print(f"run: {full['run']['name']}  tier: {full['run']['tier']}")
        print(
            f"架构：{config.num_layers} 层 / hidden {config.hidden_size} / "
            f"Q{config.num_query_heads} KV{config.num_kv_heads} / ffn {config.ffn_size}"
        )
        print(f"参数量（tied，实测）：{model.count_parameters():,}")
        print(f"  其中 embedding {embedding:,}；transformer 主体 {transformer:,}")
    elif args.command == "export":
        out = export_hf(args.ckpt, args.out, args.tokenizer)
        print(f"已导出 HF 目录：{out}")


if __name__ == "__main__":
    main()
