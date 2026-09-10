"""PPL 双口径评测：打包 val（我们口径）+ 单文档 val（MiniMind 口径）。

用法（val 文件由 preprocess_ufw.py pack 产出）：
    python llm_train/scripts/run_ppl.py \
        --config llm_train/configs/firstllm_64m_exp24.yaml --tier mini \
        --ckpt llm_train/checkpoints/firstllm_64m_exp24/mini_seed42.pt

两个口径不可互相替代（协议见 数据实验方案_精选数据超越MiniMind.md 第四节）：
    - 打包口径：val 流按 block 连续切块，attention 跨文档，训练怎么打包就怎么测
    - 单文档口径：每条文档独立编码（BOS+EOS），文档之间不共享上下文，
      与 MiniMind 的评测方式可比
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modeling_firstllm import FirstLLMForCausalLM, config_from_yaml


def parse_args():
    """解析命令行参数；val 路径缺省时从 manifest 读取。"""
    parser = argparse.ArgumentParser(description="FirstLLM PPL 双口径评测")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", default=None, choices=["mini", "full"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--val-bin", default=None)
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def packed_ppl(model, val_data, block_size, batch_size, device):
    """打包口径：val 流切成连续块，返回 (ppl, 评估块数)。"""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    blocks = len(val_data) // (block_size + 1)
    for start in range(0, blocks, batch_size):
        count = min(batch_size, blocks - start)
        batch = np.stack([val_data[(start + i) * (block_size + 1):
                                   (start + i) * (block_size + 1) + block_size + 1]
                          for i in range(count)]).astype(np.int64)
        x = torch.from_numpy(batch[:, :-1]).to(device)
        y = torch.from_numpy(batch[:, 1:]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss = model(x, labels=y).loss
        total_nll += loss.item() * y.numel()
        total_tokens += y.numel()
    return float(np.exp(total_nll / total_tokens)), blocks


@torch.no_grad()
def single_doc_ppl(model, docs, tokenizer, block_size, device, special_ids):
    """单文档口径：每条文档独立测，返回 (corpus_ppl, doc_mean_ppl, docs 数)。"""
    model.eval()
    bos_id, eos_id = special_ids
    total_nll, total_tokens, doc_ppls = 0.0, 0, []
    for doc in docs:
        ids = ([bos_id] + tokenizer.encode(doc["content"]).ids + [eos_id])
        ids = ids[:block_size + 1]  # 超长文档截断，与打包口径一致
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]]).to(device)
        y = torch.tensor([ids[1:]]).to(device)
        logits = model(x).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        doc_nll = nll.item()
        total_nll += doc_nll
        total_tokens += y.numel()
        doc_ppls.append(np.exp(doc_nll / y.numel()))
    corpus = float(np.exp(total_nll / total_tokens))
    return corpus, float(np.mean(doc_ppls)), len(doc_ppls)


def main():
    """加载 checkpoint，依次输出两个口径的 PPL 与对比说明。"""
    args = parse_args()
    config, full_config = config_from_yaml(args.config)
    tier = args.tier or full_config["run"]["tier"]
    data_cfg = full_config["data"][tier]
    block_size = full_config["data"]["block_size"]

    val_bin = Path(args.val_bin or data_cfg["val_bin"])
    val_jsonl = Path(args.val_jsonl or Path(val_bin).parent / "val.jsonl")
    tokenizer_path = args.tokenizer or full_config["data"]["tokenizer"]
    assert val_bin.exists(), f"缺少 {val_bin}，先运行 preprocess_ufw.py pack"
    assert val_jsonl.exists(), f"缺少 {val_jsonl}（pack 会同步产出）"

    device = torch.device(args.device)
    model, extra = FirstLLMForCausalLM.load_checkpoint(args.ckpt)
    model = model.to(device)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    special_ids = (tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>"))
    print(f"checkpoint：{args.ckpt}（step {extra.get('step')}）｜tier={tier}")

    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    ppl_a, blocks = packed_ppl(model, val_data, block_size, args.batch_size, device)
    print(f"打包 val 口径 PPL = {ppl_a:.4f}（{blocks:,} 个 block，attention 跨文档）")

    docs = [json.loads(line) for line in val_jsonl.open(encoding="utf-8")]
    ppl_b, ppl_doc_mean, doc_count = single_doc_ppl(
        model, docs, tokenizer, block_size, device, special_ids)
    print(f"单文档 val 口径 PPL = {ppl_b:.4f}"
          f"（{doc_count:,} 条文档独立测；文档均值口径 {ppl_doc_mean:.4f}）")
    print("说明：打包口径带跨文档上下文、分数通常更低；跨实验对比须用同一口径。")


if __name__ == "__main__":
    main()
