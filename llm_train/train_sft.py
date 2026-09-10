"""FirstLLM 64M SFT 脚本：沿用 PT 权重，只让 assistant 回答承担 loss。

用法（先跑完 train_pretrain.py 得到 PT checkpoint）：
    python llm_train/train_sft.py --config llm_train/configs/firstllm_64m_exp24.yaml \
        --ckpt llm_train/checkpoints/firstllm_64m_exp24/mini_seed42.pt

口径备忘：
    - Chat Template 与 notebook 11 的教学版保持一致：
        <s>user\n{问题}\nassistant\n{回答}</s>
    - labels 中问题与模板部分置 -100，只有回答 token 与 </s> 参与 loss
    - 方案未单独记录 SFT 超参：默认 lr 1e-4、batch 64、2 epoch，可按实验记录调整
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from modeling_firstllm import FirstLLMForCausalLM, config_from_yaml


def parse_args():
    """解析命令行参数，返回 (args)。"""
    parser = argparse.ArgumentParser(description="FirstLLM 64M SFT")
    parser.add_argument("--config", required=True, help="实验 yaml 路径")
    parser.add_argument("--ckpt", required=True, help="PT checkpoint（.pt）")
    parser.add_argument("--data", default=None, help="conversations JSONL，默认取 yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def build_sft_example(item, tokenizer, block_size, bos_id, eos_id, pad_id):
    """把一条 user/assistant 对话转成 (input_ids, labels)，格式与 notebook 11 相同。

    返回 None 表示该条无法使用（缺两轮、前缀过长或回答为空）。
    """
    turns = item.get("conversations", [])
    if len(turns) < 2:
        return None
    if turns[0].get("role") != "user" or turns[1].get("role") != "assistant":
        return None
    user_text = turns[0].get("content", "").strip()
    assistant_text = turns[1].get("content", "").strip()

    prefix_text = f"user\n{user_text}\nassistant\n"
    prefix = [bos_id] + tokenizer.encode(prefix_text).ids
    if len(prefix) >= block_size:
        return None
    answer = tokenizer.encode(assistant_text).ids + [eos_id]
    answer = answer[:block_size + 1 - len(prefix)]
    if not answer:
        return None

    full = prefix + answer
    padding = [pad_id] * (block_size + 1 - len(full))
    input_ids = (full + padding)[:-1]
    labels = [-100] * (len(prefix) - 1) + answer
    labels += [-100] * (block_size - len(labels))
    return input_ids, labels


def load_sft_samples(data_path, tokenizer, block_size, special_ids):
    """读取 conversations JSONL 并构建带 mask 的训练样本，返回张量与统计。"""
    bos_id, eos_id, pad_id = special_ids
    inputs, labels = [], []
    skipped = 0
    with Path(data_path).open(encoding="utf-8") as source:
        for line in source:
            example = build_sft_example(json.loads(line), tokenizer, block_size,
                                        bos_id, eos_id, pad_id)
            if example is None:
                skipped += 1
                continue
            inputs.append(example[0])
            labels.append(example[1])
    return (torch.tensor(inputs), torch.tensor(labels),
            {"kept": len(inputs), "skipped": skipped})


def lr_at(step, base_lr, warmup, total, min_ratio=0.1):
    """warmup 线性升温 + 余弦退火，与预训练脚本一致。"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress)))


def main():
    """加载 PT 权重与对话数据，完成 SFT 并落盘 checkpoint 与 metrics.jsonl。"""
    args = parse_args()
    config, full_config = config_from_yaml(args.config)
    sft_cfg = full_config["sft"]
    seed = args.seed or full_config["run"]["seeds"][0]
    data_path = args.data or sft_cfg["data"]
    block_size = sft_cfg["block_size"]

    tokenizer = Tokenizer.from_file(full_config["data"]["tokenizer"])
    # MiniMind 6400 词表无 <s>/</s>/<pad>：BOS 复用 <|endoftext|>(0)，
    # EOS 用 <|im_end|>(2)（与 PT 打包边界一致），pad 复用 <|endoftext|>（labels 已屏蔽）
    bos_id = tokenizer.token_to_id("<|endoftext|>")
    eos_id = tokenizer.token_to_id("<|im_end|>")
    pad_id = tokenizer.token_to_id("<|endoftext|>")
    assert None not in (bos_id, eos_id, pad_id), "tokenizer 缺少边界符"
    special_ids = (bos_id, eos_id, pad_id)

    sft_inputs, sft_labels, stats = load_sft_samples(
        data_path, tokenizer, block_size, special_ids)
    supervised = (sft_labels != -100).sum(dim=1)
    print(f"SFT 样本：{stats['kept']:,} 条（跳过 {stats['skipped']:,} 条）")
    print(f"平均监督 token：{supervised.float().mean().item():.1f}"
          f"（共 {int(supervised.sum()):,} 个）")

    torch.manual_seed(seed)
    device = torch.device(args.device)
    model, extra = FirstLLMForCausalLM.load_checkpoint(args.ckpt)
    model = model.to(device)
    print(f"从 {args_ckpt_name(args.ckpt)} 继续训练（step {extra.get('step')}）")
    print(f"参数量：{model.count_parameters():,}（tied）｜ seed={seed} device={device}")

    steps_per_epoch = int(np.ceil(len(sft_inputs) / sft_cfg["batch_size"]))
    total_steps = steps_per_epoch * sft_cfg["max_epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_cfg["lr"],
                                  betas=(0.9, 0.95),
                                  weight_decay=sft_cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 1)

    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_sft_seed{seed}.jsonl"

    model.train()
    step = 0
    for epoch in range(sft_cfg["max_epochs"]):
        order = torch.randperm(len(sft_inputs), generator=generator)
        for start in range(0, len(order), sft_cfg["batch_size"]):
            index = order[start:start + sft_cfg["batch_size"]]
            x = sft_inputs[index].to(device)
            y = sft_labels[index].to(device)
            lr = lr_at(step, sft_cfg["lr"], sft_cfg["warmup_steps"], total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            step_start = time.time()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss = model(x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == total_steps - 1:
                record = {"step": step, "epoch": epoch, "loss": round(loss.item(), 4),
                          "lr": lr, "tokens_per_sec": round(
                              x.numel() / max(1e-6, time.time() - step_start))}
                with metrics_path.open("a", encoding="utf-8") as target:
                    target.write(json.dumps(record) + "\n")
                print(f"SFT step {step}/{total_steps} | loss {loss.item():.4f}"
                      f" | lr {lr:.2e}")
            step += 1

    save_path = out_dir / Path(args.ckpt).name.replace(".pt", "_sft.pt")
    model.save_checkpoint(save_path, extra={"step": step, "loss": loss.item(),
                                            "sft_data": str(data_path), "seed": seed})
    print(f"完成：{save_path}")


def args_ckpt_name(path):
    """打印用：只显示 checkpoint 文件名，避免长路径刷屏。"""
    return Path(path).name


if __name__ == "__main__":
    main()
