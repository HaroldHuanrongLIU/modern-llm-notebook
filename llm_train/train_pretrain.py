"""FirstLLM 64M 预训练脚本（Ultra-FineWeb-zh mini/full 两档）。

用法（配置见 configs/firstllm_64m_exp24.yaml，数据先用 preprocess_ufw.py 产出）：
    python llm_train/train_pretrain.py --config llm_train/configs/firstllm_64m_exp24.yaml \
        --tier mini --seed 42
    python llm_train/train_pretrain.py --config ... --tier full --seed 777

口径备忘（以 WORK_记录_firstllm数据清洗.md 为准）：
    - 数据是用户自筛选的 Ultra-FineWeb-zh（DJ 1.5.5 清洗），MiniMind 仅作对照基线
    - mini 档 216,186 文档 / 2.7 亿 token；full 档 20 片 score>=0.7，目标 <22.1 亿 token
    - block 512 x batch 128 = 64K token/step；mini 5120 步、full 34560 步
    - lr：mini 档 2e-3、full 档 1e-3；warmup 100、weight decay 0.1；seed 42+777 双跑
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from modeling_firstllm import FirstLLMForCausalLM, config_from_yaml


def parse_args():
    """解析命令行参数，返回 (args)。tier/seed 默认取 yaml 里的第一个。"""
    parser = argparse.ArgumentParser(description="FirstLLM 64M 预训练")
    parser.add_argument("--config", required=True, help="实验 yaml 路径")
    parser.add_argument("--tier", default=None, choices=["mini", "full"])
    parser.add_argument("--seed", type=int, default=None, help="42/777 双 seed 各跑一次")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None, help="checkpoint 输出目录")
    return parser.parse_args()


def resolve_tier(full_config, tier):
    """按 tier 取数据路径与训练超参，返回 (data_cfg, train_cfg)。"""
    tier = tier or full_config["run"]["tier"]
    data_cfg = dict(full_config["data"][tier])
    train_cfg = dict(full_config["train"])
    train_cfg["max_steps"] = train_cfg[f"max_steps_{tier}"]
    train_cfg["lr"] = train_cfg[f"lr_{tier}"]
    return tier, data_cfg, train_cfg


def lr_at(step, base_lr, warmup, total, min_ratio):
    """warmup 线性升温 + 余弦退火到 base_lr x min_ratio。"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress)))


def batch_from_bin(data, block_size, batch_size, generator, device):
    """从打包 bin 里随机取 batch：x 是前 block 个 token，y 向后移一位。"""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[s:s + block_size].astype(np.int64))
                     for s in starts])
    y = torch.stack([torch.from_numpy(data[s + 1:s + block_size + 1].astype(np.int64))
                     for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_packed(model, val_data, block_size, batch_size, device, max_batches=50):
    """打包 val 口径的 PPL（跨文档上下文，我们口径）。"""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    usable = len(val_data) - block_size - 1
    for index in range(max_batches):
        start = (index * batch_size) % max(1, usable - batch_size)
        starts = np.arange(start, start + batch_size)
        x = torch.stack([torch.from_numpy(val_data[s:s + block_size].astype(np.int64))
                         for s in starts]).to(device)
        y = torch.stack([torch.from_numpy(val_data[s + 1:s + block_size + 1].astype(np.int64))
                         for s in starts]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss = model(x, labels=y).loss
        total_nll += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    return float(np.exp(total_nll / total_tokens))


def main():
    """加载配置与打包数据，完成一次预训练并落盘 checkpoint 与 metrics.jsonl。"""
    args = parse_args()
    config, full_config = config_from_yaml(args.config)
    tier, data_cfg, train_cfg = resolve_tier(full_config, args.tier)
    seed = args.seed or full_config["run"]["seeds"][0]

    train_bin = Path(data_cfg["train_bin"])
    val_bin = Path(data_cfg["val_bin"])
    assert train_bin.exists(), f"缺少 {train_bin}，先运行 preprocess_ufw.py pack 产出"
    assert val_bin.exists(), f"缺少 {val_bin}，先运行 preprocess_ufw.py pack 产出"

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)  # 取数与权重用不同流，稳定复现
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    block_size = full_config["data"]["block_size"]
    batch_size = train_cfg["batch_size"]

    device = torch.device(args.device)
    model = FirstLLMForCausalLM(config).to(device)
    print(f"tier={tier} seed={seed} device={device}")
    print(f"参数量：{model.count_parameters():,}（tied）")
    print(f"训练 token：{len(train_data):,}；步数 {train_cfg['max_steps']}"
          f" x {batch_size} x {block_size} = "
          f"{train_cfg['max_steps'] * batch_size * block_size:,} token")

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg["lr"],
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )

    out_dir = Path(args.out_dir or f"llm_train/checkpoints/{full_config['run']['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{tier}_seed{seed}.jsonl"
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                      enabled=device.type == "cuda")

    model.train()
    run_tag = f"{tier}/seed{seed}"
    for step in range(train_cfg["max_steps"]):
        lr = lr_at(step, train_cfg["lr"], train_cfg["warmup_steps"],
                   train_cfg["max_steps"], train_cfg["lr_min_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = batch_from_bin(train_data, block_size, batch_size, generator, device)
        step_start = time.time()
        with autocast():
            loss = model(x, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % train_cfg["log_interval"] == 0 or step == train_cfg["max_steps"] - 1:
            tokens_per_sec = batch_size * block_size / max(1e-6, time.time() - step_start)
            record = {"step": step, "loss": round(loss.item(), 4), "lr": lr,
                      "tokens_per_sec": round(tokens_per_sec), "time": time.time()}
            with metrics_path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(record) + "\n")
            print(f"[{run_tag}] step {step} | loss {loss.item():.4f} | lr {lr:.2e}")

        if (step + 1) % train_cfg["eval_interval"] == 0 or step == train_cfg["max_steps"] - 1:
            ppl = eval_packed(model, val_data, block_size, batch_size, device)
            print(f"[{run_tag}] step {step} | 打包 val PPL = {ppl:.4f}")
            model.save_checkpoint(
                out_dir / f"{tier}_seed{seed}.pt",
                extra={"step": step, "loss": loss.item(), "val_ppl": ppl, "tier": tier,
                       "seed": seed},
            )

    print(f"完成：{out_dir / (tier + '_seed' + str(seed) + '.pt')}")


if __name__ == "__main__":
    main()
