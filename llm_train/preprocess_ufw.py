"""Ultra-FineWeb-zh 数据管线：下载 → Data-Juicer 清洗 → 真实 BPE token 截断 → packing bin。

口径以 WORK_记录_firstllm数据清洗.md 为准（2026-08-13 实测），四条硬约束：

1. 不做 SimHash 去重：DJ 的 num_blocks=4 分桶导致 57% 假阳性误删；
   Ultra-FineWeb 上游已经 DUP 去重（面壁 MiniCPM 管线），实测重复率约 0.1%
2. 无浓缩步骤：top_frac=1.0 时必须显式跳过 source 均衡截断，
   否则 per_source=总数/8 会把数据偷偷砍掉（v2 教训：339,773 → 196,837 条）
3. token 预算必须按真实 BPE 计数：字符/2 估算会低估约 1.7 倍
   （实测字符/token = 1.26，中文几乎 1 字符 = 1 token）
4. score 列是字符串，粗筛前必须转 float

两档预算（截断方式：按 score 降序累加真实 token，到预算即停）：
    mini：001-004 片，score>=0.8，截到 2.7 亿 token（实得 216,186 文档，
          截断处 score 0.8833；方案上限 3.31 亿）
    full：001-020 片，score>=0.7，上限 22.1 亿 token（记录预计实得 ~18-20 亿）

完整流程示例：
    python llm_train/preprocess_ufw.py download --tier mini --raw-dir data/ufw_raw
    python llm_train/preprocess_ufw.py clean   --tier mini --raw-dir data/ufw_raw \
        --work-dir data/ufw_mini
    python llm_train/preprocess_ufw.py truncate --tier mini \
        --cleaned data/ufw_mini/ufw_mini_clean.jsonl --out data/ufw_mini
    python llm_train/preprocess_ufw.py pack --tier mini \
        --final data/ufw_mini/ufw_mini.jsonl --out data/ufw_mini \
        --tokenizer notebooks/part1-foundation/mini_tokenizer.json

说明：仓库不存放数据与清洗中间产物；bin/manifest 产出后按 yaml 的 data 段供
train_pretrain.py 使用。源数据集：openbmb/Ultra-FineWeb-zh，256 分片 x 1.27GB parquet，
列为 content / score（质量分）/ source（来源）。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 两档的固定口径；budget_token 是「按真实 BPE token 累加截断」的上限
TIERS = {
    "mini": {
        "shards": list(range(1, 5)),       # 001-004
        "min_score": 0.8,
        "budget_token": 270_000_000,       # 2.7 亿（WORK 记录 v4 实截口径）
        "history": "实测 216,186 文档，截断处 score 0.8833",
    },
    "full": {
        "shards": list(range(1, 21)),      # 001-020
        "min_score": 0.7,
        "budget_token": 2_210_000_000,     # 22.1 亿上限（实得以 manifest 为准）
        "history": "记录预计实得 ~18-20 亿 token",
    },
}
DJ_PROCESS = [
    # 只改内容、不减条数的三个 mapper + 长度过滤（阶段 2），清洗漏斗 100% 保留
    {"clean_html_mapper": {}},
    {"fix_unicode_mapper": {}},
    {"whitespace_normalization_mapper": {}},
    {"text_length_filter": {"min_len": 100, "max_len": 8000}},  # 单位：字符
]


def parse_args():
    """解析子命令与公共参数，返回 (args)。"""
    parser = argparse.ArgumentParser(
        description="Ultra-FineWeb-zh 清洗与打包（口径见模块 docstring）")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_tier(target):
        """每个子命令都带 --tier：mini=4 片 score>=0.8；full=20 片 score>=0.7。"""
        target.add_argument("--tier", default="mini", choices=list(TIERS))

    download = sub.add_parser("download", help="按 tier 下载指定分片 parquet")
    add_tier(download)
    download.add_argument("--raw-dir", required=True)
    download.add_argument("--repo", default="openbmb/Ultra-FineWeb")

    clean = sub.add_parser("clean", help="score 粗筛 + Data-Juicer 四算子清洗")
    add_tier(clean)
    clean.add_argument("--raw-dir", required=True)
    clean.add_argument("--work-dir", required=True)
    clean.add_argument("--dry-run", action="store_true", help="只写 DJ 配置并打印命令")

    truncate = sub.add_parser("truncate", help="按真实 BPE token 截断到预算")
    add_tier(truncate)
    truncate.add_argument("--cleaned", required=True, help="DJ 清洗后的 jsonl")
    truncate.add_argument("--out", required=True, help="输出目录")
    truncate.add_argument("--count-tokenizer", default=None,
                          help="预算计数用词表；记录口径为 MiniMind 6400 词表，"
                               "缺省回退到打包词表（字符/token≈1.26，量级一致）")

    pack = sub.add_parser("pack", help="BOS+EOS 打包成 bin，并写 data manifest")
    add_tier(pack)
    pack.add_argument("--final", required=True, help="truncate 产出的最终 jsonl")
    pack.add_argument("--out", required=True, help="输出目录")
    pack.add_argument("--tokenizer", required=True, help="模型用 tokenizers 库 json")
    pack.add_argument("--block-size", type=int, default=512)
    pack.add_argument("--val-docs", type=int, default=1000, help="尾部留作 val 的文档数")

    return parser.parse_args()


DEFAULT_REPO = "openbmb/Ultra-FineWeb"
ZH_SUBDIR = "data/ultrafineweb_zh"


def shard_files(repo, shards):
    """列出数据集 parquet 文件并按 1-based 分片号挑选，返回文件名列表。"""
    from huggingface_hub import HfApi

    files = sorted(name for name in HfApi().list_repo_files(repo, repo_type="dataset")
                   if name.endswith(".parquet") and ZH_SUBDIR in name)
    assert len(files) >= max(shards), (
        f"{repo} 只有 {len(files)} 个 parquet，无法满足分片 {shards}")
    return [files[index - 1] for index in shards]


def command_download(args):
    """下载 tier 对应的分片（断点续传）；限流可设 HF_ENDPOINT 镜像或手动下载。"""
    from huggingface_hub import hf_hub_download

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in shard_files(args.repo, TIERS[args.tier]["shards"]):
        target = hf_hub_download(args.repo, filename, repo_type="dataset",
                                 local_dir=raw_dir)
        print(f"已就位：{target}")


def score_to_float(value):
    """score 列实测是字符串（WORK 记录），统一转 float，坏值记 0。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def command_clean(args):
    """阶段 1 score 粗筛 + 阶段 2 Data-Juicer 四算子；不做去重与浓缩。"""
    import pandas as pd

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    coarse_path = work_dir / f"ufw_{args.tier}_coarse.jsonl"
    cleaned_path = work_dir / f"ufw_{args.tier}_clean.jsonl"
    config_path = work_dir / f"datajuicer_ufw_{args.tier}.yaml"

    raw_total, kept_total = 0, 0
    min_score = TIERS[args.tier]["min_score"]
    with coarse_path.open("w", encoding="utf-8") as target:
        for parquet in sorted(Path(args.raw_dir).rglob("*.parquet")):
            frame = pd.read_parquet(parquet, columns=["content", "score", "source"])
            raw_total += len(frame)
            frame = frame[frame["score"].map(score_to_float) >= min_score]
            kept_total += len(frame)
            for row in frame.itertuples(index=False):
                item = {"content": row.content, "score": score_to_float(row.score),
                        "source": row.source}
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"粗筛 score>={min_score}：{raw_total:,} → {kept_total:,} 条")

    # 注意：这里刻意没有 document_deduplicator / SimHash（57% 假阳性，上游已去重），
    # 也没有按 source 均衡截断（top_frac=1.0，显式跳过浓缩），见模块 docstring 第 1/2 条
    config_text = (
        f"project_name: ufw_{args.tier}\n"
        f"dataset_path: '{coarse_path}'\n"
        f"text_keys: 'content'\n"
        f"np: 8\n"
        f"export_path: '{cleaned_path}'\n"
        f"process:\n"
    )
    for step in DJ_PROCESS:
        for name, options in step.items():
            option_text = json.dumps(options, ensure_ascii=False)
            config_text += f"  - {name}: {option_text}\n"
    config_path.write_text(config_text, encoding="utf-8")
    print(f"DJ 配置已写入：{config_path}")

    command = [sys.executable, "-m", "data_juicer.tools.process_data",
               "--config", str(config_path)]
    print("运行：", " ".join(command))
    if args.dry_run:
        print("dry-run：跳过执行")
        return
    subprocess.run(command, check=True)
    cleaned_total = sum(1 for _ in cleaned_path.open(encoding="utf-8"))
    print(f"清洗完成：{kept_total:,} → {cleaned_total:,} 条"
          f"（mapper 只改内容，减量来自长度过滤，约 2.3%）")


def command_truncate(args):
    """按 score 降序累加真实 BPE token，到预算即停；写最终 jsonl 与 manifest。"""
    from tokenizers import Tokenizer

    pack_tokenizer_path = "notebooks/part1-foundation/mini_tokenizer.json"
    count_path = args.count_tokenizer or pack_tokenizer_path
    counter = Tokenizer.from_file(count_path)

    rows = []
    with Path(args.cleaned).open(encoding="utf-8") as source:
        for line in source:
            rows.append(json.loads(line))
    rows.sort(key=lambda item: -item["score"])  # 文件已按 score 有序，稳定排序保序

    budget = TIERS[args.tier]["budget_token"]
    total_tokens, kept, char_total = 0, 0, 0
    source_stats = {}
    kept_rows = []
    for row in rows:
        tokens = len(counter.encode(row["content"]).ids)
        if total_tokens + tokens > budget:
            break
        total_tokens += tokens
        char_total += len(row["content"])
        kept += 1
        kept_rows.append(row)
        stats = source_stats.setdefault(row["source"], {"docs": 0, "tokens": 0})
        stats["docs"] += 1
        stats["tokens"] += tokens

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"ufw_{args.tier}.jsonl"
    with final_path.open("w", encoding="utf-8") as target:
        for row in kept_rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    cutoff_score = kept_rows[-1]["score"] if kept_rows else None
    manifest = {
        "tier": args.tier,
        "budget_token": budget,
        "docs": kept,
        "tokens_real_bpe": total_tokens,
        "char_per_token": round(char_total / max(1, total_tokens), 3),
        "cutoff_score": cutoff_score,
        "count_tokenizer": count_path,
        "source_dist": source_stats,
        "tier_history": TIERS[args.tier]["history"],
    }
    manifest_path = out_dir / f"ufw_{args.tier}.truncate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"截断：{len(rows):,} → {kept:,} 文档，{total_tokens:,} token"
          f"（预算 {budget:,}）")
    print(f"截断处 score：{cutoff_score}｜字符/token：{manifest['char_per_token']}")
    top = sorted(source_stats.items(), key=lambda item: -item[1]["tokens"])[:3]
    print("token 占比前三位：", [(name, f"{item['tokens'] / total_tokens:.1%}")
                                for name, item in top])


def command_pack(args):
    """文档边界用 <|endoftext|>（MiniMind 6400 词表无 <s>/</s>），连成一条流切 bin。"""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert eos_id is not None, "tokenizer 缺少 <|endoftext|>"
    bos_id = eos_id  # 记录口径：文档间以 <|im_end|>/EOS 分隔，BOS 复用同一 id

    docs = [json.loads(line) for line in Path(args.final).open(encoding="utf-8")]
    assert len(docs) > args.val_docs, "文档数不足以切出 val，调小 --val-docs"
    train_docs, val_docs = docs[:-args.val_docs], docs[-args.val_docs:]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_bin(rows, path):
        """逐文档 [BOS]+ids+[EOS] 连接后写 uint16 bin，返回 (文档数, token 数)。"""
        stream = []
        for row in rows:
            stream.append(bos_id)
            stream.extend(tokenizer.encode(row["content"]).ids)
            stream.append(eos_id)
        array = np_uint16(stream)
        array.tofile(path)
        return len(rows), len(stream)

    train_count, train_tokens = write_bin(train_docs, out_dir / "train.bin")
    val_count, val_tokens = write_bin(val_docs, out_dir / "val.bin")
    # val 文档单独留一份 jsonl：run_ppl.py 的单文档口径需要原始文档
    with (out_dir / "val.jsonl").open("w", encoding="utf-8") as target:
        for row in val_docs:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    truncate_manifest_path = out_dir / f"ufw_{args.tier}.truncate_manifest.json"
    manifest = {"tier": args.tier, "tokenizer": args.tokenizer,
                "block_size": args.block_size,
                "doc_boundary": "BOS+EOS，块间连续（attention 跨文档）",
                "train": {"docs": train_count, "tokens": train_tokens},
                "val": {"docs": val_count, "tokens": val_tokens},
                "utilization": "100%（packing 无 padding，无丢弃）"}
    if truncate_manifest_path.exists():
        manifest["truncation"] = json.loads(
            truncate_manifest_path.read_text(encoding="utf-8"))
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    blocks = train_tokens // (args.block_size + 1)
    print(f"train.bin：{train_count:,} 文档 / {train_tokens:,} token"
          f"（≈{blocks:,} 个 block={args.block_size} 训练块）")
    print(f"val.bin：{val_count:,} 文档 / {val_tokens:,} token")
    print(f"manifest：{manifest_path}")


def np_uint16(stream):
    """token id 流转成 uint16 numpy 数组（词表 6400 远小于 65535 上限）。"""
    import numpy as np

    return np.array(stream, dtype=np.uint16)


def main():
    """按子命令分发；公共的 --tier 决定分片/分数/预算三件套。"""
    args = parse_args()
    handlers = {"download": command_download, "clean": command_clean,
                "truncate": command_truncate, "pack": command_pack}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
