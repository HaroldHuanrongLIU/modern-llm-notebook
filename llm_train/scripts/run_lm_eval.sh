#!/usr/bin/env bash
# lm-eval 11 任务统一评测（协议见 数据实验方案_精选数据超越MiniMind.md 第四节）：
#   固定 harness / 0-shot / acc_norm（生成类任务除外），11 任务为：
#   ceval-valid, cmmlu, mmlu, arc_easy, arc_challenge, piqa, openbookqa,
#   hellaswag, siqa(social_iqa), winogrande, gsm8k
# gsm8k 的 greedy + 5 次采样属于生成评测，另行用 generate 脚本统计，不在本脚本内。
#
# 用法示例：
#   CKPT=llm_train/checkpoints/firstllm_64m_exp24/mini_seed42.pt \
#   TOKENIZER=notebooks/part1-foundation/mini_tokenizer.json \
#   bash llm_train/scripts/run_lm_eval.sh
set -euo pipefail

LLM_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="${CKPT:?需要设置 CKPT=训练产出的 .pt 路径}"
TOKENIZER="${TOKENIZER:-notebooks/part1-foundation/mini_tokenizer.json}"
EXPORT_DIR="${EXPORT_DIR:-${CKPT%.pt}_hf_export}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT%.pt}_lm_eval}"
TASKS="${TASKS:-ceval-valid,cmmlu,mmlu,arc_easy,arc_challenge,piqa,openbookqa,\
hellaswag,siqa,winogrande,gsm8k}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"      # 统一协议 0-shot
BATCH_SIZE="${BATCH_SIZE:-auto}"
DEVICE="${DEVICE:-cuda}"

# 1) .pt checkpoint → HF 目录（remote code），lm-eval 的 --model hf 后端直接加载
python "${LLM_TRAIN_DIR}/modeling_firstllm.py" export \
  --ckpt "${CKPT}" --out "${EXPORT_DIR}" --tokenizer "${TOKENIZER}"

# 2) 统一协议跑 11 任务；loglikelihood 类任务取 acc_norm，多选类同时保留 acc
python -m lm_eval \
  --model hf \
  --model_args "pretrained=${EXPORT_DIR},trust_remote_code=True" \
  --tasks "${TASKS}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output_path "${OUTPUT_DIR}"

echo
echo "评测完成，结果 JSON 在：${OUTPUT_DIR}"
echo "汇总提示：loglikelihood 任务读 acc_norm；gsm8k 读 exact_match（严格匹配）。"
echo "统计要求：seed 42 与 777 各跑一次，报告均值±标准差；"
echo "          并做训练数据与评测题的 n-gram 去污染检查（报告污染子集分数）。"
