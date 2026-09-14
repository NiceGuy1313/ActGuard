#!/usr/bin/env bash

export PYTHONPATH=src

python -m agentdojo.scripts.benchmark \
  -s workspace \
  --model openai/gpt-4o-mini-2024-07-18 \
  --defense reflective_audit \
  --attack important_instructions \
  --benchmark-version v1.2.2 \
  --reflective-audit-embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --reflective-audit-logprob-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --reflective-audit-verifier-model OPENROUTER_GPT_5_MINI \
  --reflective-audit-logprob-delta-threshold 0 \
  --reflective-audit-top-k 3
