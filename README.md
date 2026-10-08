# ActGuard

ActGuard is a pre-execution action-auditing defense against indirect prompt
injection in tool-using LLM agents. This repository contains the final ActGuard
implementation and the AgentDojo-based runtime needed to evaluate it.

The release contains only the final, full ActGuard configuration. Internal
components still use the identifier `reflective_audit` for compatibility with
the AgentDojo benchmark and trace formats.

## Requirements

- Python 3.10 or newer
- An OpenRouter API key
- Access to the `meta-llama/Meta-Llama-3.1-8B-Instruct` checkpoint, or an
  equivalent local checkpoint
- A CUDA-capable GPU is strongly recommended for local log-probability scoring

## Installation

Using [uv](https://docs.astral.sh/uv/) is recommended:

```bash
uv sync --frozen --extra actguard
```

Alternatively, install into an activated Python environment:

```bash
python -m pip install -e '.[actguard]'
```

## Quick start

From the repository root:

```bash
export OPENROUTER_API_KEY=...
uv run --frozen --extra actguard ./scripts/run_actguard.sh
```

If the project was installed with `pip`, run:

```bash
export OPENROUTER_API_KEY=...
./scripts/run_actguard.sh
```

The script evaluates the `workspace` suite with the
`important_instructions` attack. It uses OpenRouter for the main agent and
verifier, and Hugging Face-compatible local models for embedding retrieval and
contrastive log-probability attribution. Hugging Face authentication may be
required before the default gated Llama checkpoint can be downloaded.

To use a Claude API key directly for both the main agent and the ActGuard
verifier, run:

```bash
export ANTHROPIC_API_KEY=...
ACTGUARD_MODEL=claude-sonnet-4-6 \
ACTGUARD_VERIFIER_MODEL=claude-sonnet-4-6 \
uv run --frozen --extra actguard ./scripts/run_actguard.sh
```

This calls Anthropic's OpenAI SDK compatibility endpoint directly; it does not
use OpenRouter. If the API key has access to multiple Anthropic workspaces, set
`ANTHROPIC_WORKSPACE_ID` as well. The original OpenAI models remain the defaults
for reproducible runs. Claude Sonnet 4 is also available through OpenRouter as
`anthropic/claude-sonnet-4`.

To use downloaded checkpoints, replace the values of
`--reflective-audit-embedding-model` and
`--reflective-audit-logprob-model` in `scripts/run_actguard.sh` with their local
paths. Use `uv run --frozen actguard-benchmark --help` to see all benchmark
options.

## Default configuration

| Parameter | Setting |
| --- | --- |
| Backend model | `openai/gpt-4o-mini-2024-07-18` via OpenRouter |
| Backend model temperature | `0` |
| Chunk length | 80--180 characters |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieved chunks | `k = 3` |
| Contrastive log-probability model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Log-probability threshold | `0` |
| Similar-chunk grouping threshold | `0.9` |
| Tool-result context | Complete auditable history |
| Verifier model | `openai/gpt-5-mini` via OpenRouter |
| Verifier temperature | `0` |
| Predicted next-tool set size | 1--3 |
| Neighbor-chunk expansion | 1 chunk on each side |
| Maximum repair attempts per action | 2 |
| Local masking text | `[Removed suspicious instruction from external tool result.]` |

## Output

Results are written below:

```text
runs/openai_gpt-4o-mini-2024-07-18-reflective_audit/actguard/
```

The `runs/` directory is ignored by Git.

## Tests

Run the ActGuard regression tests:

```bash
uv run --frozen pytest -q tests/test_agent_pipeline/test_reflective_audit.py
```

## Repository layout

- `src/agentdojo/agent_pipeline/reflective_audit.py`: ActGuard implementation
- `src/agentdojo/agent_pipeline/agent_pipeline.py`: pipeline integration
- `src/agentdojo/scripts/benchmark.py`: benchmark CLI
- `src/agentdojo/default_suites/`: evaluation suites and task definitions
- `tests/test_agent_pipeline/test_reflective_audit.py`: regression tests
- `src/agentdojo/reflective_audit_trace.py`: trace generation and inspection

Run logs, local environments, credentials, paper drafts, temporary analysis
outputs, and comparison defenses are intentionally excluded from this release.

## License

This project is distributed under the MIT License. It includes an
AgentDojo-derived benchmark runtime; see `LICENSE` for the retained upstream
copyright notice.
