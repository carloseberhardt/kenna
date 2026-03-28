# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Engram is an implicit, user-scoped memory system for Claude Code. It extracts durable knowledge about the user from Claude Code session transcripts (JSONL) and makes it available via a read-only MCP tool (`engram_recall`). It is distinct from Claude Code's built-in project memory — engram captures *implicit* knowledge that accumulates without conscious effort, while `/remember` and `CLAUDE.md` handle explicit, project-scoped notes.

## Build & Run

```
cargo build                  # Build (first build is slow — llama.cpp compiles from source via ROCm)
cargo run -- <subcommand>    # Run CLI
cargo run -- reconcile --dry-run  # Preview what would be processed
cargo run -- reconcile       # Full extraction + curation + storage pipeline
cargo run -- list            # List engrams
cargo run -- search <query>  # Semantic vector search
cargo run -- serve           # Start MCP server (stdio, for Claude Code)
cargo run -- stats           # Show counts
```

## Architecture

- **Language**: Rust (edition 2024)
- **Binary**: `engram`
- **Vector store**: LanceDB 0.27 (embedded, Arrow-backed)
- **LLM inference**: `llama-cpp-2` 0.1.140 with ROCm backend (AMD GPU)
- **Extraction model**: Gemma 3 4B IT Q6_K GGUF (~3.2GB) — fast structured extraction
- **Curation model**: Qwen3 8B Q4_K_M GGUF (~5GB) — thinking model for quality filtering
- **Embedding model**: nomic-embed-text v1.5 Q8_0 GGUF (~138MB, CPU)
- **Data dir**: `~/.local/share/engram/` (db, models, state, logs)
- **Config**: `~/.config/engram/config.toml`

### Three-Phase Reconcile Pipeline

Models are loaded sequentially to manage GPU memory:

1. **Extract** (Gemma 3 4B on GPU) — processes preprocessed conversation chunks, produces candidate engrams as JSON
2. **Curate** (Qwen3 8B on GPU) — reviews candidates against source conversation text, drops hallucinations/trivia, fixes scope misclassification
3. **Embed + Reconcile** (nomic-embed on CPU) — dedup via cosine similarity, confidence gating, store to LanceDB

### Module Structure

```
src/
  main.rs              # Entry point, clap CLI
  config.rs            # XDG paths, TOML config, ReconcileConfig
  cli/
    reconcile.rs       # Three-phase pipeline orchestration
    serve.rs           # MCP server startup
    list/show/search/manage/stats/export.rs
  pipeline/
    discover.rs        # Session file discovery, exclusion filtering, aside_question scanning
    preprocess.rs      # JSONL parsing, noise stripping, chunking, aside extraction
    extract.rs         # Extraction prompt + LLM response parsing
    curate.rs          # Curation prompt + keep/drop/merge decisions
    reconcile.rs       # Dedup, supersession, confidence gating, storage
  mcp/
    server.rs          # rmcp-based stdio MCP server, engram_recall tool
  storage/
    models.rs          # Engram struct, enums, Arrow schema, RecordBatch conversion
    db.rs              # LanceDB operations (insert, list, get, delete, vector_search)
    cursor.rs          # Incremental processing cursor (tracks processed sessions)
  inference/
    mod.rs             # InferenceBackend trait
    llama.rs           # llama-cpp-2 implementation (generation, chat, embedding)
```

## Known Issues & Workarounds

### BERT embedding: Vulkan segfaults, ROCm works (resolved)

The safe `llama-cpp-2` wrappers (`embeddings_seq_ith`) segfault on BERT-type models
when using the Vulkan backend. Switching to ROCm fixed this. The safe API now works
for both generation and embedding.

### GBNF grammar sampling crashes (llama-cpp-2 0.1.140)

`LlamaSampler::grammar()` causes a foreign exception abort on both Vulkan and ROCm.
Not a grammar syntax issue — the crash happens during sampling, not initialization.
JSON validity is enforced by the response parser + truncation repair instead.

Search terms for investigating: `llama.cpp GBNF grammar crash abort sampler`,
`llama-cpp-2 rust grammar foreign exception`.

### Extraction model selection

Do NOT use reasoning/thinking models (Qwen3, DeepSeek-R1, etc.) for extraction.
They generate chain-of-thought before JSON, wasting tokens and causing parse failures.
Qwen3 IS used for curation, where thinking is an asset.

**Gemma 3 recommended sampling**: temp=1.0, top_k=64, top_p=0.95 (per Google).
Lower temperatures distort the model's calibration and cause worse output.

To compare extraction models: `engram reconcile --limit 1 --model <filename.gguf>`

### Extraction hallucination

Gemma 3 4B will hallucinate plausible-sounding facts from its training prior
(e.g., "Uses macOS" for a developer). The extraction prompt includes
"You know nothing about this user" framing and anti-inference rules. The Qwen3
curation pass catches most hallucinations by verifying claims against source text,
but some still slip through.

### Rust 2024 edition

`gen` is a reserved keyword — cannot be used as a variable name.

### ROCm GPU selection

`HIP_VISIBLE_DEVICES=0` is set in code to ensure the discrete GPU (7900 XT)
is used instead of the integrated GPU (Raphael). This is set before model loading.

## CLI Commands

```
engram reconcile [--dry-run] [--limit N] [--model file.gguf]
engram serve                # MCP server on stdio
engram list [--pending] [--scope personal|project] [--category X] [--entity X]
engram show <id>            # UUID prefix match supported
engram search <query>       # Vector search if embedding model available
engram accept <id>          # Candidate -> Accepted
engram delete <id>
engram promote <id>         # Project -> Personal scope
engram stats
engram export               # JSON to stdout
engram debug-insert "..."   # Test helper with real embeddings
```
