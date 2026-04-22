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

1. **Extract** (Gemma 3 4B on GPU) — processes preprocessed conversation chunks via multi-turn few-shot chat, produces candidate engrams as JSON. Few-shot examples are in assistant turns (not system prompt) to avoid hallucination contamination.
2. **Curate** (Qwen3 8B on GPU) — per-chunk verification against source text. Strips `<think>` blocks before parsing. Drops hallucinations, bare choices without reasoning, session events, and trivia. Fixes scope misclassification. Outputs reason field for diagnostics.
3. **Embed + Reconcile** (nomic-embed on CPU) — dedup via cosine similarity, confidence gating, store to LanceDB

### Module Structure

```
src/
  main.rs              # Entry point, clap CLI
  config.rs            # XDG paths, TOML config, ReconcileConfig
  cli/
    reconcile.rs       # Three-phase pipeline orchestration (per-chunk curation)
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
    mod.rs             # InferenceBackend trait (generate, generate_chat, generate_chat_multi, embed)
    llama.rs           # llama-cpp-2 implementation (generation, multi-turn chat, embedding)
```

## Known Issues & Workarounds

### BERT embedding: Vulkan segfaults, ROCm works (resolved)

The safe `llama-cpp-2` wrappers (`embeddings_seq_ith`) segfault on BERT-type models
when using the Vulkan backend. Switching to ROCm fixed this. The safe API now works
for both generation and embedding.

### GBNF grammar sampling crashes (llama-cpp-2 0.1.140)

`LlamaSampler::grammar()` causes a foreign exception abort on both Vulkan and ROCm.
Not a grammar syntax issue — the crash happens during sampling, not initialization.
As of 2026-03-29, 0.1.140 is the latest published version — no fix available yet.

**CHECK ON EACH VISIT**: Run `cargo search llama-cpp-2` to see if a newer version
exists. If so, test whether `LlamaSampler::grammar()` still crashes with ROCm.
If grammar sampling works, it would eliminate all of the JSON repair workarounds
below (but validate that constrained output doesn't degrade extraction quality —
grammar constraints can sometimes cause repetition or truncation in small models).

Search terms for investigating: `llama.cpp GBNF grammar crash abort sampler`,
`llama-cpp-2 rust grammar foreign exception`.

**Workarounds in place because of this bug** (can be simplified/removed if grammar works):
- `extract.rs::parse_extraction_response()` — multi-stage fallback parser:
  code fence stripping, `[`-to-`]` extraction, object-level salvage for missing
  braces, multi-array bracket-depth splitting (`find_array_boundary`), truncation
  repair (`repair_truncated_json`), debug dump to `~/.local/share/engram/debug/`
- `curate.rs::strip_thinking_block()` — strips Qwen3 `<think>` blocks (separate
  issue from grammar, but related to JSON parsing robustness)
- `extract.rs::split_json_objects()` — string-context-aware object splitting for
  recovering individual candidates from malformed arrays

### Chat template: use oaicompat + use_jinja, not apply_chat_template

`src/inference/llama.rs` routes chat messages through
`LlamaModel::apply_chat_template_oaicompat(&template, &params)` with
`use_jinja: true` and `add_bos: true`, **not** the older
`LlamaModel::apply_chat_template()`.

Why: `apply_chat_template()` binds to the C API `llama_chat_apply_template`,
which uses a hand-coded token-substring detector in
`src/llama-chat.cpp::llm_chat_detect_template()` — e.g. `<start_of_turn>`
implies `LLM_CHAT_TEMPLATE_GEMMA`. Newer templates that don't match any
substring (Gemma 4's `<|turn>`, for example) return `LLM_CHAT_TEMPLATE_UNKNOWN`
and the function fails with FFI -1. The oaicompat path binds to a separate
C wrapper that calls `common_chat_templates_apply` — the Jinja-based engine
in `common/chat.cpp` that `llama-server` uses. That path renders the model's
embedded Jinja chat template directly, so new templates work out of the box.

BOS handling: `add_bos: true` tells the Jinja runtime to **strip** any
leading `bos_token` string the template emits (see
`common/chat.cpp:779`), so the caller is expected to re-add BOS at tokenize
time. That matches our `str_to_token(prompt, AddBos::Always)` in
`generate_with_grammar()`. Uniform behavior across Gemma 3 (whose template
has no `{{bos_token}}`) and Gemma 4 (whose template does).

### Gemma 4 extraction — works but JSON output is flakier than Gemma 3

As of 2026-04-22 (after the oaicompat migration above), Gemma 4 E4B loads,
applies its chat template, and extracts end-to-end. Apples-to-apples test
on session `e12417b5` (8 chunks) comparing the same Q6_K precision:

| Model | Extracted | Parse fails | Curation kept | Accepted |
|---|---|---|---|---|
| Gemma 3 4B Q6_K | 15 | 0/8 | 53% | 7 |
| Gemma 4 E4B Q6_K (bartowski imatrix) | 10 | 2/8 | 80% | 7 |

Gemma 4 is more selective — fewer candidates, higher quality per candidate —
but **loses whole chunks to JSON output errors that Gemma 3 doesn't produce**:
- Unescaped literal double quotes inside content strings (`"...complex, "real" long-term..."`)
- Occasional missing `confidence` / `scope_confidence` fields

Neither is a quantization artifact (tested Q4_K_M → Q6_K, same failures).
Both are Gemma 4 behavioral differences. Options if we want to lean on G4:
1. Prompt tuning — add explicit "escape all interior double quotes" rule
2. Make schema permissive — default confidences when missing
3. Post-process repair — detect unescaped quotes in content values and escape them

Not acted on yet; Gemma 3 4B remains the default extraction model.

**On the 26B-A4B settling model:** untested with the new API path. The 26B
at Q4_K_M is ~17 GB — borderline on 20 GB VRAM alongside overhead. That's
a separate investigation.

**On GGUF choice for E4B**: we use `bartowski/google_gemma-4-E4B-it-Q6_K.gguf`
(imatrix-calibrated). At Q6 and above, imatrix gives only a small edge vs
ggml-org's uniform quants; below Q4 the gap matters more.

### Extraction model selection

Do NOT use reasoning/thinking models (Qwen3, DeepSeek-R1, etc.) for extraction.
They generate chain-of-thought before JSON, wasting tokens and causing parse failures.
Qwen3 IS used for curation, where thinking is an asset — the `<think>` block is
stripped before JSON parsing in `curate.rs::strip_thinking_block()`.

**Gemma 3 recommended sampling**: temp=1.0, top_k=64, top_p=0.95 (per Google).
Lower temperatures distort the model's calibration and cause worse output.

To compare extraction models: `engram reconcile --limit 1 --model <filename.gguf>`

### Extraction hallucination

Gemma 3 4B will hallucinate plausible-sounding facts from its training prior
(e.g., "Uses macOS" for a developer). Mitigations:
- "You know nothing about this user" framing and anti-inference rules in extraction prompt
- Multi-turn few-shot examples in assistant turns (NOT system prompt — any concrete noun in the system prompt becomes a hallucination seed that Gemma parrots into every session)
- Qwen3 curation verifies claims against source text with explicit "absence of contradiction is not evidence of support" rule
- Curation reasons field enables debugging which hallucinations slip through

**Critical lesson**: Even obviously fictional examples (e.g., "Works at Initech") placed in the system prompt will be extracted verbatim from nearly every session. Few-shot examples must go in chat history turns, not instructions.

### Token estimation and chunking

`approx_token_count()` uses `chars/4` (not `words * 1.3`). The word-based estimate
severely undercounts for paths, UUIDs, base64, and other non-prose content.
Oversized paragraphs (no `\n\n` breaks) are force-split on single newlines.
Multi-array JSON responses (Gemma sometimes outputs `[...][...]`) are parsed by
bracket-depth counting in `parse_extraction_response()`.

### Inference performance tuning

Context size is 8192 (down from 16384) — sufficient for system prompt + few-shot +
4K chunk + output headroom, uses half the KV cache VRAM. Batch size is 2048 for
faster prompt prefill (long prompts relative to short outputs).

**Future investigation if speed becomes a concern:**
- Flash attention on ROCm/RDNA3 (`GGML_HIP_ROCWMMA_FATTN=ON`) — may help at
  longer contexts but has been hit-or-miss on consumer RDNA3 cards
- CPU thread count for nomic-embed — should match physical cores (6 on Ryzen)
- Quantization tradeoffs — Gemma Q8_0 for better extraction quality (fits in VRAM),
  Qwen Q4_K_S for faster curation if quality is sufficient

### Rust 2024 edition

`gen` is a reserved keyword — cannot be used as a variable name.

### ROCm GPU selection

`HIP_VISIBLE_DEVICES=0` is set in code to ensure the discrete GPU (7900 XT)
is used instead of the integrated GPU (Raphael). This is set before model loading.

### Re-running reconcile on an already-processed session (testing workflow)

The cursor at `~/.local/share/engram/state/cursor.json` records `(mtime,
byte_offset)` per processed session file. `reconcile --session <prefix>`
filters the session list but does **not** bypass the cursor — an
already-processed session will be skipped silently with "No candidates
extracted."

To test a specific session (e.g. A/B comparing models), temporarily remove
its cursor entry, run reconcile, then restore:

```
# 1. back up
cp ~/.local/share/engram/state/cursor.json /tmp/cursor.bak

# 2. delete the entry (filename format: <project-dir>/<uuid>.jsonl)
python3 -c "
import json; p='/home/carlose/.local/share/engram/state/cursor.json'
with open(p) as f: c=json.load(f)
c['processed'].pop('-home-carlose-projects-FOO/<uuid>.jsonl')
with open(p,'w') as f: json.dump(c,f,indent=2)"

# 3. run the test
cargo run -- reconcile --session <uuid-prefix> [--model <gguf>]

# 4. restore
cp /tmp/cursor.bak ~/.local/share/engram/state/cursor.json
```

Reprocessed engrams may only partially dedup (temp=1.0 is non-deterministic),
so the DB can end up with near-duplicates. `engram settle` handles that, or
delete the extra engrams by hand.

Subagent `agent-aside_question-*.jsonl` files and `<uuid>/` directories
(without `.jsonl`) do exist in `~/.claude/projects/` but are **not** main
sessions — confirm the filename exists before editing the cursor.

## CLI Commands

```
engram reconcile [--dry-run] [--limit N] [--model file.gguf] [--session ID_PREFIX]
engram settle [--dry-run]   # Cross-project promotion + entity synthesis
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
