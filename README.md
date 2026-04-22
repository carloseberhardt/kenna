# Kenna

_Implicit, user-scoped memory for Claude Code._

Kenna reads Claude Code's own session transcripts in the background and
extracts knowledge — preferences, decisions, hardware, recurring patterns,
etc. It makes that knowledge available to future sessions through a
read-only MCP tool. The goal is cross-project continuity.

It is **not** a replacement for Claude Code's built-in project memory
(`/remember`, `CLAUDE.md`). Those are for explicit, project-scoped notes
you choose to persist. Kenna is the implicit layer that accumulates on its
own.

## Status

Single-user personal project, alpha. Opinionated about hardware:

- Linux with an AMD GPU via ROCm
- Should work on other ROCm-capable cards with enough VRAM (~10 GB free)
- CUDA/Metal have not been tested; the `llama-cpp-2` features in
  `Cargo.toml` would need to change

Expect rough edges. See [`kenna-design.md`](kenna-design.md) for the deep
dive — architecture, known issues, and the reasoning behind specific
choices.

## How it works, briefly

Three-phase pipeline, one model on GPU at a time:

1. **Extract** — Gemma 4 E4B reads preprocessed conversation chunks and
   emits candidate memories as JSON
2. **Curate** — Qwen3 8B verifies each candidate against the source text,
   drops hallucinations, session trivia, and bare choices without
   reasoning
3. **Embed + reconcile** — nomic-embed-text on GPU; dedup and supersession
   via cosine similarity against the existing store

Stored in an embedded LanceDB vector store. Retrieved via the
`kenna_recall` MCP tool (semantic vector search).

A periodic settling pass (`kenna settle`) consolidates the store —
promotes patterns that recur across projects to personal scope, and
synthesizes entity-level summaries from atomic memories.

## Requirements

- Rust (edition 2024)
- ROCm installed and working (`rocm-smi` on PATH)
- An AMD GPU with enough free VRAM for Gemma 4 E4B or Qwen3 8B (~10 GB)
- GGUF models — downloaded separately into `~/.local/share/kenna/models/`:
  - `google_gemma-4-E4B-it-Q6_K.gguf` (~3.5 GB) — extraction
  - `qwen3-8b-q4_k_m.gguf` (~5 GB) — curation
  - `nomic-embed-text-v1.5.Q8_0.gguf` (~138 MB) — embedding

## Build and install

```sh
cargo install --path . --locked
```

The first build compiles llama.cpp from source via ROCm, so expect a long
wait (several minutes on a warm cache, much longer cold). The binary
lands at `~/.cargo/bin/kenna`.

## Configure

Default config is written on first run to `~/.config/kenna/config.toml`.
Edit thresholds, exclusions, and model filenames there. All fields have
sensible defaults — the file ships with everything commented out.

Data dir: `~/.local/share/kenna/` (db, models, state, debug, logs).

## Usage

```sh
kenna reconcile --dry-run     # preview what would be processed
kenna reconcile               # full extraction + curation + storage pipeline
kenna settle                  # cross-project promotion + entity synthesis
kenna list                    # list stored memories
kenna search "<query>"        # vector search
kenna stats                   # counts
kenna serve                   # MCP server on stdio (for Claude Code)
```

## Background reconciliation (systemd)

```sh
./systemd/install.sh
```

Installs and enables two user-level timers:

- `kenna-reconcile.timer` — every 2 hours, processes new sessions
- `kenna-settle.timer` — daily at 3am, consolidates the store

Both soft-exit cleanly if the GPU is busy, so running games or other LLM
work alongside is fine.

## Claude Code integration

Register the MCP server with Claude Code so `kenna_recall` is available
to every session:

```sh
claude mcp add kenna "$HOME/.cargo/bin/kenna" serve
```

There is also a skill template in `skill/SKILL.md` that wraps the CLI for
direct shell use. Copy it to `~/.claude/skills/kenna-recall/SKILL.md` if
you want the skill form (you don't need both).

## Design document

[`kenna-design.md`](kenna-design.md) covers the schema, pipeline internals,
settling pass design, directory layout, known issues and workarounds
(BERT/Vulkan, GBNF sampling, chat template handling, Gemma 4 behavior),
and the design history behind current choices.

## License

MIT. See [LICENSE](LICENSE).
