# kenna

_Implicit, user-scoped memory for Claude Code._

> **Archived, November 2026.** I've stopped running Kenna. The code stays
> here for reference, but it isn't maintained.

Kenna reads Claude Code's own session transcripts in the background and
extracts knowledge about the user: preferences, decisions, hardware,
recurring patterns. It makes that knowledge available to future sessions
through a read-only MCP tool and a `kenna search` CLI. The goal was
continuity across projects.

## Why it's archived

Claude Code shipped [auto memory](https://code.claude.com/docs/en/memory#auto-memory)
after I started this. During a session, Claude decides on its own what to
save, including notes about you (`user`, `feedback`), and those notes load
at the start of later sessions in the same repo. That's most of what Kenna
was for.

I measured the overlap on my own data before deciding. Auto memory already
held about half of the memories Kenna rated useful. For personal facts,
the thing Kenna was built around, it added almost nothing new: 2 useful
memories out of 51. What Kenna did add was mostly project-level preferences
the session never thought to save, plus reach across repos, since auto
memory is per repository. Numbers and method are in
[`kenna-design.md`](kenna-design.md#overlap-with-auto-memory-measured).

That wasn't enough to justify a GPU pipeline running every two hours,
especially since my sessions rarely called it. If cross-repo reach is what
you want, try pointing auto memory's `autoMemoryDirectory` setting at one
shared directory first. The tradeoff is that every project then shares one
200-line index.

## Status when archived

Single-user personal project, alpha. Opinionated about hardware:

- Linux with an AMD GPU via ROCm
- Should work on other ROCm-capable cards with enough VRAM (~11 GB free)
- CUDA/Metal have not been tested; the `llama-cpp-2` features in
  `Cargo.toml` would need to change

See [`kenna-design.md`](kenna-design.md) for the deep dive: architecture,
known issues, and the reasoning behind specific choices.

## How it works, briefly

Three-phase pipeline, one model on GPU at a time:

1. **Extract** — Gemma 4 E4B reads preprocessed conversation chunks and
   emits candidate memories as JSON
2. **Curate** — Gemma 4 12B verifies each candidate against the source text,
   drops hallucinations, session trivia, bare choices without reasoning, and
   claims that are really about the assistant rather than the user
3. **Embed + reconcile** — nomic-embed-text on GPU; dedup and supersession
   via cosine similarity against the existing store

Stored in an embedded Turso (SQLite) database, with embeddings kept as
BLOBs. Retrieved via the `kenna_recall` MCP tool, which ranks candidates
by exact cosine similarity computed in-app.

A periodic settling pass (`kenna settle`) consolidates the store —
promotes patterns that recur across projects to personal scope, and
synthesizes entity-level summaries from atomic memories.

## Requirements

- Rust (edition 2024)
- ROCm installed and working (`rocm-smi` on PATH)
- An AMD GPU with enough free VRAM for the curation model (~11 GB; one model
  is resident at a time)
- GGUF models — downloaded separately into `~/.local/share/kenna/models/`:
  - `gemma-4-E4B-it-UD-Q6_K_XL.gguf` (~7.5 GB) — extraction
  - `gemma-4-12b-it-UD-Q6_K_XL.gguf` (~10.7 GB) — curation (and settling)
  - `nomic-embed-text-v1.5.Q8_0.gguf` (~138 MB) — embedding

On a VRAM-constrained machine you can set `curation_model = "qwen3-8b-q4_k_m.gguf"`
(~5 GB) in the config instead — it still works, just with lower curation
quality (see [`kenna-design.md`](kenna-design.md) for the comparison). Lower the
`*_min_free_vram_gb` gates to match.

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
kenna reconcile --dry-run             # read-only preview; depth: chunks (default), extract, curate, reconcile
kenna reconcile --dry-run=extract     # e.g. run extraction only, print candidates (writes nothing)
kenna reconcile                       # full extraction + curation + storage pipeline
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
