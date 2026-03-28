# Engram — Design Sketch (v2)

## Naming Convention

- **Project**: engram
- **Binary**: `engram`
- **MCP tool**: `engram_recall` (singular — read-only from Claude Code's perspective)
- **Data dir**: `~/.local/share/engram/`
- **Config**: `~/.config/engram/config.toml`
- **CLAUDE.md reference**: "engrams" (not "memories")

---

## What Engram Is

Engram builds a durable, evolving understanding of its user from the traces
left by Claude Code sessions. It captures who you are, how you work, what
you care about, what you've built, what you've decided, what makes you laugh,
and what you keep coming back to — then makes that understanding available
at the start of every future session.

It is NOT a cross-project note-taking system. It is NOT a replacement for
Claude Code's built-in project memory (`/remember`, `CLAUDE.md`, auto memory).
Those handle explicit, project-scoped knowledge that the user consciously
chooses to persist.

Engram is the *implicit* layer — the things that accumulate naturally from
working together over time. The goal is continuity: sessions that feel like
picking up a conversation with someone who knows you, rather than introducing
yourself to a stranger every time.

**Relationship to Claude Code's built-in memory:**
- Claude Code's project memory → explicit, project-scoped, user-initiated
- Engram → implicit, user-scoped, background-extracted, available everywhere

---

## Memory Record Schema

```rust
struct Engram {
    // Identity
    id: Uuid,

    // Content
    content: String,          // The extracted fact/belief, concise natural language
    embedding: Vec<f32>,      // Vector from nomic-embed-text or similar

    // Classification
    scope: Scope,             // Personal | Project
    category: Category,       // Fact | Preference | Decision | Pattern | Context
                              //   Interest | Humor | Opinion
    entity: Option<String>,   // Primary entity/topic key, e.g. "solis", "rainy-lake-cabin"
                              // Used for supersession detection

    // Provenance
    source_project: Option<String>,  // Absolute path of originating project
    source_session: String,          // Session UUID from Claude Code JSONL
    source_timestamp: DateTime,      // When the source interaction occurred

    // Lifecycle
    lifecycle: Lifecycle,     // Candidate | Accepted
    confidence: f32,          // 0.0-1.0 from extraction model
    created_at: DateTime,
    updated_at: DateTime,
    accessed_at: Option<DateTime>,  // Last time retrieved via MCP

    // Evolution
    supersedes: Option<Uuid>, // If this engram replaces an earlier one
    superseded_by: Option<Uuid>,
}

enum Scope {
    Personal,   // About the user as a person, regardless of project
    Project,    // Specific to a codebase: architecture decisions, conventions, tool choices
}

enum Category {
    Fact,        // "Works at IBM as a Platform Architect"
    Preference,  // "Prefers Rust for CLI tools, Python when prototyping"
    Decision,    // "Chose LanceDB for the vector store because embedded + Rust-native"
    Pattern,     // "Tends to sketch the tech stack before designing data models"
    Context,     // "Currently building Solis demo for IBM stakeholders"
    Interest,    // "Reads dense science fiction — currently Hannu Rajaniemi's Jean le Flambeur trilogy"
    Humor,       // "Jokingly suggested writing the screenshot tool in Haskell for 'street cred'"
    Opinion,     // "Thinks ContextForge is agent-as-tool, not true peer-to-peer A2A"
}

enum Lifecycle {
    Candidate,   // Freshly extracted, not yet reconciled against existing engrams
    Accepted,    // Validated: no contradictions, passes confidence threshold
}
```

### Design Notes

- **No session scope**: The paper's ephemeral/session tier maps to "just don't
  extract it." If something is only relevant within a session, the extraction
  model should skip it. We don't store throwaway context.

- **Entity field**: Key for supersession. When the reconciler sees a new engram
  with entity="solis-gateway" and an existing accepted engram with the same
  entity, it can compare timestamps and content to decide if the new one
  replaces the old. Without this, supersession requires full semantic comparison
  of every incoming engram against the whole store.

- **Confidence threshold**: Engrams below a configurable threshold (e.g., 0.6)
  are dropped entirely, never even reaching Candidate. Candidates above threshold
  but below a higher bar (e.g., 0.85) stay as Candidate for review. Above 0.85
  auto-promote to Accepted.

- **accessed_at**: Enables future decay logic. Engrams that haven't been retrieved
  in N months could be flagged for review or archival. Not part of v1 but the
  field is cheap to keep.

- **Broader categories**: Interest, Humor, and Opinion capture the kind of
  personal texture that makes continuity feel real. Knowing that someone
  enjoys Valheim on a modded server or made a joke about Haskell isn't a
  "decision" or "fact" — but it's exactly the kind of thing that makes a
  future session feel like a continuation rather than a cold start.

---

## Claude Code Session Data Format

Claude Code stores session data under `~/.claude/projects/`. Understanding
this structure is essential for the preprocessing pipeline.

### Directory Layout

```
~/.claude/
├── projects/
│   ├── -home-carlose-projects-engram/          # One dir per project
│   │   ├── 08da7c6a-db97-...jsonl              # Main session files (UUID)
│   │   ├── agent-a1f5617a...jsonl              # Agent-spawned sessions
│   │   ├── 08da7c6a-.../                       # Per-session subdirectory
│   │   │   └── subagents/
│   │   │       ├── agent-aside_question-*.jsonl # /btw command sessions
│   │   │       ├── agent-acompact-*.jsonl       # Context compression (system)
│   │   │       ├── agent-aprompt_suggestion-*.jsonl  # Prompt suggestions (system)
│   │   │       └── agent-*.jsonl                # Tool subagents (system)
│   │   └── ...
│   └── -home-carlose-projects-ralph-trader/
│       └── ...
├── history.jsonl                               # Prompt history (all sessions)
└── settings.json
```

**Directory naming**: Project paths are encoded by replacing `/` with `-`.
Example: `/home/carlose/projects/engram` → `-home-carlose-projects-engram`.
This encoding is lossy — hyphens in directory names are indistinguishable
from path separators. Exclusion matching works on the encoded form.

### JSONL Record Format

Each line is a JSON object. Key fields:

```
{
  "type": "user" | "assistant" | "system" | "progress" |
          "file-history-snapshot" | "last-prompt",
  "uuid": "...",
  "parentUuid": "...",
  "sessionId": "...",
  "timestamp": "2026-03-27T21:04:25.628Z",
  "message": {
    "role": "user" | "assistant",
    "content": <string> | <array of content blocks>
  }
}
```

**Content types** (when `message.content` is an array):
- `{"type": "text", "text": "..."}` — conversation text
- `{"type": "thinking", ...}` — model reasoning (stripped by preprocessor)
- `{"type": "tool_use", "name": "...", "input": {...}}` — tool invocations
- `{"type": "tool_result", "tool_use_id": "...", "content": "..."}` — tool output

**User messages** can have `content` as either:
- A plain string (direct user input)
- An array of content blocks (when tool results are included)

### Subagent Types

Claude Code spawns subagents for various purposes. Their JSONL files live
in `<session_uuid>/subagents/`. Each subagent file contains the full parent
session context replayed, followed by the subagent-specific exchange.

| Prefix | Purpose | User content? |
|--------|---------|---------------|
| `agent-aside_question-*` | `/btw` command — user asks a side question without interrupting the main agent | **Yes** — real user-authored input, opinions, decisions. High signal for engram. |
| `agent-*` (generic) | Tool-use subagents spawned by the main agent | No — "user" messages are system-generated task prompts |
| `agent-acompact-*` | Context compression when conversation gets long | No — system prompt to summarize |
| `agent-aprompt_suggestion-*` | Slash command suggestions | No — system prompt for suggestion mode |

**Key discovery**: `/btw` content is NOT recorded in the main session JSONL.
It only exists in the `aside_question` subagent file. If engram doesn't scan
subagents, `/btw` content is invisible to the extraction pipeline.

### Noise in User Messages

Several XML-style tags are injected by the Claude Code harness into user
messages. These are not user-authored content and must be stripped:

- `<system-reminder>...</system-reminder>` — system instructions injected
  into user turns
- `<local-command-caveat>...</local-command-caveat>` — wrapper around
  output from user's local commands
- `<local-command-stdout>`, `<local-command-stderr>` — shell output tags
- `<bash-input>`, `<bash-stdout>`, `<bash-stderr>` — command execution tags
- `<command-message>...</command-message>` and `<command-name>...</command-name>`
  — slash command invocations (e.g. `/init`)

### What We Scan

1. **Main session JSONL files** (`*.jsonl` directly in project dirs) —
   primary conversation data. Gated by minimum 4 human turns.
2. **Aside question subagents** (`agent-aside_question-*.jsonl` in
   `subagents/` dirs) — `/btw` content. Bypasses turn minimum since
   these are typically 1-2 turns of high-signal personal content.
   Only the aside-specific turns are extracted, not the replayed context.

We skip: `agent-acompact-*`, `agent-aprompt_suggestion-*`, and generic
`agent-*` subagents — these contain only system-generated content.

---

## Extraction Prompt

The extraction model receives preprocessed conversation chunks and produces
structured JSON. The prompt is the most critical design artifact — it determines
what gets remembered and how well it's classified.

```
<s>
You are an engram extraction system. You analyze conversations between a
developer and an AI coding assistant to identify durable knowledge about
the user — who they are, how they think, what they care about, and what
they've done.

Think of yourself as building a portrait of a person over time. You're
looking for anything that would help a future conversation feel like a
continuation rather than a fresh start.

Extract things like:
- Facts about the user (role, location, setup, projects, people they mention)
- Preferences and opinions (tools, languages, approaches, aesthetics)
- Decisions made and their rationale (especially "we chose X because Y")
- Patterns in how they work (planning style, communication style, priorities)
- Interests and hobbies mentioned in passing
- Recurring topics or running jokes
- Technical context about their environment (hardware, OS, stack)
- Things they're currently working on or thinking about

DO NOT extract:
- Transient debugging steps or intermediate reasoning
- Code snippets or file contents (reference them, don't reproduce them)
- Questions that were asked but not resolved
- Speculative ideas that were discussed but explicitly rejected
- Generic technical knowledge (e.g., "Rust has a borrow checker")
- Purely procedural exchanges ("run this command" / "ok done")

For each extracted item, classify:
- scope: "personal" if it's about the user regardless of project, "project"
  if it's specific to the codebase being worked on
- category: one of "fact", "preference", "decision", "pattern", "context",
  "interest", "humor", "opinion"
- entity: a short kebab-case key identifying the primary topic
  (e.g., "solis-gateway", "rainy-lake-cabin", "hockey", "home-workstation")
- confidence: 0.0-1.0 how certain you are this is durable and accurate
- content: a concise, self-contained statement in natural language. It must
  be understandable without the original conversation. Write it as something
  you know about this person, not as a summary of dialogue.

Respond with a JSON array. If nothing worth extracting, respond with [].

Example output:
[
  {
    "content": "Chose LanceDB over Qdrant for the engram vector store because it is embedded, Rust-native, and requires no server process",
    "scope": "project",
    "category": "decision",
    "entity": "engram-vector-store",
    "confidence": 0.92
  },
  {
    "content": "Prefers to define the tech stack before designing data models or writing code",
    "scope": "personal",
    "category": "pattern",
    "entity": "workflow-style",
    "confidence": 0.78
  },
  {
    "content": "Plays recreational hockey in the Wyoming, MN area",
    "scope": "personal",
    "category": "interest",
    "entity": "hockey",
    "confidence": 0.95
  },
  {
    "content": "Jokingly suggested writing a screenshot tool in Haskell for 'classy street cred'",
    "scope": "personal",
    "category": "humor",
    "entity": "language-humor",
    "confidence": 0.70
  }
]
</s>

<user>
Project: {project_path}
Session: {session_id}
Timestamp range: {start_time} — {end_time}

<conversation>
{preprocessed_conversation_chunk}
</conversation>
</user>
```

### Preprocessing Rules (applied before sending to extraction model)

The raw JSONL contains a lot of noise. Preprocess to keep the extraction
model focused and within context window:

1. **Strip tool results with large payloads**: File contents, search results,
   command output > ~500 chars → replace with "[tool result: {tool_name},
   {byte_count} bytes]"
2. **Collapse thinking blocks**: Extended thinking → keep only the last
   paragraph or conclusion sentence. Or omit entirely.
3. **Remove system prompts**: The initial system message is boilerplate,
   not conversation content.
4. **Preserve role markers**: Keep clear Human/Assistant turn structure.
5. **Omit pure code blocks > 20 lines**: Replace with "[code block:
   {language}, {line_count} lines, filename: {path}]"
6. **Keep decision-laden exchanges intact**: If a turn contains words like
   "decided", "going with", "let's use", "prefer", "chose", preserve it
   in full.
7. **Keep casual/personal exchanges intact**: Tangential conversation about
   hobbies, jokes, asides — these are high-signal for personal engrams.
   Don't strip them as noise.
8. **Session length gate**: Skip sessions with fewer than 4 human turns
   entirely.

### Chunk Sizing

If a preprocessed session exceeds the extraction model's effective context
window (~6K tokens for quality extraction on an 8B model), split into
overlapping chunks with ~200 token overlap. Extract from each chunk
independently; the reconciler deduplicates downstream.

---

## Reconciliation Logic

After extraction produces candidate engrams, the reconciler validates them
against the existing store before persisting:

### 1. Deduplication
- Embed the candidate
- Search existing engrams with cosine similarity > 0.92
- If near-duplicate found: skip (don't store redundant beliefs)

### 2. Supersession Detection
- Query existing engrams with matching `entity` field
- If found: compare timestamps
  - If candidate is newer and content differs meaningfully
    (cosine similarity of content embeddings < 0.85):
    mark old engram's `superseded_by`, set candidate's `supersedes`
  - If content is ~identical: skip (already known)

### 3. Contradiction Detection (lightweight)
- For entity-matched existing engrams, pass both old and new content
  to the extraction model with a short prompt:
  "Do these two statements contradict? Respond yes/no."
- If contradiction detected: keep both as Candidate, flag for manual review
  (CLI output on next `engram list --pending`)

### 4. Confidence Gating
- Below 0.6: drop entirely
- 0.6–0.85: persist as Candidate (awaiting review or auto-promotion after
  N days without contradiction)
- Above 0.85: auto-promote to Accepted

### 5. Persist
- Generate embedding via nomic-embed-text
- Write to LanceDB with all fields populated

---

## MCP Tool Definition

Engram exposes a single, read-only MCP tool to Claude Code. All write
operations happen via the background reconciler or the CLI.

### engram_recall
```json
{
  "name": "engram_recall",
  "description": "Recall what you know about the user from past interactions. Returns relevant engrams — facts, preferences, decisions, interests, opinions, and patterns — drawn from the user's history across all Claude Code sessions. Use this to ground your responses in continuity with past work and conversation. Call at session start to orient yourself, or mid-session when you need context about the user's preferences, past decisions, or interests.",
  "parameters": {
    "query": {
      "type": "string",
      "description": "What you want to recall — a natural language description of the context, topic, or question"
    },
    "scope": {
      "type": "string",
      "enum": ["personal", "project"],
      "description": "Optional filter. 'personal' for user-level knowledge, 'project' for codebase-specific knowledge. Omit to search both."
    },
    "limit": {
      "type": "integer",
      "default": 5,
      "description": "Maximum number of engrams to return"
    }
  }
}
```

**Why read-only**: Claude Code already has robust project-scoped memory
via `/remember` and `CLAUDE.md`. Adding a competing write path through
engram creates ambiguity about where things should be stored. Engram's
value is the *implicit* knowledge that accumulates without conscious effort.
Explicit write tools (`engram_store`, `engram_forget`) are roadmapped but
deferred until v1 proves the concept and we understand the boundary better.

---

## CLI Management Commands

The user manages engrams directly through the CLI. This is intentional —
memory management is a human activity, not something the agent does
mid-session.

```
engram reconcile              # Run extraction pipeline (called by systemd)
engram serve                  # Start MCP server (stdio mode for Claude Code)

engram search <query>         # Semantic search, formatted output
engram list                   # List recent engrams (default: last 20)
engram list --pending         # Show Candidates awaiting review
engram list --scope personal  # Filter by scope
engram list --category humor  # Filter by category
engram list --entity solis    # Filter by entity key

engram show <id>              # Full detail view of a single engram
engram accept <id>            # Promote Candidate → Accepted
engram delete <id>            # Hard delete (no soft delete / archive)
engram promote <id>           # Change scope: Project → Personal
engram edit <id>              # Open content in $EDITOR for correction

engram forget <query>         # Semantic search + interactive bulk delete
                              # Shows matches, asks for confirmation

engram stats                  # Counts by scope, category, lifecycle
engram export                 # Dump all engrams as JSON (for backup)
```

---

## CLAUDE.md Instruction Block

Add to `~/.claude/CLAUDE.md` (user-level, applies to all projects):

```markdown
## Engram

You have access to `engram_recall` — a tool that retrieves knowledge about
the user drawn from past Claude Code sessions. This includes facts,
preferences, technical context, past decisions, interests, opinions, and
conversational patterns.

Call `engram_recall` at the start of a session with a brief description of
what we're about to work on. You can also call it mid-session when context
about past work, preferences, or decisions would be useful.

Engrams are separate from your built-in project memory. Use /remember and
CLAUDE.md for project-specific notes you're asked to store. Engrams are
background knowledge about the user that accumulates automatically — treat
them as things you already know, not as search results to present.
```

### Key phrasing choices:
- "things you already know" — encourages natural integration rather than
  "according to engram_recall result #3..."
- "background knowledge" — frames engrams as context, not citations
- No mention of store/forget — those aren't available via MCP in v1

---

## Triggering and Scheduling

Engram's reconciliation pipeline needs a trigger — something that bridges
"Claude Code wrote session data" to "engram reconcile runs." The design
uses systemd's native inotify support to achieve this with zero polling
and zero resident processes.

### Architecture

```
~/.claude/projects/   ──inotify──▸  engram-watch.path
                                         │
                                    (file changed)
                                         │
                                         ▼
                                  engram-debounce.timer
                                    (2-3 min quiet)
                                         │
                                         ▼
                                  engram-reconcile.service
                                    (oneshot: engram reconcile)
```

### systemd Units

**engram-watch.path** — Kernel-level file watch. No process runs; the
kernel wakes systemd when files change under the watched path.

```ini
[Unit]
Description=Watch Claude Code sessions for changes

[Path]
PathChanged=%h/.claude/projects
Unit=engram-debounce.timer

[Install]
WantedBy=default.target
```

**engram-debounce.timer** — Debounce mechanism. Waits for a quiet period
after the last write before triggering reconciliation. Each new .path
activation resets the timer. `OnActiveSec=2min` means reconcile fires
~2 minutes after the last session write (i.e., after the user stops
working for a bit).

```ini
[Unit]
Description=Debounce timer for engram reconciliation

[Timer]
OnActiveSec=2min
AccuracySec=30s
Unit=engram-reconcile.service
```

**engram-reconcile.service** — Oneshot service that runs the pipeline.

```ini
[Unit]
Description=Engram reconciliation pipeline

[Service]
Type=oneshot
ExecStart=%h/.cargo/bin/engram reconcile
Environment=RUST_LOG=engram=info
```

### Project Exclusions

Some projects should be excluded from extraction (e.g., work projects
with sensitive content, or projects belonging to other users on a shared
machine). Exclusions are configured in `config.toml`:

```toml
[reconcile]
# Project paths to exclude from extraction.
# Matched against the project path embedded in the JSONL directory name.
# Supports exact paths and glob patterns.
exclude_projects = [
    "/home/carlose/projects/ralph-trader",
    "/home/carlose/projects/milhouse",
]

# Minimum quiet period (seconds) before reconcile triggers.
# This is documentation for the systemd timer — the actual value
# is in engram-debounce.timer's OnActiveSec.
quiet_period_secs = 120
```

The exclusion filter is applied early in the pipeline — during session
discovery in `engram reconcile`, before any JSONL is read or processed.

### Installation

```bash
# Copy unit files
mkdir -p ~/.config/systemd/user
cp systemd/engram-watch.path ~/.config/systemd/user/
cp systemd/engram-debounce.timer ~/.config/systemd/user/
cp systemd/engram-reconcile.service ~/.config/systemd/user/

# Enable and start the file watcher
systemctl --user daemon-reload
systemctl --user enable --now engram-watch.path
```

### Why Not a Claude Code Hook?

Claude Code has a hook system that can run commands on events. We
considered using a post-session hook instead of systemd, but:
- Hooks run synchronously and block the session
- Reconciliation is slow (LLM inference) and should be background work
- systemd gives us debouncing, logging, failure handling, and `journalctl`
  integration for free
- The .path unit uses zero resources when idle (kernel inotify, not polling)

A Claude Code hook *is* used separately for session-start recall (see
Roadmap: Session-Start Hook), but not for triggering reconciliation.

---

## Directory Layout

```
~/.local/share/engram/
├── db/                     # LanceDB data directory
├── models/
│   ├── qwen3-8b.gguf      # Extraction model (or smaller, TBD)
│   └── nomic-embed.gguf   # Embedding model
├── state/
│   └── cursor.json         # Tracks last-processed position per project hash
└── logs/
    └── reconcile.log       # Processing log for debugging

~/.config/engram/
└── config.toml             # Exclusions, thresholds, model paths, quiet period
```

---

## Roadmap (post-v1)

### MCP Write Tools
- `engram_store` — explicit "remember this" from within a session
- `engram_forget` — "forget that" from within a session
- Deferred until the boundary between engram and Claude Code's native
  memory is better understood through real usage.

### TUI Browser
- `engram browse` — interactive terminal UI for exploring, filtering,
  and managing engrams. Replaces `list`/`show`/`delete` workflow with
  a single browseable interface. Build with ratatui if CLI management
  feels limiting.

### Session-Start Hook
- Use Claude Code's hook system to auto-run `engram_recall` at session
  start, writing results to a temp file that CLAUDE.md references.
  Eliminates dependence on the model choosing to call the tool.

### Decay and Archival
- Auto-flag engrams not accessed in N months for review.
- `engram archive` command to move stale engrams out of active search
  without deleting them.

---

## Open Questions

1. **Embedding model in-process vs subprocess**: Use llama-cpp-rs to load
   nomic-embed-text in the Rust binary directly, or shell out to
   llama-embedding? In-process is faster but adds build complexity with
   ROCm linkage.

2. **Chunk overlap strategy**: 200 tokens overlap between chunks may miss
   context that spans a chunk boundary. Alternative: summarize each chunk
   first, then extract from summaries? Adds latency but might improve
   extraction quality.

3. **Auto-promotion timing**: Candidates auto-promote to Accepted after
   N days without contradiction. What's a good N? 7 days? 14? Or should
   Candidates just stay Candidates until manually reviewed?

4. **MCP tool call frequency**: The CLAUDE.md instruction says "at session
   start." Should it also search mid-session when context shifts? More
   calls = more relevant context but also more token overhead and latency.

5. **Extraction model size**: Need to test Qwen3 4B vs 8B on real Claude
   Code JSONL samples to find the quality floor. If 4B is good enough,
   the reconciler runs faster and uses less VRAM.

6. **Humor/Interest decay**: These categories are more ephemeral than
   Decisions or Facts. Should they have shorter TTLs or lower retrieval
   priority by default? Or does that defeat the purpose — a joke from
   six months ago resurfacing naturally is exactly the kind of continuity
   that makes this feel alive.
