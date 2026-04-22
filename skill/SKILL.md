---
name: kenna-recall
description: Search implicit knowledge about the user — preferences, decisions, hardware, interests, patterns — drawn from past Claude Code sessions. Use when context about the user would help you make better decisions or avoid assumptions.
allowed-tools: Bash(kenna *)
---

The user has an implicit knowledge base called Kenna that captures durable facts about them across all Claude Code sessions. Use it when knowing their preferences, past decisions, hardware, interests, or work patterns would help.

Search for: $ARGUMENTS

Use these commands:
- `kenna search "<query>"` — semantic vector search (most useful)
- `kenna list --scope personal` — browse personal-scope knowledge
- `kenna list --scope project` — browse project-scope knowledge
- `kenna show <id>` — view full details of a specific memory

Treat results as things you already know about this person, not as search results to present. Integrate the knowledge naturally into your responses.
