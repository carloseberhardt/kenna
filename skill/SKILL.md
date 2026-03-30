---
name: engram-recall
description: Search implicit knowledge about the user — preferences, decisions, hardware, interests, patterns — drawn from past Claude Code sessions. Use when context about the user would help you make better decisions or avoid assumptions.
allowed-tools: Bash(/home/carlose/.cargo/bin/engram *)
---

The user has an implicit knowledge base called engram that captures durable facts about them across all Claude Code sessions. Use it when knowing their preferences, past decisions, hardware, interests, or work patterns would help.

Search for: $ARGUMENTS

Use these commands:
- `/home/carlose/.cargo/bin/engram search "<query>"` — semantic vector search (most useful)
- `/home/carlose/.cargo/bin/engram list --scope personal` — browse personal-scope knowledge
- `/home/carlose/.cargo/bin/engram list --scope project` — browse project-scope knowledge
- `/home/carlose/.cargo/bin/engram show <id>` — view full details of a specific engram

Treat results as things you already know about this person, not as search results to present. Integrate the knowledge naturally into your responses.
