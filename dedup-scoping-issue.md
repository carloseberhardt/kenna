# Dedup scoping defeats cross-project promotion

**Status:** identified, not yet fixed. Orthogonal to the Turso migration.

## Symptom

`kenna settle` reports "Nothing to settle" even when the user clearly exhibits a
trait across many projects. Cross-project **promotion** (project-scoped memories
→ personal scope) effectively never fires for strongly/consistently expressed
traits.

## Root cause

Reconcile-time dedup is **global across all projects**, but promotion counts
**distinct source projects**. The first destroys the signal the second needs.

- `pipeline/reconcile.rs` dedup step calls
  `db.vector_search(embedding, 1, None)` — the scope filter is `None`, and even
  when set it filters on the `Scope` enum (`personal`/`project`), **not** on
  `source_project`. So a new candidate is compared against the *entire* store.
- If similarity > `dedup_cosine_threshold` (~0.85), the candidate is
  `DroppedDuplicate` **before being stored**. Only the first occurrence
  survives, carrying whichever `source_project` was processed first.
- `pipeline/settle.rs` promotion clusters project-scoped memories at
  `cluster_cosine_threshold` (0.75) and promotes a cluster only if it spans
  `min_projects_for_promotion` (3) **distinct `source_project` values**.
- Because global dedup keeps at most one row per near-identical trait, that
  cluster's distinct-project count is structurally pinned at **1** → it can
  never reach 3 → it never promotes.

**Consequence:** the strongest cross-project signal — the same trait stated the
same way in every project — is exactly the one promotion is blind to. Promotion
can only ever fire in the narrow **0.75–0.85** band (similar enough to cluster,
worded differently enough across projects to survive dedup). Extractor/curator
phrasing convergence pushes genuine repeats *above* 0.85, shrinking that band
further.

**Not time-dependent.** This is not rescued by "express it in different runs /
on different days." Once project A's row is stored it persists and keeps
suppressing project B's, C's, … on every future run. Only wording *drift* into
the 0.75–0.85 band helps; elapsed time only matters as a proxy for "worked in
more projects."

## Why it's a design-layer bug

Cross-project consolidation is **settle's** job — that is what the entire
promotion / Union-Find clustering machinery exists to do. Reconcile-time global
dedup is doing settle's job prematurely and consuming settle's input in the
process. Reconcile dedup should only suppress **intra-project reprocessing
noise** (the same session chunk seen twice, the same project re-extracted), not
collapse signal across project boundaries.

## Proposed fix

Scope dedup so cross-project occurrences coexist:

- A new **project**-scoped candidate dedups only against **same-project** rows,
  plus optionally against **personal** rows (so we don't re-hoard a trait that
  has already been promoted).
- **Personal** dedups against personal.
- Cross-project near-duplicates are allowed to **coexist** as separate rows —
  precisely the multiplicity promotion counts. Settle then consolidates them.

## Implementation notes / gotchas

- **Not a one-line param flip.** `vector_search` filters by the `Scope` enum,
  not by `source_project`. To dedup within the same project, either:
  - add a `source_project` filter to the dedup query path, or
  - retrieve candidates and match `source_project` in-app.
- **Supersession is also global.** The entity-based supersession step right
  after dedup (`reconcile.rs`, ~step 5) also searches across projects. It
  probably wants the same treatment: supersession ("Prefers Vim → Neovim") is a
  within-context temporal replacement; across projects the rows should coexist
  so promotion can see them.
- **Transient duplication between runs.** Until the next `settle`, recall can
  surface several near-identical project rows for one trait. Mitigations:
  settle runs daily; recall (`kenna_recall`) could dedup at query time. Name
  this cost explicitly — it's the price of moving consolidation to the layer
  that owns it.

## Tests to add

- Insert the same (near-identical) trait under 3 distinct `source_project`
  values; assert all 3 rows persist (no cross-project `DroppedDuplicate`).
- Assert intra-project near-duplicate is still deduped.
- End-to-end: 3-project trait → `settle` promotes it to personal scope and
  supersedes the 3 sources (exercises `apply_settlement`).
