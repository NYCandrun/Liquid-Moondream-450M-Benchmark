# Liquid AI Study Guide — The Edge AI Field Curriculum

A single self-contained HTML field-guide that compresses ~90 days of Liquid AI onboarding into roughly two weeks of focused reading. Built as one prose-first document — eight Units, recurring "six lenses," an integrated glossary drawer, and six alternate reading paths through the same material.

**Live file:** [`index.html`](./index.html) — open in any modern browser. No build step, no server, no external dependencies, no telemetry.

## What it is

`index.html` is the entire guide — HTML, CSS, JavaScript, and content in one file (~200KB). All progress and notes save locally to your browser via `localStorage`; nothing leaves your machine.

The guide is written like a textbook chapter — long-form prose with callouts, tables, timelines, and `<details>` deep-dives — not as a deck of discrete lessons. You read it top to bottom or jump by perspective.

## What's inside

- **8 Units** covering Liquid AI from the inside out:
  - **Unit 0 — Orientation:** how to use the guide, the three-phase company arc, six entry points.
  - **Unit 1 — The Technology:** why transformers struggle at the edge, the C. elegans / liquid-neural-network lineage, LFM2 architecture, STAR, the deployment stack, numbers worth memorizing, benchmark hygiene.
  - **Unit 2 — The Market:** the five forces making the edge-AI category, market sizing, scale-vs.-efficiency debate.
  - **Unit 3 — The Company:** history & timeline, the three-phase positioning, the full model catalog, platform and products, partnerships as proof, the license.
  - **Unit 4 — The Competition:** the five competitive rings, ten deployment environments, the May 2026 inflection, head-to-head battlecards.
  - **Unit 5 — Vulnerabilities:** eight honest weaknesses, the disclosure gap, what would break the thesis.
  - **Unit 6 — Narrative & Comms:** the message house, the Dec 2025 → Jun 2026 narrative arc, owned-vs.-earned coverage, the three executive voices.
  - **Unit 7 — The Six Lenses:** the lenses in full, question banks, the "you're ramped when…" rubric.
- **~92-term Lexicon** in a searchable drawer (right side), with inline dotted-underlined terms that pop a definition on click and link to the full entry.
- **Six reading paths** (top of the left rail) that re-sequence the same sections:
  Linear · By product · By competitor · By environment · Mental models · Vulnerabilities.
  When a non-linear path is active, a floating path bar appears at the bottom with prev / next / exit.
- **Six recurring lenses** — copper-bordered callouts that recur throughout the guide as a skeptical reading framework:
  1. Competitive-set selection IS the positioning.
  2. Whoever defines the trade-off wins the category.
  3. Production proof has a half-life.
  4. License choice is business-model choice.
  5. Information opacity has a brand cost.
  6. Beachheads are wedges, not moats.
- **Drill chips** — sideways links at the end of most sections into related material (e.g. from architecture to media coverage to license economics).
- **Deep-dives** — collapsed `<details>` blocks under most sections for readers who want to go further.
- **Personal layer (desktop)** — every section has a *Mark read* toggle and a private notepad. Both persist in your browser. The left rail shows progress count and per-section checkmarks. *Export* downloads notes and progress as a Markdown file.

## How to use it

1. Open `index.html` in a browser.
2. Start at Unit 0 (Orientation) or pick one of the six perspective tiles at the bottom of Unit 0 to jump into the discipline that matters most to you.
3. Click any dotted-underlined term to pop its glossary entry; click **⌘ Lexicon** in the top bar (or press `/`) to open the full searchable glossary.
4. Pick a reading path from the buttons at the top of the left rail to re-sequence the guide by perspective.
5. On desktop: mark sections read, jot per-section notes, and export when done.

### Keyboard shortcuts (desktop)

- `/` — open the Lexicon and focus search
- `g` — toggle the Lexicon drawer
- `j` / `k` — next / previous section in the active path
- `Esc` — close popovers and side panels

## Mobile

The guide is responsive and tested on small viewports. To stay clean on phones, desktop-only affordances are gracefully degraded rather than rendered half-broken:

- **Hidden on touch / ≤640px:** per-section notes textareas, mark-read toggles, progress counter, export & reset, reading-path picker, floating path bar, keyboard shortcuts.
- **Adapted for mobile:** topbar shrinks; nav and chip tap targets enlarge to ≥40px; perspective grid collapses to one column; the glossary popover becomes a bottom sheet; the Lexicon drawer takes the full viewport.
- **Preserved everywhere:** all unit prose, the inline glossary popover, the searchable Lexicon, tables (horizontal-scroll), deep-dive `<details>`, the scroll progress bar.

Capability detection uses `matchMedia('(min-width:641px) and (hover:hover) and (pointer:fine)')` plus `@media (hover:none) and (pointer:coarse)` so an iPad in landscape also gets the touch-tuned layout.

## Privacy

All notes and read-state are stored in `localStorage` under the key `liquid-edge-curriculum-v2`. The *reset* link in the left rail (desktop) wipes everything after confirmation. The *Export notes & progress* link downloads a Markdown copy first.

## Modifying the guide

This file is hand-editable end-to-end:

- Each Unit is a `<section class="unit">` with its own anchor (`#u0`, `#u1-transformer`, etc.). Add a section by writing markup; it automatically picks up the per-section *Mark read* / *Notes* controls (on desktop) and joins the linear sequence.
- Glossary lives in the `GLOSSARY = { … }` object near the bottom of the file. Each entry is `{ t: "Term", c: "Category", d: "Definition." }`. Reference one from prose with `<span class="t" data-t="key">surface text</span>`.
- Reading paths live in the `PATHS = { … }` object — each path declares `label` plus `seq` (an array of section IDs in path order).

## Source material

Synthesis of AI-generated research reports and external coverage on Liquid AI's architecture (LFMs, LFM2, STAR), products (LEAP, Apollo, Nanos), market position, competitive landscape (five rings, ten environments), comms strategy, and structural vulnerabilities, all dated to June 2026. The source reports are referenced, not bundled.

## Related

A companion app-style version of this material — gated, lesson-indexed, 14 modules / 97 lessons — lives at [`NYCandrun/liquid-ai-prep-course`](https://github.com/NYCandrun/liquid-ai-prep-course). The two are independent renderings of the same source synthesis: this repo is the prose textbook; that one is the interactive lesson app.
