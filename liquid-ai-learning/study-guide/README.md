# Liquid AI — Interactive Curriculum

A single self-contained HTML textbook that compresses a 90-day Liquid AI onboarding into roughly two weeks of focused study. Built from 14 source briefs covering the company's architecture, product surface, competitive landscape, customers, communications, and vulnerabilities.

## What it is

`Liquid_AI_Interactive_Curriculum.html` — one file, ~145KB, no external dependencies. Open it in any modern browser. All progress and notes save locally to your browser; nothing is sent anywhere.

## What's inside

- **14 modules, 97 lessons** — from the C. elegans origin story through the May 2026 agentic-edge inflection
- **~110-term glossary** in a persistent right-hand panel, with hover popovers and click-to-jump on every underlined term
- **Six reading paths** (top-left buttons): Linear · By Product · By Competitor · By Environment · Mental Models · Vulnerabilities
- **Six mental models** surfaced as recurring callouts in the lessons they apply to, consolidated in Module 12
- **Cross-links everywhere** — every lesson has a "Drill into" footer of related lessons
- **Personal layer** — mark-as-read, per-lesson notes (auto-saved), progress bar, search across lessons and glossary, export-to-Markdown of your notes
- **Gaps checklist** in Lesson 13.5 — the place to record what's still unclear and revisit it

## How to use it

1. Open the HTML file in a browser.
2. Start at Lesson 0.1 (How to use this textbook) or jump straight into one of the alternate paths.
3. As you read, use the textarea at the bottom of each lesson to capture what clicked and what didn't. Mark lessons read with the green button.
4. When you can answer the interrogation prompts in Module 13 without notes, you've hit competence.

### Keyboard shortcuts

- `g` — toggle glossary
- `/` — focus search
- `j` / `k` — next / previous lesson
- `Esc` — close popovers and side panels

## Modifying the curriculum

The file is designed to be hand-editable. All lesson content lives inside the inline `<script>` block at the bottom of the HTML, in a `MODULES` array. Each lesson is an object with `id`, `title`, `html`, and optional `drill` (array of related lesson IDs). The glossary lives in the `GLOSSARY` object above it.

To add a lesson: append an object to the appropriate module's `lessons` array. To add a glossary term: add a key/value pair to `GLOSSARY` and reference it from lesson HTML with `<span class="term" data-key="your-key">surface text</span>`.

## Source material

Built from 14 markdown briefs covering Liquid AI's architecture, products, market position, competitive landscape (5 rings, 10 deployment environments), comms strategy, external coverage, and vulnerabilities. The source files are not bundled in the HTML — only the synthesis.

## Privacy

All notes and progress are stored in `localStorage` under the key `liquidai-curriculum-v1`. Use the "Reset progress" button in the sidebar to wipe everything. Use "Export notes & progress" to download a Markdown copy first.

---

# Product Requirements Document

## 1. Problem and opportunity

New hires, prospective hires, investors, and partner-side engineers walking into a Liquid AI conversation typically need 60–90 days of ambient absorption before they can hold an opinionated discussion about the company. The substance is unusually technical (a non-Transformer architecture descended from C. elegans neuroscience), unusually commercial (five competitor rings, ten deployment environments, a specific Apple/AMD/automotive wedge), and unusually fast-moving (the May 2026 agentic-edge inflection re-priced the whole market in a quarter).

The gap is that the source material exists — internal briefs, external coverage, competitive analyses, comms playbooks — but it is spread across 14 documents in three different voices, with no shared vocabulary and no enforced reading order. Reading them cover-to-cover takes a week and leaves you with facts but no model.

This curriculum compresses that 90-day ramp into roughly two weeks of focused study by enforcing a vocabulary, a set of mental models, and a sequence — while still letting the reader enter from whichever angle they care about.

## 2. Goals and non-goals

### Goals

- Produce a credible interlocutor in ~2 weeks of focused reading, not 90 days of ambient exposure.
- Be a single self-contained artifact — one HTML file, no build step, no server, no external dependencies, no telemetry.
- Let the reader enter from six different angles (linear, by product, by competitor, by environment, by mental model, by vulnerability) and end up at the same level of competence.
- Encode six durable mental models that survive specific facts going stale.
- Make every technical term clickable, defined, and cross-linked so the reader is never stuck on vocabulary.
- Persist the reader's notes, progress, and gaps locally so the document becomes a working notebook, not a reference.

### Non-goals

- Not a marketing site, not a press kit, not investor collateral. It is internally honest about vulnerabilities and competitive threats.
- Not a live document. It is a point-in-time synthesis (June 2026) with explicit dating on time-sensitive claims.
- Not a video course, not a quiz app, not a multi-user system. There is no backend, no auth provider, no analytics.
- Does not bundle the 14 source briefs — only the synthesis. The briefs are referenced, not embedded.

## 3. Target users

| Persona | What they need | Entry path |
|---|---|---|
| New technical hire | Architecture, training pipeline, license model, why LFM2 beats transformer baselines at their size | Linear, then By Product |
| New GTM / sales hire | Five competitor rings, ten deployment environments, which deals Liquid actually wins | By Competitor, then By Environment |
| Investor / board prep | Three-phase story, vulnerabilities, May 2026 inflection, what would invalidate the thesis | Mental Models, then Vulnerabilities |
| Partner-side engineer (AMD, automotive, Apple-silicon dev) | Specific product surface, integration points, what is real today vs. roadmap | By Product, then By Environment |
| Comms / external-facing function | Anchor sentence, narrative phases, what coverage gets right and wrong | Modules 8–9, then Mental Models |

## 4. Scope

14 modules · 97 lessons · ~110 glossary terms · 6 mental models · 6 reading paths · 1 gaps checklist.

Lessons are short (typical 200–500 words, sometimes a table or callout) and atomic — one lesson teaches one thing. Drill-into chips at the bottom of each lesson point to 2–4 related lessons.

## 5. Information architecture

### 5.1 Modules

| # | Module | Lessons | What it answers |
|---|---|---|---|
| 0 | How to use this textbook | 4 | How the document works, what mastery looks like, how the glossary behaves |
| 1 | Origins and scientific foundations | 7 | C. elegans, liquid time-constant networks, the MIT CSAIL lineage, why this is not a Transformer |
| 2 | Architecture deep dive | 9 | LFM family, LFM2 / LFM2-VL, hybrid linear-conv layers, no-KV-cache design, training pipeline |
| 3 | The product surface | 10 | LEAP, LiquidKit, Apollo iOS app, Nanos, what ships today vs. roadmap |
| 4 | Market position and the three-phase story | 8 | The on-device phase, the agentic-edge inflection, the structural advantages (AMD moat, flagship proofs) |
| 5 | The five competitive rings | 7 | Ring 1 platform vendors, Ring 2 edge specialists (the real fight), Rings 3–5 |
| 6 | Deployment environments (10-slice matrix) | 10 | Phones, PCs, automotive, robotics, wearables, embedded, sovereign, regulated, dev tools, browser |
| 7 | Customers and proof points | 7 | AMD, automotive OEMs, Apollo, sovereign pilots — what is contracted vs. demo'd |
| 8 | Communications, coverage, and voice | 7 | Comms strategy, what TechCrunch / The Information / analysts say, what they miss |
| 9 | Differentiation and the anchor sentence | 4 | The one-sentence pitch, why the competitor framing matters more than benchmarks |
| 10 | Vulnerabilities and threats | 9 | Six structural risks, including the beachhead-vs-moat problem and platform-vendor encroachment |
| 11 | May 2026 inflection — agentic edge | 4 | Why the market re-priced, what Liquid did right, what's now contested |
| 12 | The six mental models (consolidated) | 6 | All six mental models in one place, with the lessons that exemplify each |
| 13 | Interrogation prompts and gaps | 5 | Self-test prompts, the gaps checklist, what to revisit |

### 5.2 The six reading paths

- **Linear** — module 0 through 13 in order. Best for new hires who want the full arc.
- **By Product** — LEAP, LiquidKit, Apollo, Nanos, LFM2 family. Best for engineers integrating against the product.
- **By Competitor** — the five rings, in priority order. Best for GTM and competitive prep.
- **By Environment** — the 10 deployment slices. Best for partners and BD.
- **Mental Models** — the six durable abstractions and the lessons that exemplify them. Best for investor / board prep.
- **Vulnerabilities** — the structural risks and how they manifest. Best for honest internal review and red-team prep.

### 5.3 The six mental models

Each one surfaces as a recurring callout in the lessons it applies to, and gets a consolidated treatment in Module 12.

1. **Set selection IS positioning.** Choosing which competitor set you belong to is the most consequential strategic move — more than benchmarks within a set.
2. **Defining the trade-off wins.** Whoever frames the axis (latency vs. quality, privacy vs. capability, on-device vs. cloud) controls the comparison.
3. **Production proof has a half-life.** A shipped customer is worth more than a benchmark, but only for ~12 months — narratives have to refresh.
4. **License choice is business model choice.** Apache 2.0 vs. LFM Open License vs. commercial — each implies a different revenue model and a different competitive surface.
5. **Information opacity has a brand cost.** What you don't disclose (training data, eval methodology, customer names) is read as weakness, not discretion.
6. **Beachheads ≠ moats.** Every Liquid foothold (automotive, AMD PCs, Apple-silicon dev, sovereign) is a wedge, not a permanent defense — the white space closes at platform-vendor pace.

## 6. Key user flows

1. **First-time read.** Open the file → unlock the gate → land on Lesson 0.1 → read modules 0–13 linearly over ~10 sessions → fill in the gaps checklist in 13.5 → run the interrogation prompts in 13.1–13.4 unaided.
2. **Targeted deep dive.** Open the file → pick a reading path → read only the lessons in that path → use drill-into chips to follow tangents.
3. **Reference lookup.** Open the file → press `/` → search lessons and glossary → click an underlined term anywhere to pin its popover or jump to its glossary entry.
4. **Working notebook.** As the reader goes, write per-lesson notes in the textarea (auto-saved), mark lessons read, watch the progress bar advance. Export the notebook to Markdown when done.

## 7. Functional requirements

### 7.1 Content

- 14 modules, 97 lessons, ~110 glossary terms, 6 paths, 6 mental models, gaps checklist.
- Every technical term that appears in a lesson must be defined in the glossary and clickable inline.
- Every lesson must declare 2–4 drill-into targets where applicable.
- Time-sensitive claims (prices, valuations, customer lists, employee counts) must be dated.

### 7.2 Reading experience

- Three-column desktop layout: sidebar nav · lesson body · glossary pane.
- Path-toggle buttons re-render the sidebar to match the chosen path.
- Hover or click on any underlined term pops a definition; click pins it.
- Keyboard shortcuts: `g` glossary, `/` search, `j`/`k` next/prev lesson, `Esc` close panels.
- Mental-model callouts use a distinct teal style and are tagged so they can be filtered.

### 7.3 Personal layer

- Mark-as-read per lesson with progress percentage.
- Per-lesson notes textarea, auto-saved on input.
- Export notes + progress as a single Markdown file.
- Reset progress wipes localStorage with confirmation.

### 7.4 Access control

- Inline password gate (hardcoded password) plus a refreshable math captcha.
- Failed-attempt cooldown after 5 misses.
- Session-storage unlock so a tab refresh does not re-prompt.
- App is blurred and uninteractive until the gate is cleared.

### 7.5 Mobile

- Responsive at iPhone widths (393×852 validated).
- Sidebar and glossary become left/right drawers with a scrim overlay.
- Sticky mobile bar with hamburger, current lesson title, and glossary toggle — pinned above drawers so it stays usable.
- Inputs use 16px font to avoid Safari auto-zoom.
- Safe-area insets honored for notch and home indicator.
- Wide tables scroll horizontally inside their container; stat grids drop to 2 columns.

### 7.6 Portability

- Single HTML file, no external assets, no network calls, no analytics.
- All state in `localStorage` under `liquidai-curriculum-v1` and `sessionStorage` under `liquidai-curriculum-unlock`.
- Plain text editable — content lives in the inline `<script>` blocks.

## 8. Non-functional requirements

- **Size.** Single file under ~200KB.
- **Performance.** First paint of the unlocked view under 200ms on a modern laptop.
- **Offline.** Fully functional offline after the file is on disk.
- **No telemetry.** No fetches, no beacons, no third-party scripts.
- **Editability.** A non-developer should be able to add a lesson by appending an object to the `MODULES` array.
- **Accessibility.** Semantic landmarks, focus management on the gate, ARIA on dialogs and live regions, keyboard navigability.

## 9. Success metrics

Measured per reader, on the honor system (no instrumentation by design):

- Time-to-competence: can answer the Module 13 interrogation prompts unaided in under 2 weeks of part-time reading.
- Vocabulary fluency: can read external coverage of Liquid AI without needing to look up terms.
- Mental-model recall: can name the six mental models and give an example of each.
- Critical reading: can articulate at least three structural vulnerabilities without prompting.
- Personal notebook: completed gaps checklist with concrete follow-up questions, not vague ones.

## 10. Out of scope (for v1)

- Multi-user accounts, server-side persistence, cloud sync.
- Live data (real-time competitor moves, current pricing).
- Embedded video or audio.
- Quiz / scoring engine.
- Real auth — the password gate is a soft access control, not a security boundary.
- Translation / localization.

## 11. Open questions and follow-ups

- Refresh cadence: time-sensitive lessons (Module 11 especially) will go stale. Quarterly refresh or rolling?
- Source-brief linking: should the underlying 14 briefs be bundled in a `/sources` panel for traceability, or kept separate?
- Editorial review: who signs off on changes to the anchor sentence (Module 9) and the vulnerabilities (Module 10)?
- Versioning: should the document carry a visible version string and changelog modal?

## 12. Design choices worth flagging

- **Honesty over polish.** Module 10 (Vulnerabilities) is as long as Module 3 (Product surface) on purpose. A document that hides risks fails its job.
- **One file, no build.** The cost of a build pipeline (Webpack, React, hosting) is higher than the cost of editing a flat HTML file. The latter also survives leaving the company.
- **Local-only persistence.** Notes are private to the reader. A shared notebook is a different product.
- **Soft access control.** The password gate is friction, not security. Anyone who views source can read the content. That's acceptable for the use case; real distribution control would be a different product.
- **Mental models as the durable layer.** Specific facts will go stale within a year. The six mental models are designed to survive that and still be useful.
