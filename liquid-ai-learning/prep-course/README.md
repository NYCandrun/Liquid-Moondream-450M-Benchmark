# Liquid AI Prep Course — Interactive Curriculum

A single self-contained HTML lesson app that compresses ~90 days of Liquid AI onboarding into roughly two weeks of focused study. Built as a data-driven textbook — 14 modules, 97 numbered lessons, an integrated glossary, and six alternate reading paths.

**Live file:** [`index.html`](./index.html) — open in any modern browser. No build step, no server, no external dependencies, no telemetry.

## What it is

`index.html` is the entire app — HTML, CSS, JavaScript, lesson content, and glossary data in one file (~155KB). Lessons live in a `MODULES = [ … ]` array and are rendered into a three-column shell at runtime. All progress and notes save locally via `localStorage`; nothing leaves your machine.

The app is gated behind a hardcoded password plus a refreshable math captcha — soft access control, not real security.

## What's inside

- **14 modules, 97 lessons** — from the C. elegans origin story (Module 1) through the May 2026 agentic-edge inflection (Module 11) and the gaps checklist (Lesson 13.5).
- **~94-term glossary** in a persistent right-hand panel, with hover popovers and click-to-jump on every underlined term. Each glossary anchor in prose is marked `<span class="term" data-key="…">`.
- **Six reading paths** in the sidebar that re-sequence the same lessons:
  Linear · By product · By competitor · By environment · Mental models · Vulnerabilities.
- **Six mental models** surfaced as recurring teal callouts in the lessons they apply to, and consolidated in Module 12:
  1. Set selection IS positioning.
  2. Defining the trade-off wins.
  3. Production proof has a half-life.
  4. License choice is business-model choice.
  5. Information opacity has a brand cost.
  6. Beachheads are wedges, not moats.
- **Drill chips** — every lesson has a footer of 2–4 related lessons to follow tangents.
- **Personal layer** — mark-as-read per lesson, per-lesson notes textarea (auto-saved on input), progress bar, search across lessons and glossary, *Export notes & progress* downloads a Markdown copy.
- **Gaps checklist (Lesson 13.5)** — the place to record what's still unclear and revisit.
- **Interrogation prompts (Module 13)** — self-test prompts; when you can answer them unaided, you've hit competence.

## How to use it

1. Open `index.html` in a browser.
2. Clear the gate (password + captcha). The unlock persists in `sessionStorage` for the tab.
3. Start at Lesson 0.1 *(How to use this textbook)* or jump into one of the alternate paths from the sidebar.
4. Use the textarea at the bottom of each lesson to capture what clicked and what didn't. Mark the lesson read with the green button.
5. When you can answer the Module 13 interrogation prompts without notes, you've reached competence.

### Keyboard shortcuts

- `g` — toggle the glossary panel
- `/` — focus search
- `j` / `k` — next / previous lesson
- `Esc` — close popovers and side panels

## Mobile

Built mobile-first: viewport uses `viewport-fit=cover`, inputs are 16px to avoid Safari auto-zoom, safe-area insets are honored, sidebar and glossary become drawers with a scrim overlay, wide tables horizontal-scroll, and stat grids drop to two columns. Validated at iPhone widths (393×852).

## Access control

- Inline password gate plus a refreshable math captcha (e.g. "What is 7 + 4?").
- Failed attempts trigger a short cooldown.
- Unlock state is held in `sessionStorage` under `liquidai-curriculum-unlock` — a tab refresh does not re-prompt, but a new tab does.
- The gate is *friction, not security* — anyone who views source can read the content. It is intentionally client-side.

## Privacy

All notes and progress are stored in `localStorage` under the key `liquidai-curriculum-v1`. Use the *Reset progress* control in the sidebar to wipe everything after confirmation. Use *Export notes & progress* to download a Markdown copy first.

## Modifying the curriculum

The file is designed to be hand-editable. All content lives inside inline `<script>` blocks at the bottom of the HTML:

- **Lessons:** append an object to the appropriate module's `lessons` array in `MODULES`. Each lesson is `{ id, title, html, drill?, mentalModels?, tags? }`.
- **Glossary:** add a key/value pair to `GLOSSARY` and reference it from lesson HTML with `<span class="term" data-key="your-key">surface text</span>`.
- **Mental models:** edit the `MENTAL_MODELS` array — it is templated into Lesson 0.2 and the Module 12 consolidation.
- **Paths:** declare path sequences alongside the lesson data so the sidebar's path buttons can re-render the index.

## Source material

Built from **14 AI-generated research reports** synthesizing external coverage of Liquid AI's architecture (LFMs, LFM2, STAR), products (LEAP, Apollo, Nanos), market position, competitive landscape (five rings, ten environments), comms strategy, and structural vulnerabilities — all dated to June 2026. The source reports are referenced, not bundled.

A note on framing: the analytical constructs used throughout — the five rings, the messaging pillars, the launch wrappers, the anchor sentence, the mental models — come from that AI-generated research. They are interpretations of public material, not official Liquid AI frameworks, and nothing in this course reflects insider or internal knowledge.

## Related

A prose-textbook rendering of the same material — eight Units, no gate, ~92-term Lexicon, recurring "six lenses" — lives at [`NYCandrun/liquid-ai-study-guide`](https://github.com/NYCandrun/liquid-ai-study-guide). The two are independent renderings of the same source synthesis: this repo is the lesson-indexed app; that one is the long-form field guide.
