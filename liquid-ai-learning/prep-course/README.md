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
