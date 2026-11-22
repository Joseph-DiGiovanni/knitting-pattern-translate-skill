---
name: knitting-pattern-translate
description: >
  Translate a Japanese or Korean knitting/crochet (or similar craft) pattern PDF into English IN PLACE, keeping
  every photo, chart and diagram: exact text-layer extraction or OCR (outlines/scans), side-model translation,
  redact/cover-and-redraw rendering, then cheap targeted QA. Use when asked to translate a Japanese or Korean
  pattern book, magazine, leaflet or designer pattern (PDF) to English, e.g. "translate this Japanese/Korean
  knitting PDF", "English version of this pattern". Also for re-running or fixing a previous translation, or
  adding another source language.
---

# Knitting pattern translate

`KPT=<this skill's directory>/scripts/kpt` (the folder holding this SKILL.md; bash wrapper, first run builds a
private Python 3.12 venv in `scripts/.venv` with rapidocr/onnxruntime/pymupdf/opencv via `uv`, ~1 min). All state
lives in a work dir `W`; every step is resumable and prints one-line summaries. Languages: `ja`, `ko` (table `LANGS`
in `kpt.py`).

## Two text sources (chosen per page by `ocr`, automatic)
- **Text layer** (PDFs exported from layout or design software): exact text, font size, colour, weight, slant,
  underline; source glyphs are removed by redaction (backgrounds untouched), English drawn in the same style.
  Takes seconds; no OCR risk sheet needed.
- **OCR** (text converted to vector outlines, or scanned pages): multi-pass RapidOCR; glyphs covered by flat fill
  or inpaint. Needs the risk-sheet step.

## Token budget rules
- Never `read` the source PDF or whole-page images in the main context. Translation runs outside the main
  context (side model or subagents, step 3); page-level visual QA runs in subagents; the main context sees
  summaries and finding lines only.
- Images you view yourself: only `qa/s*.png` sheets (risky OCR crops), `crop` zooms, and at most 1–2 rendered
  pages right after the first render to catch systematic problems. One look at the contents/intro page
  (`W/orig/pNNN.png`) or its text (`get_text`) is allowed to write `book.md`.
- Reuse images: every QA finding names the crop its subagent viewed (`W/crops/…`, translation over original in
  one image). To judge a finding or a `REVIEW` line, open that file; do not crop the same area again. `crop`
  caches by page and rect, so asking for the same rect returns the existing file until that page is re-rendered.
- Inspect units with `$KPT units -w W N…` (source, current English, fixes/layout flags; before translation it
  lists the source lines) instead of reading the work-dir JSON; measure with `ink` and `width` (step 7) instead
  of ad-hoc scripts.
- Render each page as few times as possible: after the first full render, never run another full `render`
  except the final one (step 8). Re-render only changed pages (`render -p …`), and when QA touched most pages
  skip that partial pass: render the final PDF and spot-check it with `crop --pdf <final>`.
- Do not re-audit edits by reading files back; command output is the verification.

## Workflow
1. **Init + extract**: `$KPT init book.pdf -w W` (language auto-detected from the text layer; pass `-l ja|ko` for
   scans) then `$KPT ocr -w W` (text pages: seconds; OCR pages ≈5–10 min/100 pages; `--fast` skips tiled passes).
2. **book.md** (optional, cheap, improves consistency): titles, yarn/brand names, designer names, one per line
   `source = English`. Use brands' and products' own English names where they exist (look them up).
3. **Prompts + translate**: `$KPT prompts -w W` (glossary `prompts/glossary-<g>-<lang>.md`, `-g` to pick another
   `<g>`). It writes `W/system.txt` (rules + glossary) and one `W/prompts/pNNN.txt` per page. Translate them
   outside the main context, whichever your agent supports:
   - **Side-model helper** (agents with a Python eval kernel exposing `completion()`, e.g. oh-my-pi): two separate
     cells `%load <skill>/scripts/translate_eval.py` then `translate('W')`; `translate('W', [5, 9])` redoes pages.
   - **Subagents** (Claude Code Task, Codex, Copilot, Gemini CLI, Cursor…): spawn parallel subagents, ~4–8 pages
     each, with `prompts/translate-agent.md` as their brief (W filled in). They write `W/tr/pNNN.json` themselves
     and reply only "done".
   - **No subagents** (e.g. pi without a subagent extension): follow `prompts/translate-agent.md` yourself, 3–5
     pages per batch, writing `W/tr/pNNN.json` directly; never echo the JSON into the conversation.
   Then `$KPT validate -w W` (every translatable line covered exactly once; lists pages to redo) and
   `$KPT unify -w W`: labels repeated across pages (legends, headings, notes) get the majority translation
   (fixes.json; never overrides a fix).
4. **OCR risk sheet** (OCR pages only; prints 0 items for text-layer books): `$KPT qa-sheet -w W`, view
   `W/qa/s*.png`. Each tile is `page:id 「OCR text」` + crop. Fix misreads in one call:
   `$KPT correct -w W <page>:<id>='<corrected text>' <page>:<id>=-` (`-` = not text). Typical misreads: vertical
   text with digits loses or merges characters; check against the crop. Then `$KPT checked -w W` and translate the
   missing pages again (step 3; `correct` rebuilt those pages' prompts and dropped their translations).
5. **Render**: `$KPT render -w W --png 100` → `W/out.pdf` + `W/png/pNNN.png`. It prints units it left in the
   source language (English still contained source letters; hashtags are allowed) — fix those in `fixes.json`.
6. **Residual scan**: `$KPT resid -w W` lists source text still present as `page | bbox | text` (`[text]` = text
   layer, exact; others = OCR of pixels, e.g. text inside photos). `--no-ocr` for a seconds-fast text-only check.
   Turn real ones into findings lines (step 7 format) and `apply`.
7. **Visual QA in subagents** (most cost, most quality; wall time scales with pages per agent). Spawn enough
   subagents for ~4–6 pages each (fewer for dense diagram pages), all in one batch; give pages that share a
   layout (size variants of one chart, repeated legends) to the same agent, which checks the first fully and
   the rest by difference. Context = `prompts/qa-agent.md` with `W`, `KPT`, `{LANG}` filled in, pages that have
   translations only (`ls W/tr`). The brief checks meaning, style and layout (alignment, containment, overlap
   with artwork, attached leaders and underlines, consistent paragraph sizing, text shrunk despite free space,
   legends that only repeat their symbol) using `units`, pair `crop`s, `ink` (tight ink bbox) and `width`
   (wording width in the unit's font). Each finding line ends with the crop that shows the problem. Save each
   agent's reply to its own file and apply them together: `$KPT apply -w W W/qa_*.txt` (files are joined line
   by line; their final newlines do not matter). It maps SRC→`manual.json` (blank cover if
   English empty), WRONG→`fixes.json` (patching only the quoted part of multi-line units, or the whole unit
   when the correction rewrites it; the quoted English picks the unit when bboxes are ambiguous; a
   "correction" that only appends a bracketed remark, or repeats the current text, is refused), ALIGN/FIT→`layout.json`
   (alignment override / exact free box the English is fitted into, wrapped when two lines tall; a re-flowed
   paragraph stays inside it too),
   STYLE→`layout.json` (whole-unit weight, slant, colour), MERGE→`tr/` (units inside the bbox become one
   re-flowed paragraph, inline graphics as `[[text]]`), DAMAGE→`restore.json` (original artwork redrawn on top —
   on text-layer pages that also brings back source text inside the box, so keep DAMAGE boxes clear of text);
   image paths are ignored. Before applying, read the lines: a finding whose fix field holds a description
   instead of the value its type expects goes through the matching type or a hand edit, never into
   `fixes.json` as text. Resolve `REVIEW` lines (printed with their crop) by viewing that crop and editing
   `fixes.json`/`layout.json`/`tr/` yourself. Then re-render touched pages (`$KPT render -w W -p 3,7 --png
   100`; `crop` reads that newer preview) — or, when most pages were touched, go to step 8 directly — and
   spot-check one or two fixes with `crop --trans` on the finding's rect. One QA round usually suffices.
8. **Final**: `$KPT render -w W -o "<dir>/<name>_English.pdf"` then one `$KPT resid -w W --pdf <that>`
   (spot checks on it: `crop --pdf <that>`).

## Work-dir files
`config.json` (src, pages, lang, glossary) · `ocr/pNNN.json` (a JSON list of items, id = list index: bbox pt,
text, score, src `pdf`|OCR pass, tl = translate, v = vertical; text-layer items add size, color, bold, italic,
underline, rot, origin, inline =
enclosing shape of an inline graphic, inline_marks = rules under that shape, styled = text with `<sN>…</sN>`
around runs styled unlike the line, styles = `{N: {bold, italic, color, size, underline}}`) ·
`r300/` `orig/` `crops/` (cached `crop` zooms) page renders · `prompts/` + `system.txt` · `tr/pNNN.json`
(`{"units":[{"ids":[..],"en":..}]}`) · `fixes.json` (`{page:{"ids":"English"|""|null}}`; `""` removes without
text, `null` leaves the source) · `manual.json` (`{page:[{bbox,en,v,solid}]}` extra covers/labels) ·
`restore.json` (`{page:[bbox]}`) · `layout.json` (`{page:{"ids":{"align":"left|center|right",
"box":[x0,y0,x1,y1],"style":{"bold":…,"italic":…,"color":[r,g,b]}}}}`).

## What the renderer already handles (do not re-fix by hand)
Text layer: per-glyph redaction (neighbours survive), exact size/bold/italic, colour as rendered (text that looks
unlike its declared fill, e.g. through transparency, keeps the look); runs styled unlike their line (bold words,
coloured figures, small notes, underlined words) are tagged `<sN>` in the prompt, the translator tags the
matching English and the renderer draws those words in that style, underlines with the original pen (dash,
width, colour); labels stacked in a column with other lines (tables, label lists) keep the shared
left/right/centre edge; paragraphs (multi-line units, units with inline `[[graphics]]`, long lines with blank
space below) re-flowed as one block at one size: blank lines below (same background only, so text never runs
out of a tinted panel; keeping the paragraph's own line gap to the next block), then tighter line spacing, then
blank space to the right are used before the size drops, and inline
graphics (small shapes or glyphs within running text) move with their word, together with rules drawn right
under them; graphics attached to re-flowed words (leader lines or arrows from an underline or inline graphic to
a note) never stay behind: leader and note move together when the new spot is clear; else the note is laid out
again there as one line in the words' column; else the leader is stretched from the words to the unmoved note;
else both move with the words anyway and render prints `pN: leader and note moved … check it`; lines split at wide gaps
keep other elements between the pieces; letter-spaced headings; angled lines redrawn on their baseline and shrunk
to the source's extent; text centred between graphics stays centred; underlines under replaced single-line
labels painted out and redrawn as plain underlines under the tagged English.
OCR: flat-fill vs inpaint per box (photos, gradients, coloured shapes); text colour/polarity; vertical labels
rotated; Japanese diagram conventions (vertical row counts merged with a neighbouring length, "symbol = name"
legends keeping their swatches and symbols, markers that OCR dropped kept visible); leader-line `-`/`/` stripped.
Both: single-line labels grow into blank space (up to half way to the page edge), then wrap onto two lines in
blank space above/below, before shrinking; full-width glyph widths; fonts subset (output ≈ source size).

## Adding a language
Add a `LANGS` entry in `kpt.py` (script regexes, OCR-junk symbol set, one-character labels, RapidOCR rec model)
and `prompts/glossary-knitting-<lang>.md`. `system.md` and `qa-agent.md` use `{LANG}`.

## Known limits
- Text inside photos/raster images is not in the text layer; `resid` OCR finds it, fix via SRC findings.
- OCR misses some tiny labels in dense diagrams and letter-spaced headings; residual scan + QA catch the rest.
- English longer than the source shrinks in crowded diagrams with no blank space nearby; QA reports < ~3pt.
- Re-flow needs the text layer; OCR'd paragraphs are drawn box by box.
- Dashed underlines keep their dash pattern only in re-flowed paragraphs; single-line labels get a solid one.
  OCR'd text has no style runs: it is drawn in one weight and colour per box.
- Attached leaders are recognised as small paths touching an underline or inline graphic of a re-flowed
  paragraph; a pointer that starts next to plain (not underlined) words, or long arrows across a diagram, stay
  where they are.
