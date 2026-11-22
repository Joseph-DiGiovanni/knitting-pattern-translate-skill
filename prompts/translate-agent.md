# Translation brief (paste as subagent context; fill in W; give each subagent its own page list)

You translate pages of a craft pattern PDF for an in-place English edition. Files (W = work dir, NNN = zero-padded page number):
- `W/system.txt` — translation rules, glossary and book-specific names. Read it once and follow it exactly.
- `W/prompts/pNNN.txt` — one page: a header, then one tab-separated line per text box: `id`, `T`/`C` (translate / context only), orientation, bbox, source text.

For each assigned page:
1. Read `W/prompts/pNNN.txt`.
2. Produce JSON `{"units": [{"ids": [<int>, ...], "en": "<English>" | null}, ...]}` following `system.txt`: every `T` id in exactly one unit, ids in reading order, `C` lines never in the output.
3. Write it to `W/tr/pNNN.json` (UTF-8, no Markdown fences, nothing else in the file).

Do not read the PDF or page images; the prompt files have everything needed. Do not edit any other file.
Reply with one line only: `done <page numbers>` (or `failed <page>: <reason>`).
