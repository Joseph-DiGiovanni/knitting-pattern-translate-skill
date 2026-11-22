# Side-model translation for agents whose Python eval kernel provides `completion()` (e.g. oh-my-pi).
# Other agents: use subagents with prompts/translate-agent.md instead (see SKILL.md step 3).
#   %load <skill dir>/scripts/translate_eval.py
#   translate('/path/to/work')            # all untranslated pages
#   translate('/path/to/work', [38, 39])  # redo specific pages
# Translation runs on a side model; only a one-line summary reaches the main context.
import json as _json, os as _os, glob as _glob

_SCHEMA = {"type": "object", "properties": {"units": {"type": "array", "items": {"type": "object", "properties": {
    "ids": {"type": "array", "items": {"type": "integer"}}, "en": {"type": ["string", "null"]}},
    "required": ["ids", "en"]}}}, "required": ["units"]}


def _tids(txt):
    out = set()
    for line in txt.splitlines()[2:]:
        f = line.split('\t')
        if len(f) > 1 and f[1] == 'T':
            out.add(int(f[0]))
    return out


def translate(work, pages=None, model='default', retries=3):
    system = open(_os.path.join(work, 'system.txt')).read()
    _os.makedirs(_os.path.join(work, 'tr'), exist_ok=True)
    files = sorted(_glob.glob(_os.path.join(work, 'prompts', 'p*.txt')))
    want = {int(_os.path.basename(f)[1:4]) for f in files}
    if pages:
        want &= set(pages)
    else:
        want = {n for n in want if not _os.path.exists(_os.path.join(work, 'tr', f'p{n:03d}.json'))}
    pending = sorted(want)
    for _ in range(retries):
        if not pending:
            break
        texts = {n: open(_os.path.join(work, 'prompts', f'p{n:03d}.txt')).read() for n in pending}
        hs = {n: completion(texts[n], model=model, system=system, schema=_SCHEMA) for n in pending}
        bad = []
        for n, h in hs.items():
            try:
                r = h.wait()
                r = r if isinstance(r, dict) else _json.loads(r)
                got = [i for u in r['units'] for i in u['ids']]
                if not _tids(texts[n]) <= set(got) or len(got) != len(set(got)):
                    raise ValueError('ids mismatch')
                with open(_os.path.join(work, 'tr', f'p{n:03d}.json'), 'w') as f:
                    _json.dump(r, f, ensure_ascii=False)
            except Exception:
                bad.append(n)
        pending = bad
    print(f'translated {len(want) - len(pending)}/{len(want)}; failed: {pending or "-"}')
