#!/usr/bin/env python3
"""kpt (knitting-pattern-translate) — translate a Japanese or Korean pattern PDF to English in place.

Pipeline (all state lives in a work dir):
  init -> ocr -> prompts -> [translate in agent eval] -> qa-sheet/correct -> render -> resid/apply -> render
`ocr` reads the PDF text layer when a page has one (exact text, size, color; removed by redaction at render time)
and falls back to OCR for pages whose text is outlines or scans (removed by cover/inpaint).
Outputs are deliberately terse: the caller is an agent paying per token.
"""
import argparse, glob, html, json, os, re, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
DPI = 300
K = DPI / 72

# Per source language: `src` finds text to translate (incl. ー・), `letters` guards rendered English,
# `sym` = diagram symbols OCR misreads as letters, `keep1` = real one-character labels, `ocr` = RapidOCR rec model.
LANGS = {
    'ja': {'name': 'Japanese',
           'src': r'[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f々]',
           'letters': r'[\u3041-\u3096\u30a1-\u30fa\u3400-\u9fff\uff66-\uff9f々]',
           'sym': set('入人小下上亚开卫刃个中米平三大ねタと良广穷商园文实安辰凶品土干工十丁了拼戊卜廾艹丫丬丨乂兀一二日ダニー区'),
           'keep1': set('段目脇袖衿裾'), 'ocr': None},
    'ko': {'name': 'Korean',
           'src': r'[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]',
           'letters': r'[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]',
           'sym': set(), 'keep1': set('코단겉안앞뒤목'), 'ocr': ('korean', 'PP-OCRv5', 'mobile')},
}


# ---------------------------------------------------------------- helpers
def cfg(w):
    return json.load(open(os.path.join(w, 'config.json')))


def lang(w_or_cfg):
    c = cfg(w_or_cfg) if isinstance(w_or_cfg, str) else w_or_cfg
    L = LANGS[c.get('lang', 'ja')]
    return L, re.compile(L['src']), re.compile(L['letters'])


def jload(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def jsave(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def pages_arg(s, total):
    if not s:
        return list(range(1, total + 1))
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def cover(a, b):
    """fraction of box a covered by box b"""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    ar = (a[2] - a[0]) * (a[3] - a[1])
    return max(0, x1 - x0) * max(0, y1 - y0) / ar if ar > 0 else 1


def untag(s):
    """drop <sN>…</sN> style tags (styled runs, see _text_items)"""
    return re.sub(r'</?s\d+>', '', s)


def _style_desc(st):
    """what an <sN> tag means, e.g. 'bold, underlined, colour #0080ea, 10pt'"""
    return ', '.join([k for k in ('bold', 'italic') if st[k]] + (['underlined'] if st.get('underline') else []) +
                     ['colour #%02x%02x%02x' % tuple(st['color']), f"{st['size']:g}pt"])


def classify(x, L, src_re):
    """sets x['tl'] (translate this line) and x['v'] (vertical)"""
    jc = src_re.findall(x['text'])
    src = x.get('src', 'v1')
    ok = bool(jc) and not all(c in L['sym'] for c in jc)
    if src not in ('v1', 'pdf'):  # extra OCR passes produce more junk: stricter
        ok = ok and (len(jc) >= 2 or set(jc) <= L['keep1']) and x['score'] >= 0.6
    w, h = x['bbox'][2] - x['bbox'][0], x['bbox'][3] - x['bbox'][1]
    x['tl'] = ok
    x['v'] = ok and (x.get('rot', 0) in (90, 270) if src == 'pdf' else (h > 1.5 * w and len(x['text']) > 1))


def ocr_engine(L, **extra):
    from rapidocr import RapidOCR
    p = {"Global.log_level": "error", "EngineConfig.onnxruntime.intra_op_num_threads": 4, **extra}
    if L['ocr']:
        from rapidocr.utils.typings import LangRec, OCRVersion, ModelType
        p["Rec.lang_type"], p["Rec.ocr_version"], p["Rec.model_type"] = (
            LangRec(L['ocr'][0]), OCRVersion(L['ocr'][1]), ModelType(L['ocr'][2]))
    return RapidOCR(params=p)


# ---------------------------------------------------------------- init / ocr
def cmd_init(a):
    import pymupdf
    os.makedirs(a.work, exist_ok=True)
    doc = pymupdf.open(a.pdf)
    lg = a.lang
    if lg == 'auto':  # count script use in the text layer; scans/outlines have none -> Japanese
        txt = ''.join(p.get_text() for p in doc)
        ko, ja = len(re.findall(LANGS['ko']['src'], txt)), len(re.findall(r'[\u3040-\u30ff]', txt))
        lg = 'ko' if ko > ja else 'ja'
    jsave(os.path.join(a.work, 'config.json'), {'src': os.path.abspath(a.pdf), 'pages': doc.page_count, 'lang': lg})
    print(f'work={a.work} pages={doc.page_count} lang={lg}')


def _span_style(s):
    return {'bold': bool(s['flags'] & 16) or bool(re.search(r'bold|black|heavy', s['font'], re.I)),
            'italic': bool(s['flags'] & 2) or bool(re.search(r'italic|oblique', s['font'], re.I)),
            'color': [(s['color'] >> 16) & 255, (s['color'] >> 8) & 255, s['color'] & 255],
            'size': round(s['size'], 2)}


def _style_differs(a, b):
    return a['bold'] != b['bold'] or a['italic'] != b['italic'] or a['color'] != b['color'] or \
        not 0.89 < a['size'] / max(b['size'], 0.1) < 1.12 or a.get('underline') != b.get('underline')


def _rules(drawings):
    """thin horizontal strokes (underlines, dotted rules) joined into runs: [{'rule': [x0, y, x1], 'pen': {…}}]"""
    segs = sorted((g for g in drawings if g['rect'].height <= 1.5 and g['rect'].width >= 0.3 and g.get('color')
                   and 0 < len(g['items']) <= 4 and all(i[0] == 'l' for i in g['items'])),
                  key=lambda g: (round(g['rect'].y0, 1), g['rect'].x0))
    groups = []
    for g in segs:
        r = g['rect']
        y = (r.y0 + r.y1) / 2
        if groups and abs(groups[-1]['y'] - y) < 0.3 and r.x0 - groups[-1]['x1'] <= 2:
            groups[-1]['x1'] = max(groups[-1]['x1'], r.x1)
            groups[-1]['segs'].append(g)
        else:
            groups.append({'y': y, 'x0': r.x0, 'x1': r.x1, 'segs': [g]})
    out = []
    for G in groups:
        if G['x1'] - G['x0'] < 3:
            continue
        m = max(G['segs'], key=lambda g: g['rect'].width)
        out.append({'rule': [round(G['x0'], 2), round(G['y'], 2), round(G['x1'], 2)],
                    'pen': {'width': round(m.get('width') or 1, 2), 'dashes': m.get('dashes') or '',
                            'color': [round(v * 255) for v in m['color']]}})
    return out


def _text_items(page, src_re):
    """Text-layer lines, split at wide gaps (another element sits there), with exact style. Runs styled unlike
    the rest of their line (bold, italic, colour, size, underline) are tagged <sN>…</sN> in `styled`; N indexes
    the page-wide style table in each item's `styles`. An underline style keeps its source rule, so graphics
    attached to it (leader lines to a note) can follow the English words."""
    import pymupdf
    items, sids = [], {}
    drawings = page.get_drawings()
    # small closed shapes (boxes, pills, buttons) that a run of text may sit inside: inline graphics
    boxes = [g['rect'] for g in drawings
             if 4 < g['rect'].width < 140 and 4 < g['rect'].height < 30 and (g.get('fill') or g.get('color'))]
    rules = _rules(drawings)

    def cstyle(ch, s):
        st = _span_style(s)
        cx, bot = (ch['bbox'][0] + ch['bbox'][2]) / 2, ch['bbox'][3]
        for r in rules:
            x0, y, x1 = r['rule']
            if bot - 2 <= y <= bot + 3 and x0 - 0.5 <= cx <= x1 + 0.5:
                st['underline'] = dict(r['pen'], rule=r['rule'])
                break
        return st
    for b in page.get_text('rawdict', flags=pymupdf.TEXTFLAGS_TEXT)['blocks']:
        for l in b.get('lines', []):
            dx, dy = l['dir']
            horiz = abs(dy) < 0.01
            segs, cur, last = [], [], None
            for s in l['spans']:
                for ch in s['chars']:
                    if ch['c'].strip():
                        if horiz and last and ch['bbox'][0] - last['bbox'][2] > 1.5 * s['size']:
                            segs.append(cur); cur = []
                        last = ch
                    cur.append((ch, s))
            segs.append(cur)
            for seg in segs:
                ink = [(ch, s) for ch, s in seg if ch['c'].strip()]
                if not ink:
                    continue
                text = ''.join(ch['c'] for ch, _ in seg).strip()
                bb = [min(ch['bbox'][0] for ch, _ in ink), min(ch['bbox'][1] for ch, _ in ink),
                      max(ch['bbox'][2] for ch, _ in ink), max(ch['bbox'][3] for ch, _ in ink)]
                # line style = the most common character style; runs that differ are tagged
                # (letters/digits only: coloured symbols are handled by symcolors)
                cst = [cstyle(ch, s) if ch['c'].strip() else None for ch, s in seg]
                keys = [json.dumps(st, sort_keys=True) for st in cst if st]
                base = json.loads(max(set(keys), key=keys.count))
                main = max((s for ch, s in ink), key=lambda s: sum(1 for c in s['chars'] if c['c'].strip()))
                col = main['color']
                it = {'bbox': [round(v, 2) for v in bb], 'text': text, 'score': 1.0, 'src': 'pdf',
                      'size': base['size'], 'color': base['color'], 'bold': base['bold']}
                if base['italic']:
                    it['italic'] = True
                if base.get('underline'):
                    it['underline'] = base['underline']
                runs = []
                for (ch, s), st in zip(seg, cst):
                    st = st or (runs[-1][1] if runs else base)
                    st = st if _style_differs(st, base) else base
                    if runs and not _style_differs(runs[-1][1], st):
                        runs[-1][0] += ch['c']
                    else:
                        runs.append([ch['c'], st])
                if any(st is not base and re.search(r'\w', t) for t, st in runs):
                    out, used = '', {}
                    for t, st in runs:
                        if st is base or not re.search(r'\w', t):
                            out += t
                            continue
                        key = json.dumps(st, sort_keys=True)
                        n_ = sids.setdefault(key, len(sids) + 1)
                        used[str(n_)] = st
                        core = t.strip()
                        lead_ = t[:len(t) - len(t.lstrip())]
                        out += f'{lead_}<s{n_}>{core}</s{n_}>' + t[len(lead_) + len(core):]
                    it['styled'], it['styles'] = out.strip(), used
                # symbols printed in their own colour (a coloured ● that keys a diagram) keep it in the English
                symc = {}
                for ch, s in ink:
                    if s['color'] != col and unicodedata.category(ch['c']) == 'So':
                        sc = s['color']
                        symc[ch['c']] = [(sc >> 16) & 255, (sc >> 8) & 255, sc & 255]
                if symc:
                    it['symcolors'] = symc
                # baseline start: paragraphs are re-flowed from it
                it['origin'] = [round(v, 2) for v in ink[0][0]['origin']]
                if horiz:
                    it['rot'] = 0 if dx > 0 else 180
                    br = pymupdf.Rect(bb)
                    enc = [g for g in boxes if g.x0 <= br.x0 + 0.5 and g.y0 <= br.y0 + 0.5 and g.x1 >= br.x1 - 0.5
                           and g.y1 >= br.y1 - 0.5 and g.height <= br.height * 2.8
                           and g.width <= br.width + 4 * main['size']]
                    if enc:  # text inside a small shape: an inline graphic that may sit within a sentence
                        g = min(enc, key=lambda r: r.width * r.height)
                        it['inline'] = [round(v, 2) for v in g]
                        # a rule right under the shape (a leader's start) belongs to it and moves with it
                        marks = [r for r in rules if g.y1 - 0.5 <= r['rule'][1] <= g.y1 + 4
                                 and r['rule'][0] >= g.x0 - 1.5 and r['rule'][2] <= g.x1 + 1.5]
                        if marks:
                            it['inline_marks'] = marks
                elif abs(dx) < 0.01:
                    it['rot'] = 270 if dy > 0 else 90
                else:  # arbitrary angle: keep geometry for a rotated redraw
                    it['rot'] = -1
                    it['dir'] = [round(dx, 4), round(dy, 4)]
                    it['chars'] = [[round(v, 2) for v in ch['bbox']] for ch, _ in ink]
                items.append(it)
    return items


def _compress(crop):
    """drop wide blank columns: letter-spaced headings defeat the recognizer otherwise"""
    import numpy as np, cv2
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    bg = np.median(np.concatenate([g[0], g[-1]]))
    ink = (np.abs(g.astype(int) - bg) > 40).any(0)
    keep, gap = [], 0
    for i, v in enumerate(ink):
        gap = 0 if v else gap + 1
        if gap <= 6:
            keep.append(i)
    return crop[:, keep] if keep else crop


def _ocr_worker(args):
    w, pages, fast, mode = args
    import numpy as np, cv2, pymupdf
    c = cfg(w)
    L, src_re, _ = lang(c)
    doc = pymupdf.open(c['src'])
    eng = rec = None

    def run(img, ox, oy, k, src, items, edge=None):
        r = eng(img)
        if getattr(r, 'boxes', None) is None:
            return
        for b, t, s in zip(r.boxes, r.txts, r.scores):
            xs, ys = [p[0] for p in b], [p[1] for p in b]
            if edge:  # tile: drop boxes touching an inner tile edge (truncated)
                T, W_, H_ = edge
                if (min(xs) < 3 and ox) or (min(ys) < 3 and oy) or (max(xs) > T - 3 and ox + T < W_) or (max(ys) > T - 3 and oy + T < H_):
                    continue
            bb = [float(v) for v in ((min(xs) + ox) * k, (min(ys) + oy) * k, (max(xs) + ox) * k, (max(ys) + oy) * k)]
            if src != 'v1' and any(cover(bb, y['bbox']) > 0.3 or cover(y['bbox'], bb) > 0.3 for y in items):
                continue
            items.append({'bbox': bb, 'text': t, 'score': float(s), 'src': src})

    for n in pages:
        out = os.path.join(w, 'ocr', f'p{n:03d}.json')
        if os.path.exists(out):
            continue
        page = doc[n - 1]
        pix = page.get_pixmap(dpi=300)
        img300 = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy()
        cv2.imwrite(os.path.join(w, 'r300', f'p{n:03d}.png'), cv2.cvtColor(img300, cv2.COLOR_RGB2BGR))
        page.get_pixmap(dpi=100).save(os.path.join(w, 'orig', f'p{n:03d}.png'))
        use_text = mode == 'text' or (mode == 'auto' and src_re.search(page.get_text()))
        if use_text:
            items = _text_items(page, src_re)
        else:
            if eng is None:
                eng, rec = ocr_engine(L), ocr_engine(L, **{"Global.use_det": False, "Global.use_cls": False})
            imgs = {300: img300}
            for dpi in (200, 400):
                p_ = page.get_pixmap(dpi=dpi)
                imgs[dpi] = np.frombuffer(p_.samples, np.uint8).reshape(p_.h, p_.w, p_.n)[:, :, :3].copy()
            items = []
            run(imgs[300], 0, 0, 72 / 300, 'v1', items)
            for dpi in (200, 400):
                run(imgs[dpi], 0, 0, 72 / dpi, f'd{dpi}', items)
            if not fast:  # tiles find small labels lost in dense diagrams
                for dpi, T, O in ((300, 1100, 250), (400, 1300, 300)):
                    img = imgs[dpi]
                    H_, W_ = img.shape[:2]
                    for y in range(0, H_, T - O):
                        for x in range(0, W_, T - O):
                            run(img[y:y + T, x:x + T], x, y, 72 / dpi, f't{dpi}', items, (T, W_, H_))
            for x in items:  # re-read weak horizontal lines with letter spacing removed
                w_, h_ = x['bbox'][2] - x['bbox'][0], x['bbox'][3] - x['bbox'][1]
                if x['score'] >= 0.9 or h_ > 1.5 * w_:
                    continue
                x0, y0, x1, y1 = [int(v * K) for v in x['bbox']]
                crop = img300[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
                if crop.size == 0 or crop.shape[1] < 8:
                    continue
                r = rec(_compress(crop))
                if r.txts and r.scores[0] > x['score'] + 0.05:
                    x['ocr0'], x['text'], x['score'] = x['text'], r.txts[0], float(r.scores[0])
                    x['src'] += '+rr'
        for x in items:
            classify(x, L, src_re)
        jsave(out, items)
        print(n, 'text' if use_text else 'ocr', len(items), sum(x['tl'] for x in items), flush=True)


def cmd_ocr(a):
    from concurrent.futures import ProcessPoolExecutor
    c = cfg(a.work)
    for d in ('ocr', 'r300', 'orig'):
        os.makedirs(os.path.join(a.work, d), exist_ok=True)
    pages = pages_arg(a.pages, c['pages'])
    nw = max(1, min(a.jobs, len(pages)))
    with ProcessPoolExecutor(nw) as ex:
        list(ex.map(_ocr_worker, [(a.work, pages[i::nw], a.fast, a.mode) for i in range(nw)]))
    tot = sum(sum(x['tl'] for x in json.load(open(f))) for f in glob.glob(os.path.join(a.work, 'ocr', '*.json')))
    print(f"ocr done: {len(pages)} pages, {tot} {LANGS[c['lang']]['name']} lines")


# ---------------------------------------------------------------- prompts
def cmd_prompts(a):
    c = cfg(a.work)
    if c.get('glossary') != a.glossary:
        c['glossary'] = a.glossary
        jsave(os.path.join(a.work, 'config.json'), c)
    L = LANGS[c['lang']]
    pd = os.path.join(a.work, 'prompts')
    os.makedirs(pd, exist_ok=True)
    gl = os.path.join(SKILL, 'prompts', f"glossary-{a.glossary}-{c['lang']}.md")
    if not os.path.exists(gl):
        gl = os.path.join(SKILL, 'prompts', f'glossary-{a.glossary}.md')
    parts = [open(os.path.join(SKILL, 'prompts', 'system.md')).read().replace('{LANG}', L['name']),
             open(gl).read()]
    book = os.path.join(a.work, 'book.md')
    if os.path.exists(book):
        parts.append('Book-specific names (use exactly):\n' + open(book).read())
    open(os.path.join(a.work, 'system.txt'), 'w').write('\n\n'.join(parts))
    made = chars = 0
    for n in pages_arg(a.pages, c['pages']):
        items = jload(os.path.join(a.work, 'ocr', f'p{n:03d}.json'), [])
        t = [x for x in items if x['tl']]
        if not t:
            continue
        # context lines only near translatable ones (diagrams can hold hundreds of bare numbers)
        near = lambda x: any(x['bbox'][0] < y['bbox'][2] + 25 and y['bbox'][0] - 25 < x['bbox'][2] and
                             x['bbox'][1] < y['bbox'][3] + 12 and y['bbox'][1] - 12 < x['bbox'][3] for y in t)
        rows, styles = [], {}
        for i, x in enumerate(items):
            if x['tl'] or (x.get('src') in ('v1', 'pdf') and near(x)):
                b = [round(v) for v in x['bbox']]
                o = 'V' if x['v'] else ('G' if x.get('inline') else 'H')  # G = text inside an inline graphic
                txt = x.get('styled', x['text']) if x['tl'] else x['text']
                if x['tl']:
                    styles.update(x.get('styles', {}))
                rows.append(f"{i}\t{'T' if x['tl'] else 'C'}\t{o}\t{b[0]},{b[1]},{b[2]},{b[3]}\t{txt}")
        s = f"Page {n}/{c['pages']}\nid\tT/C\tor\tbbox\ttext\n" + '\n'.join(rows)
        if styles:  # what each <sN> tag means, so the translator can tag the matching English
            s += '\nStyles: ' + '; '.join(f's{k} = {_style_desc(v)}' for k, v in sorted(styles.items(), key=lambda kv: int(kv[0])))
        open(os.path.join(pd, f'p{n:03d}.txt'), 'w').write(s)
        made += 1
        chars += len(s)
    print(f'prompts: {made} pages, {chars} chars -> {pd}')


# ---------------------------------------------------------------- QA sheets / corrections
def cmd_qa_sheet(a):
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont
    font_path = os.path.join(a.work, 'cjk.otf')
    if not os.path.exists(font_path):
        open(font_path, 'wb').write(pymupdf.Font('cjk').buffer)
    font = ImageFont.truetype(font_path, 22)
    qd = os.path.join(a.work, 'qa')
    os.makedirs(qd, exist_ok=True)
    for f in glob.glob(os.path.join(qd, 's*.png')):
        os.remove(f)
    tiles = []
    c = cfg(a.work)
    for n in pages_arg(a.pages, c['pages']):
        items = jload(os.path.join(a.work, 'ocr', f'p{n:03d}.json'), [])
        img = None
        for i, x in enumerate(items):
            if not x['tl'] or x.get('checked') or x.get('src') == 'pdf':  # text-layer text is exact
                continue
            # risky: vertical labels with digits (tategaki numbers misread) or low confidence
            if not ((x['v'] and re.search(r'\d', x['text'])) or x['score'] < 0.9):
                continue
            if img is None:
                img = Image.open(os.path.join(a.work, 'r300', f'p{n:03d}.png')).convert('RGB')
            b = x['bbox']
            cr = img.crop((int((b[0] - 3) * K), int((b[1] - 3) * K), int((b[2] + 3) * K), int((b[3] + 3) * K)))
            sc = min(1.0, 170 / cr.height, 520 / cr.width)
            if sc < 1:
                cr = cr.resize((max(1, int(cr.width * sc)), max(1, int(cr.height * sc))))
            tiles.append((f"{n}:{i} 「{x['text']}」", cr))
    W, H = 560, 2000
    cols, col, h = [], [], 0
    for t in tiles:
        if h + t[1].height + 32 > H and col:
            cols.append(col); col, h = [], 0
        col.append(t); h += t[1].height + 32
    if col:
        cols.append(col)
    for si in range(0, len(cols), 3):
        grp = cols[si:si + 3]
        sheet = Image.new('RGB', (W * len(grp), H), 'white')
        d = ImageDraw.Draw(sheet)
        for ci, cc in enumerate(grp):
            y = 0
            for label, cr in cc:
                d.text((ci * W + 4, y), label[:34], fill=(200, 0, 0), font=font)
                sheet.paste(cr, (ci * W + 4, y + 26))
                y += cr.height + 32
            d.line([(ci * W + W - 2, 0), (ci * W + W - 2, H)], fill=(0, 0, 255), width=2)
        sheet.save(os.path.join(qd, f's{si // 3:02d}.png'))
    print(f'{len(tiles)} items on {(len(cols) + 2) // 3} sheets: {qd}/s*.png')


def cmd_correct(a):
    """P:ID=TEXT fixes OCR text; P:ID=- marks a line as not text. Affected translations are invalidated."""
    touched = set()
    by_page = {}
    for spec in a.specs:
        pid, _, text = spec.partition('=')
        p, i = map(int, pid.split(':'))
        by_page.setdefault(p, []).append((i, text))
    for p, lst in by_page.items():
        path = os.path.join(a.work, 'ocr', f'p{p:03d}.json')
        items = json.load(open(path))
        for i, text in lst:
            x = items[i]
            if text == '-':
                x['tl'] = x['v'] = False
            else:
                x.setdefault('ocr0', x['text'])
                x['text'], x['score'] = text, 1.0
                w_, h_ = x['bbox'][2] - x['bbox'][0], x['bbox'][3] - x['bbox'][1]
                x['tl'], x['v'] = True, h_ > 1.5 * w_ and len(text) > 1
            x['checked'] = True
        jsave(path, items)
        tr = os.path.join(a.work, 'tr', f'p{p:03d}.json')
        if os.path.exists(tr):
            os.remove(tr)
        touched.add(p)
    if touched:  # prompts carry the OCR text: rebuild them for the corrected pages
        cmd_prompts(argparse.Namespace(work=a.work, pages=','.join(map(str, sorted(touched))),
                                       glossary=cfg(a.work).get('glossary', 'knitting')))
    print('retranslate:', ','.join(map(str, sorted(touched))) or '-')


def cmd_checked(a):
    """mark every risky item on the listed pages as reviewed (so qa-sheet skips them next time)"""
    c = cfg(a.work)
    for n in pages_arg(a.pages, c['pages']):
        path = os.path.join(a.work, 'ocr', f'p{n:03d}.json')
        if not os.path.exists(path):
            continue
        items = json.load(open(path))
        for x in items:
            if x['tl']:
                x['checked'] = True
        jsave(path, items)
    print('ok')


# ---------------------------------------------------------------- render
def _render(a, pages, out):
    import numpy as np, cv2, pymupdf
    w = a.work
    c = cfg(w)
    _, _, letters = lang(c)
    PAD = 0.5
    FIX = jload(os.path.join(w, 'fixes.json'), {})
    MANUAL = jload(os.path.join(w, 'manual.json'), {})
    RESTORE = jload(os.path.join(w, 'restore.json'), {})
    LAYOUT = jload(os.path.join(w, 'layout.json'), {})  # {page: {"ids": {"align": "left|center|right"}}}
    pristine = pymupdf.open(c['src'])
    doc = pymupdf.open(c['src'])
    FONTS = {(False, False): pymupdf.Font('helv'), (True, False): pymupdf.Font('hebo'),
             (False, True): pymupdf.Font('heit'), (True, True): pymupdf.Font('hebi')}
    fonts = {False: FONTS[False, False], True: FONTS[True, False]}
    font_of = lambda st: FONTS[bool(st.get('bold')), bool(st.get('italic'))]
    fallback = pymupdf.Font('cjk')  # arrows/symbols missing from Helvetica

    def px(r):
        return [int(max(0, r[0] * K)), int(max(0, r[1] * K)), int(r[2] * K + 0.999), int(r[3] * K + 0.999)]

    def lum(x):
        return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114

    def ring(img, r, wd=3):
        x0, y0, x1, y1 = px(r)
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x0 - wd), max(0, y0 - wd), min(W, x1 + wd), min(H, y1 + wd)
        parts = [img[y0:y0 + wd, x0:x1], img[y1 - wd:y1, x0:x1], img[y0:y1, x0:x0 + wd], img[y0:y1, x1 - wd:x1]]
        return np.concatenate([p.reshape(-1, 3) for p in parts if p.size]).astype(int)

    def ink_fraction(img, r, bg):
        x0, y0, x1, y1 = px(r)
        q = img[y0:y1, x0:x1].reshape(-1, 3).astype(int)
        return float((np.abs(q - bg).max(1) > 40).mean()) if q.size else 1.0

    def enclosure_gaps(img, b, bg, limit=120):
        """blank distance (pt) from a line to the nearest ink left and right along its glyph core;
        None = no ink within `limit` (free-standing text)"""
        x0, y0, x1, y1 = px(b)
        h = y1 - y0
        band = img[y0 + int(h * 0.3):max(y0 + int(h * 0.3) + 1, y1 - int(h * 0.3))].astype(int)
        if not band.size:
            return None, None
        ink = (np.abs(band - bg).max(2) > 40).any(0)
        lim = int(limit * K)
        gl = next((i for i in range(1, lim) if x0 - i >= 0 and ink[x0 - i]), None)
        gr = next((i for i in range(1, lim) if x1 + i < len(ink) and ink[x1 + i]), None)
        return (gl / K if gl else None), (gr / K if gr else None)

    def text_color(img, r, bg):
        x0, y0, x1, y1 = px(r)
        q = img[y0:y1, x0:x1].reshape(-1, 3).astype(int)
        if len(q) < 20:
            return (0, 0, 0)
        L, bgL = lum(q), lum(np.array(bg))
        dark = (bgL - np.percentile(L, 4)) >= (np.percentile(L, 96) - bgL)
        cc = np.median(q[L <= np.percentile(L, 6)] if dark else q[L >= np.percentile(L, 94)], 0)
        if abs(lum(cc) - bgL) < 60:
            return (0, 0, 0) if bgL > 110 else (255, 255, 255)
        return tuple(int(v) for v in cc)

    def seen_color(img, it, bg):
        """text-layer colour as actually rendered: transparency, overlays or colour spaces can make text look
        unlike its declared fill (grey text declared black). The declared colour stays whenever its own pixels
        are present in the glyphs (text inside a small shape has a misleading surround); a QA override wins."""
        if it.get('color_fixed'):
            return tuple(it['color'])
        x0, y0, x1, y1 = px(pymupdf.Rect(it['bbox']))
        q = img[y0:y1, x0:x1].reshape(-1, 3).astype(int)
        ink = q[np.abs(q - np.median(q, 0)).max(1) > 60] if len(q) else q
        if len(ink) < 20 or (np.abs(ink - np.array(it['color'])).max(1) < 50).mean() >= 0.15:
            return tuple(it['color'])
        c = text_color(img, pymupdf.Rect(it['bbox']), bg)
        return c if np.abs(np.array(c) - np.array(it['color'])).max() > 60 else tuple(it['color'])

    def stroke_mask(img, r):
        x0, y0, x1, y1 = px(r)
        H, W = img.shape[:2]
        x1, y1 = min(x1, W), min(y1, H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return (x0, y0, max(x0, x1), max(y0, y1)), np.zeros((max(0, y1 - y0), max(0, x1 - x0)), np.uint8)
        ks = max(15, int(min(x1 - x0, y1 - y0) * 0.7)) | 1
        X0, Y0, X1, Y1 = max(0, x0 - ks), max(0, y0 - ks), min(W, x1 + ks), min(H, y1 + ks)
        L = lum(img[Y0:Y1, X0:X1].astype(np.float32))
        kk = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        m = ((cv2.morphologyEx(L, cv2.MORPH_CLOSE, kk) - L) > 15) | ((L - cv2.morphologyEx(L, cv2.MORPH_OPEN, kk)) > 15)
        m = m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        d = max(5, ks // 10) | 1
        return (x0, y0, x1, y1), cv2.dilate(m.astype(np.uint8) * 255, np.ones((d, d), np.uint8))

    def redact_rects(it):
        """small rects at glyph centres: removes exactly this line's glyphs, never a neighbour's"""
        if it.get('rot') == -1:
            out = []
            for b in it['chars']:
                cx, cy, hw, hh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * 0.2, (b[3] - b[1]) * 0.2
                out.append(pymupdf.Rect(cx - hw, cy - hh, cx + hw, cy + hh))
            return out
        b = pymupdf.Rect(it['bbox'])
        if it.get('rot') in (90, 270):
            m, half = (b.x0 + b.x1) / 2, b.width * 0.15
            return [pymupdf.Rect(m - half, b.y0 + 0.2, m + half, b.y1 - 0.2)]
        m, half = (b.y0 + b.y1) / 2, b.height * 0.15
        return [pymupdf.Rect(b.x0 + 0.2, m - half, b.x1 - 0.2, m + half)]

    def write_rotated(page, it, en, color):
        """text-layer line at an arbitrary angle: redraw along the same baseline"""
        dx, dy = it['dir']
        o = pymupdf.Point(it['origin'])
        # run length of the source along its own direction (`dir` is in page coordinates, y down)
        avail = max(5.0, max(((b[0] + b[2]) / 2 - o.x) * dx + ((b[1] + b[3]) / 2 - o.y) * dy for b in it['chars'])
                    + it['size'] * 0.5)
        fs = it['size']
        f = font_of(it)
        runs, cur, cf = [], '', None
        for ch in en:
            ff = f if f.has_glyph(ord(ch)) else fallback
            if ff is not cf and cur:
                runs.append((cur, cf)); cur = ''
            cur, cf = cur + ch, ff
        if cur:
            runs.append((cur, cf))
        width = sum(ff.text_length(t, fs) for t, ff in runs)
        if width > avail:  # never run past the source's extent (diagrams are tight); QA shortens tiny results
            fs *= avail / width
        tw = pymupdf.TextWriter(page.rect, color=[v / 255 for v in color])
        pos = pymupdf.Point(o)
        for t, ff in runs:
            tw.append(pos, t, font=ff, fontsize=fs)
            pos = tw.last_point
        tw.write_text(page, morph=(o, pymupdf.Matrix(dx, -dy, dy, dx, 0, 0)), overlay=True)

    zapf = pymupdf.Font('zadb')

    def gfont(ch, base):
        """font for one character: Helvetica, then CJK fallback, then ZapfDingbats (➊ ✔)"""
        for f in (base, fallback, zapf):
            if f.has_glyph(ord(ch)):
                return f
        return base

    def width(s, size, base):
        return sum(gfont(ch, base).text_length(ch, size) for ch in s)

    def style_runs(en, base, S):
        """English with <sN>…</sN> tags -> [(text, style)]; untagged text and unknown tags use `base`"""
        out, pos = [], 0
        for m in re.finditer(r'<s(\d+)>(.*?)</s\1>', en, re.S):
            out.append((untag(en[pos:m.start()]), base))
            st = S.get(m.group(1))
            out.append((untag(m.group(2)), dict(base, **st) if st else base))
            pos = m.end()
        out.append((untag(en[pos:]), base))
        return [(t, s) for t, s in out if t]

    def column_align(i0, items):
        """alignment of a label stacked in a column with other text-layer lines (table rows, label lists):
        shared left edges -> left, shared right edges -> right, shared centres of different widths -> center"""
        b = items[i0]['bbox']
        h = b[3] - b[1]
        votes = {'left': 0, 'right': 0, 'center': 0}
        for j, x in enumerate(items):
            o = x['bbox']
            if j == i0 or x.get('src') != 'pdf' or x.get('rot') != 0 or min(b[3], o[3]) > max(b[1], o[1]) \
                    or abs(o[1] - b[1]) > 8 * h:
                continue
            l, r = abs(o[0] - b[0]) < 0.8, abs(o[2] - b[2]) < 0.8
            if l and r:
                continue  # same width says nothing
            votes['left'] += l
            votes['right'] += r
            votes['center'] += abs(o[0] + o[2] - b[0] - b[2]) < 1.6
        top = max(votes.values())
        win = [k for k, v in votes.items() if v == top]
        return win[0] if top and len(win) == 1 else None

    def plan_flow(items, ids, en, img, room):
        """Text-layer paragraph (several lines, inline [[graphics]], or one long line that no longer fits but has
        blank lines below) -> layout plan, or None to use the per-box renderer. The paragraph is re-flowed as one
        block at one size; inline graphics move with the text."""
        its = [items[i] for i in ids]
        if any(it.get('src') != 'pdf' or it.get('rot') != 0 or 'origin' not in it for it in its):
            return None
        txt = [i for i in ids if not items[i].get('inline')]
        labels = re.findall(r'\[\[(.+?)\]\]', untag(en))
        if not txt:
            return None
        fs = float(np.median([items[i]['size'] for i in txt]))
        order = sorted(ids, key=lambda i: (items[i]['origin'][1], items[i]['bbox'][0]))
        groups = []
        for i in order:
            if groups and abs(items[i]['origin'][1] - groups[-1][0]) < 0.5 * fs:
                groups[-1][1].append(i)
            else:
                groups.append([items[i]['origin'][1], [i]])
        gfx = [i for _, g in groups for i in sorted(g, key=lambda i: items[i]['bbox'][0]) if items[i].get('inline')]
        if len(labels) != len(gfx):
            return None
        bases = [g[0] for g in groups]
        diffs = [b - a for a, b in zip(bases, bases[1:])]
        if any(d < 0.8 * fs or d > 3.2 * fs for d in diffs):
            return None  # not one paragraph
        x0s = [min(items[i]['bbox'][0] for i in g) for _, g in groups]
        x1s = [max(items[i]['bbox'][2] for i in g) for _, g in groups]
        cs = [(a + b) / 2 for a, b in zip(x0s, x1s)]
        centered = len(groups) > 1 and max(cs) - min(cs) < 2 and max(x0s) - min(x0s) > 3
        if not centered and max(x0s) - min(x0s) > 4 * fs:
            return None
        main = items[txt[0]]
        bg = np.median(ring(img, pymupdf.Rect(main['bbox'])), 0)
        color = seen_color(img, main, bg) if not any(items[i].get('styles') for i in txt) else tuple(main['color'])
        P = {'fs': fs, 'lead': float(np.median(diffs)) if diffs else 1.6 * fs, 'base0': bases[0],
             'nlines': len(groups), 'xf': x0s[0], 'xr': min(x0s[1:]) if len(x0s) > 1 else x0s[0],
             'xR': max(x1s), 'centered': centered, 'inline': gfx, 'labels': labels,
             'color': color, 'bold': main.get('bold', False),
             'base': {'bold': main.get('bold', False), 'italic': main.get('italic', False),
                      'color': list(color), 'size': main['size'], 'underline': main.get('underline')},
             'styles': {k: v for i in txt for k, v in items[i].get('styles', {}).items()},
             'symc': {k: v for i in txt for k, v in items[i].get('symcolors', {}).items()}}
        # room: blank space below the last original line (no other text, no ink, and the same background — a
        # pale panel ending is an edge the text must not cross), then to the right
        others = [pymupdf.Rect(x['bbox']) for j, x in enumerate(items) if j not in ids]
        xl = min(P['xf'], P['xr'])

        def clear(r):
            if not room.contains(r) or any(r.intersects(o) for o in others) or ink_fraction(img, r, bg) > 0.002:
                return False
            x0, y0, x1, y1 = px(r)
            q = img[y0:y1, x0:x1].reshape(-1, 3).astype(int)
            return not len(q) or (np.abs(q - bg).max(1) < 14).mean() >= 0.98
        # yb = lowest usable baseline; scan the strip below the paragraph's lowest box (graphics reach below text)
        top = max([bases[-1] + 0.3 * fs] + [items[i]['bbox'][3] + 0.5 for i in ids] +
                  [items[i]['inline'][3] + 1 for i in gfx])
        y = top
        while y < bases[-1] + 8 * P['lead'] and clear(pymupdf.Rect(xl, y, P['xR'], y + 1)):
            y += 1
        # a new line keeps at least the paragraph's own line gap to the next element (and 1pt of air)
        yb = max(bases[-1], y - 0.3 * fs - max(1.0, P['lead'] - fs))
        xe = P['xR']
        while xe + 2 <= room.x1 and clear(pymupdf.Rect(xe, bases[0] - 0.95 * fs, xe + 2, yb + 0.3 * fs)):
            xe += 2
        P['yb'], P['xRext'] = yb, xe
        if len(groups) == 1 and not gfx:
            # one line, no inline graphic: only worth re-flowing when the English overflows and can wrap below
            plain = width(untag(en), fs, fonts[P['bold']])
            if plain <= P['xR'] - P['xf'] or yb - bases[0] < 1.2 * fs or len(main['text']) < 12:
                return None
        return P

    def copy_paths(page, paths, T):
        """redraw source vector paths through the point map T (move / stretch a graphic)"""
        for g in paths:
            sh = page.new_shape()
            for op in g['items']:
                if op[0] == 'l':
                    sh.draw_line(T(op[1]), T(op[2]))
                elif op[0] == 'c':
                    sh.draw_bezier(T(op[1]), T(op[2]), T(op[3]), T(op[4]))
                elif op[0] == 're':
                    sh.draw_rect(pymupdf.Rect(T(op[1].tl), T(op[1].br)))
                elif op[0] == 'qu':
                    q = op[1]
                    sh.draw_quad(pymupdf.Quad(T(q.ul), T(q.ur), T(q.ll), T(q.lr)))
            lc = g.get('lineCap') or 0
            sh.finish(color=g.get('color'), fill=g.get('fill'), width=g.get('width') or 1,
                      closePath=bool(g.get('closePath')), even_odd=bool(g.get('even_odd')),
                      fill_opacity=g.get('fill_opacity') or 1, stroke_opacity=g.get('stroke_opacity') or 1,
                      lineCap=int(max(lc) if isinstance(lc, (tuple, list)) else lc),
                      lineJoin=int(g.get('lineJoin') or 0), dashes=g.get('dashes') or None)
            sh.commit(overlay=True)

    def flow_layout(items, ids, en, P):
        """place a re-flowed paragraph: P['placed'] = [(kind, text, meta, x, width, baseline)], P['fsz'],
        P['lines'] (line boxes), P['anchors'] = where underlined runs and inline graphics ended up"""
        base_st = P['base']
        fs0 = P['fs']
        # tokens: words (with "space before" flag) and inline graphics
        toks, pend, bi = [], False, iter(zip(P['inline'], P['labels']))
        for txt_, st in style_runs(en, base_st, P['styles']):
            for part in re.split(r'(\[\[.+?\]\])', txt_):
                if part.startswith('[['):
                    i, lab = next(bi)
                    b = items[i]
                    br = pymupdf.Rect(b['inline'])
                    bf = fonts[b.get('bold', False)]
                    pad = max(1.0, (br.width - (b['bbox'][2] - b['bbox'][0])) / 2)
                    bw = max(br.width, width(lab, b['size'], bf) + 2 * pad)
                    toks.append(('G', lab, (b, br, bw), pend)); pend = False
                    continue
                for wd in re.findall(r'\s+|\S+', part):
                    if wd.isspace():
                        pend = True
                    else:
                        toks.append(('W', wd, st, pend)); pend = False
        F = font_of(base_st)
        rel = lambda st: st['size'] / max(base_st['size'], 0.1)  # styled runs keep their size relative to the line
        span = P['yb'] - P['base0']  # baseline-to-baseline height available for lines 2..n

        def layout(fsz, xR):
            lines, x = [[]], P['xf']
            for kind, s, meta, sp in toks:
                w_ = meta[2] if kind == 'G' else width(s, fsz * rel(meta), font_of(meta))
                spw = width(' ', fsz, F) if sp and lines[-1] else 0
                if lines[-1] and x + spw + w_ > xR + 0.5:
                    lines.append([]); x, spw = P['xr'], 0
                lines[-1].append((kind, s, meta, x + spw, w_))
                x += spw + w_
            return lines

        # keep the size: original width with blank lines below, then tighter line spacing, then blank space to
        # the right; shrink only after that
        tries = [(sc, xR, lf) for sc in (1, 0.96, 0.92) for xR in (P['xR'], P['xRext']) for lf in (1, 0.9, 0.8)] + \
                [(sc, P['xRext'], lf) for sc in (0.88, 0.84, 0.8, 0.76, 0.72, 0.68) for lf in (1, 0.8)]
        for sc, xR, lf in tries:
            lead = max(P['lead'] * lf, fs0 * sc * 1.25)
            fsz, lines = fs0 * sc, layout(fs0 * sc, xR)
            if len(lines) == 1 or (len(lines) - 1) * lead <= span + 0.5:
                break
        placed, boxes_, anchors = [], [], {}
        for k, line in enumerate(lines):
            base = P['base0'] + k * lead
            shift = 0
            if P['centered'] and line:
                cx = (min(P['xf'], P['xr']) + P['xR']) / 2
                shift = cx - (line[0][3] + line[-1][3] + line[-1][4]) / 2
            for kind, s, meta, x, w_ in line:
                placed.append((kind, s, meta, x + shift, w_, base))
                if kind == 'G':
                    b, br, bw = meta
                    anchors[('G', tuple(b['inline']))] = {'x': x + shift, 'x0': br.x0, 'sx': bw / br.width,
                                                          'dy': base - b['origin'][1]}
                elif meta.get('underline'):
                    ul = anchors.setdefault(('U', tuple(meta['underline']['rule'])), {'pen': meta['underline'],
                                                                                         'spans': []})
                    sp_ = ul['spans']
                    if sp_ and sp_[-1][2] == base:
                        sp_[-1][1] = x + shift + w_
                    else:
                        sp_.append([x + shift, x + shift + w_, base])
            if line:
                boxes_.append(pymupdf.Rect(line[0][3] + shift, base - 0.8 * fsz,
                                           line[-1][3] + line[-1][4] + shift, base + 0.22 * fsz))
        # underline depth below the baseline, from the source line the rule sat under
        for key, a in anchors.items():
            if key[0] == 'U':
                ry = key[1][1]
                org = max((items[i]['origin'][1] for i in ids if items[i]['origin'][1] <= ry + 0.5), default=ry - 2)
                a['depth'] = (ry - org) * fsz / fs0
        P.update(placed=placed, fsz=fsz, lines=boxes_, anchors=anchors)

    def draw_flow(page, P, gfx_paths):
        fsz = P['fsz']
        rel = lambda st: st['size'] / max(P['base']['size'], 0.1)
        writers = {}

        def put(x, y, s, size, base, color):
            for ch in s:
                f = gfont(ch, base)
                tw = writers.setdefault(tuple(color), pymupdf.TextWriter(page.rect, color=[v / 255 for v in color]))
                tw.append(pymupdf.Point(x, y), ch, font=f, fontsize=size)
                x += f.text_length(ch, size)

        for kind, s, meta, x, w_, base in P['placed']:
            if kind == 'W':
                Ft, szt = font_of(meta), fsz * rel(meta)
                for ch in s:  # per-character colour (coloured ● markers)
                    put(x, base, ch, szt, Ft, P['symc'].get(ch, meta['color']))
                    x += gfont(ch, Ft).text_length(ch, szt)
                continue
            b, br, bw = meta
            dy, sx = base - b['origin'][1], bw / br.width
            T = lambda p: pymupdf.Point(x + (p.x - br.x0) * sx, p.y + dy)
            area = pymupdf.Rect(br)
            for m in b.get('inline_marks', []):  # rules under the shape travel with it
                area |= pymupdf.Rect(m['rule'][0], m['rule'][1] - 1, m['rule'][2], m['rule'][1] + 1)
            copy_paths(page, gfx_paths(area), T)
            bf = fonts[b.get('bold', False)]
            put(x + (bw - width(s, b['size'], bf)) / 2, base, s, b['size'], bf, b['color'])
        for key, a in P['anchors'].items():  # underlines under the English words that carry them
            if key[0] != 'U':
                continue
            pen = a['pen']
            for x0, x1, base in a['spans']:
                y = base + a['depth']
                page.draw_line((x0, y), (x1, y), color=[v / 255 for v in pen['color']], width=pen['width'],
                               dashes=pen['dashes'] or None, overlay=True)
        for tw in writers.values():
            tw.write_text(page, overlay=True)


    skipped = []
    for n in pages:
        items = jload(os.path.join(w, 'ocr', f'p{n:03d}.json'), [])
        units = jload(os.path.join(w, 'tr', f'p{n:03d}.json'), {'units': []})['units']
        for m in MANUAL.get(str(n), []):
            items.append({'bbox': m['bbox'], 'text': m.get('text', ''), 'tl': True, 'v': m.get('v', False),
                          'solid': m.get('solid', False)})
            units.append({'ids': [len(items) - 1], 'en': m['en']})
        if not units and not RESTORE.get(str(n)):
            continue
        fx = FIX.get(str(n), {})
        page = doc[n - 1]
        img = cv2.cvtColor(cv2.imread(os.path.join(w, 'r300', f'p{n:03d}.png')), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        prect = page.rect
        work = []
        for u in units:
            en = fx.get(','.join(map(str, u['ids'])), u['en'])
            if en is None or not u['ids']:
                continue
            sty = LAYOUT.get(str(n), {}).get(','.join(map(str, u['ids'])), {}).get('style')
            for i in u['ids'] if sty else []:  # QA style override (weight, slant, colour) for the whole unit
                items[i].update(sty)
                items[i]['color_fixed'] = 'color' in sty
            if letters.search(re.sub(r'#\S+', '', en)):  # untranslated (hashtags may stay in the source script)
                skipped.append(f"{n}:{','.join(map(str, u['ids']))}")
                continue
            en = en.replace('・', ', ')
            boxes = [pymupdf.Rect(items[i]['bbox']) for i in u['ids']]
            t = items[u['ids'][0]]['text']
            pdf = all(items[i].get('src') == 'pdf' for i in u['ids'])
            if len(boxes) == 1 and not pdf:
                b = boxes[0]
                run = re.search(r'[\u3041-\u30ff\u3400-\u9fff々]+$', t)
                pre = t[:run.start()] if run else ''
                legend = run and re.search(r'[=＝]', pre) and len(pre) <= 6 and not letters.search(pre) and re.search(r'[=＝]', en)
                eng_part = re.sub(r'^[^A-Za-z0-9(]+', '', re.split(r'[=＝]', en)[-1]).strip()
                if legend and pre.strip() in ('=', '＝') and run[0] not in ('表目', '裏目'):
                    # "swatch = name" legend: the swatch sits left of the box -> redraw "= English"; keep a swatch edge
                    en = '= ' + eng_part
                    X0, Y0, X1, Y1 = px(b)
                    crop = lum(img[Y0:Y1, X0:X1].astype(np.float32))
                    if crop.size:
                        inkm = np.abs(crop - np.median(crop)) > 60
                        cols = inkm.any(0)
                        i = 0
                        while i < len(cols) and not cols[i]:
                            i += 1
                        j = i
                        while j < len(cols) and cols[j]:
                            j += 1
                        if j > i:
                            rows = np.where(inkm[:, i:j].any(1))[0]
                            if len(rows) and rows[-1] - rows[0] + 1 > 0.6 * inkm.shape[0] and j - i < 0.15 * len(cols):
                                boxes = [pymupdf.Rect((X0 + j + 1) / K, b.y0, b.x1, b.y1)]
                elif legend:
                    # "symbol = symbol name" legend: keep the symbols, cover only the trailing source text
                    X0, Y0, X1, Y1 = px(b)
                    crop = lum(img[Y0:Y1, X0:X1].astype(np.float32))
                    inkm = np.abs(crop - np.median(crop)) > 60 if crop.size else np.zeros((0, 0), bool)
                    rows = np.where(inkm[:, int(inkm.shape[1] * 0.6):].any(1))[0] if inkm.size else []
                    em = (rows[-1] - rows[0] + 1) / K / 0.88 if len(rows) else b.height * 0.8
                    xs = b.x1 - len(run[0]) * em - 0.3
                    ink = inkm.any(0) if inkm.size else np.zeros(0, bool)
                    gaps = [i for i in range(1, len(ink) - 1) if not ink[i] and not ink[i - 1]]
                    cand = [i for i in gaps if xs - 0.5 * em <= (X0 + i) / K <= xs + 0.1 * em]
                    if cand:
                        xs = (X0 + min(cand, key=lambda i: abs((X0 + i) / K - xs))) / K
                    boxes = [pymupdf.Rect(min(max(b.x0, xs), b.x1 - 3), b.y0, b.x1, b.y1)]
                    en = eng_part
                elif t.startswith('印から'):  # Japanese "from the mark": OCR dropped the marker before it; keep it visible
                    boxes = [pymupdf.Rect(b.x0 + b.height * 0.95, b.y0, b.x1, b.y1)]
                elif len(t) > 2 and t[0] in '囚回口ロ□■' and b.width > b.height * 2:
                    # leading swatch or letter box read as a glyph: keep it, replace only the name after it
                    boxes = [pymupdf.Rect(b.x0 + b.height * 0.95, b.y0, b.x1, b.y1)]
                    en = re.sub(r'^[囚回口ロ□■]\s*', '', en)
            if not pdf:
                en = re.sub(r'^[-–—/]+\s*(?=[(\d〈<])', '', en)  # OCR'd leader-line fragments
            vert = sum(1 for i in u['ids'] if items[i].get('v')) * 2 > len(u['ids'])
            if vert and len(boxes) == 1 and re.fullmatch(r'\(?\d+段\)?', t):
                # vertical row count "(47段)" beside a horizontal length "17c": one rotated "17c (47 rows)" label
                v = boxes[0]
                for it in items:
                    if it.get('tl') or not re.fullmatch(r'\d+(\.\d+)?c', it['text']):
                        continue
                    cb = pymupdf.Rect(it['bbox'])
                    above = cb.x0 < (v.x0 + v.x1) / 2 < cb.x1 and v.y0 - 8 < cb.y1 < v.y0 + 5
                    left = v.x0 - 6 < cb.x1 < v.x0 + 4 and v.y0 - 4 < (cb.y0 + cb.y1) / 2 < v.y0 + v.height * 0.5
                    if above or left:
                        boxes = [cb, v]
                        if not en.startswith(it['text']):
                            en = f"{it['text']} {en}"
                        break
            work.append((u['ids'], en if pdf else untag(en), boxes, vert, pdf))
        all_boxes = [pymupdf.Rect(x['bbox']) for x in items]
        # text may grow into blank space past the existing text margins, but at most half way to the page edge
        cbox = pymupdf.Rect(all_boxes[0]) if all_boxes else pymupdf.Rect(prect)
        for b in all_boxes[1:]:
            cbox |= b
        room = pymupdf.Rect((cbox.x0 + prect.x0) / 2, (cbox.y0 + prect.y0) / 2,
                            (cbox.x1 + prect.x1) / 2, (cbox.y1 + prect.y1) / 2) | (cbox + (-2, -2, 2, 2))
        room &= prect
        # paragraphs (several lines, inline [[graphics]], long lines with room below) are re-flowed as one block
        flows = {}
        for k_, (ids, en, boxes, vert, pdf) in enumerate(work):
            P = plan_flow(items, ids, en, img, room) if pdf and not vert else None
            if P:
                box = LAYOUT.get(str(n), {}).get(','.join(map(str, ids)), {}).get('box')
                if box:  # QA "FIT": the re-flowed block stays inside that free area (one line if it is one tall)
                    P['xR'] = P['xRext'] = box[2]
                    P['yb'] = max(P['base0'], box[3] - 0.3 * P['fs'])
                flows[tuple(ids)] = P
                flow_layout(items, ids, en, P)
            elif '[[' in en:  # inline-graphic markup that cannot be honoured: plain text
                work[k_] = (ids, re.sub(r'\[\[(.+?)\]\]', r'\1', en), boxes, vert, pdf)
        thick_all = [min(b.width, b.height) for _, _, bs, _, p_ in work if not p_ for b in bs]
        mask = np.zeros((H, W), np.uint8)
        plans = []
        for ids, en, boxes, vert, pdf in work:  # pass 1: redact (text layer) or flat fill / inpaint (OCR)
            covers = []
            for b in boxes:
                r = pymupdf.Rect(b.x0 - PAD, b.y0 - PAD, b.x1 + PAD, b.y1 + PAD) & prect
                rg = ring(img, r)
                bg = np.median(rg, 0)
                if pdf:
                    covers.append((r, bg, 'redact'))
                    continue
                close = np.abs(rg - bg).max(1) < 14
                solid = (close.mean() > 0.5 and np.abs(rg[close] - bg).max(1).mean() < 1.5) or any(items[i].get('solid') for i in ids)
                if solid:  # label inside a coloured shape: use the shape's colour
                    (x0, y0, x1, y1), m = stroke_mask(img, b)
                    inner = img[y0:y1, x0:x1][m == 0].astype(int)
                    if len(inner) > 0.4 * m.size:
                        ib = np.median(inner, 0)
                        if np.abs(ib - bg).max() > 25 and np.abs(inner - ib).max(1).mean() < 8:
                            bg = ib
                else:
                    (x0, y0, x1, y1), m = stroke_mask(img, r)
                    mask[y0:y1, x0:x1] |= m
                covers.append((r, bg, 'solid' if solid else 'inpaint'))
            plans.append((ids, en, boxes, vert, pdf, covers))
        # pass 2: remove source text before any English is written (redaction would remove it too)
        red = [items[i] for ids, _, _, _, pdf, _ in plans if pdf for i in ids]
        for it in red:
            for r in redact_rects(it):
                page.add_redact_annot(r, fill=False)
        if red:
            # underlines under replaced words would no longer line up with the English. Graphics redaction
            # ("touched") also deletes unrelated paths drawn in the same object, so paint the background colour
            # over the rule instead — only where the surroundings are a flat colour.
            hb = [pymupdf.Rect(it['bbox']) for it in red if it.get('rot') == 0]
            flowed = [pymupdf.Rect(items[i]['bbox']) for f_ids in flows for i in f_ids]
            under = []
            for g in page.get_drawings():
                gr = g['rect']
                if gr.height > 1.5 or gr.width < 0.3 or len(g['items']) > 4 or any(it[0] != 'l' for it in g['items']):
                    continue
                for b in hb:
                    if b.y1 - 2 <= gr.y0 <= b.y1 + 3 and gr.width <= b.width + 1 and \
                            min(gr.x1, b.x1) - max(gr.x0, b.x0) >= 0.8 * gr.width:
                        hw = (g.get('width') or 1) / 2 + 0.35
                        # re-flowed paragraphs: the text above the rule is gone too, so its ring need not be flat;
                        # cover a full pixel row more there, or anti-aliasing leaves a faint line at screen zoom
                        vw = max(hw, 1.0) if b in flowed else hw
                        r = pymupdf.Rect(gr.x0 - hw, gr.y0 - vw, gr.x1 + hw, gr.y1 + vw)
                        rg = ring(img, r, 4)
                        bg = np.median(rg, 0)
                        if (np.abs(rg - bg).max(1) < 14).mean() > 0.9 or b in flowed:
                            under.append((r, bg))
                        break
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
            for r, bg in under:
                page.draw_rect(r, color=None, fill=tuple(float(v) / 255 for v in bg), overlay=True)
        # inline graphics of re-flowed paragraphs move: remove the original shape, redraw it in pass 3
        moving = [pymupdf.Rect(items[i]['inline']) for P in flows.values() for i in P['inline']]
        src_paths = pristine[n - 1].get_drawings() if flows else []
        gfx_paths = lambda br: [g for g in src_paths if (br + (-0.8, -0.8, 0.8, 0.8)).contains(g['rect'])]
        # graphics attached to re-flowed words — a leader from an underline or an inline graphic to a note —
        # follow those words together with the note, when the new place is clear; otherwise they stay put
        flowed_ids = {i for f in flows for i in f}
        text_boxes = [pymupdf.Rect(x['bbox']) + (0.5, 0.5, -0.5, -0.5) for x in items]
        small = [g for g in src_paths if g['rect'].width < 60 and g['rect'].height < 40
                 and not any(g['rect'].intersects(t) for t in text_boxes)]
        unit_of = {i: tuple(ids) for ids, *_ in work for i in ids}
        en_of = {tuple(ids): en for ids, en, *_ in work}
        shifts, carried, taken = {}, [], set()
        for f_ids, P in flows.items():
            for key, a in P['anchors'].items():
                if key[0] == 'G':
                    gi = next(items[i] for i in P['inline'] if tuple(items[i]['inline']) == key[1])
                    rules = [m['rule'] for m in gi.get('inline_marks', [])]
                    for x0, y, x1 in rules:  # the marks move with the graphic: remove the originals
                        r = pymupdf.Rect(x0 - 1, y - 1, x1 + 1, y + 1)
                        carried.append(([], None, [(r, np.median(ring(img, r, 3), 0))]))
                else:
                    rules = [list(key[1])]
                for x0, y, x1 in rules:
                    rr = pymupdf.Rect(x0, y - 1, x1, y + 1)
                    grp = [g for g in small if id(g) not in taken and g['rect'].intersects(rr + (-1.5, -1.5, 1.5, 1.5))
                           and not (rr + (-0.8, -0.8, 0.8, 0.8)).contains(g['rect'])]
                    if not grp:
                        continue
                    seed = grp[0]['rect']
                    grown = True
                    while grown and len(grp) < 12:
                        grown = False
                        for g in small:
                            if g not in grp and id(g) not in taken and not (rr + (-0.8, -0.8, 0.8, 0.8)).contains(g['rect']) \
                                    and any(g['rect'].intersects(h['rect'] + (-1.5, -1.5, 1.5, 1.5)) for h in grp):
                                grp.append(g)
                                grown = True
                    cx = min(max((seed.x0 + seed.x1) / 2, x0), x1)
                    if key[0] == 'G':
                        nx, ny = a['x'] + (cx - a['x0']) * a['sx'], y + a['dy']
                    else:  # same fraction along the underline, now possibly split over several lines
                        total = sum(s[1] - s[0] for s in a['spans']) or 1
                        pos = (cx - x0) / max(x1 - x0, 0.1) * total
                        for s0, s1, base in a['spans']:
                            nx, ny = min(s0 + pos, s1), base + a['depth']
                            pos -= s1 - s0
                            if pos <= 0:
                                break
                    dx, dy = nx - cx, ny - y
                    area = grp[0]['rect']
                    for g in grp[1:]:
                        area |= g['rect']
                    notes = [j for j, x in enumerate(items) if j not in f_ids
                             and pymupdf.Rect(x['bbox']).intersects(area + (-3, -3, 3, 3))]
                    note_units = {unit_of[j] for j in notes if j in unit_of}
                    if f_ids in note_units or (abs(dx) < 0.5 and abs(dy) < 0.5):
                        continue
                    # the note as drawn: its re-flowed lines, else its source boxes
                    note_rects = [b for u_ in note_units if u_ in flows for b in flows[u_]['lines']] + \
                                 [pymupdf.Rect(items[j]['bbox']) for j in notes if unit_of.get(j) not in flows]
                    obst = [pymupdf.Rect(x['bbox']) for j, x in enumerate(items) if j not in flowed_ids and j not in notes] + \
                           [b for k2, f2 in flows.items() if k2 not in note_units for b in f2['lines']]
                    # text boxes span the full ascent and descent: overlaps under ~1.5pt leave visible air
                    fits = lambda rs: all(prect.contains(m) for m in rs) and \
                        not any(m.intersects(o + (1.5, 1.5, -1.5, -1.5)) for m in rs for o in obst)
                    T, moved_notes, relaid = None, False, {}
                    mv = lambda r, dx=dx, dy=dy: r + (dx, dy, dx, dy)
                    plain_notes = [mv(pymupdf.Rect(items[j]['bbox'])) for j in notes if unit_of.get(j) not in flows]
                    if fits([mv(g['rect']) for g in grp] + [mv(r) for r in note_rects]):
                        moved_notes = True
                    elif any(u_ in flows for u_ in note_units):
                        # a note re-flowed into blank space at its old place may not fit at the new one: lay it
                        # out again there as one line (in the words' column, shrinking a little if needed)
                        for u_ in note_units:
                            if u_ in flows:
                                N2 = dict(flows[u_])
                                N2.update(base0=N2['base0'] + dy, xf=N2['xf'] + dx, xr=N2['xr'] + dx,
                                          xR=N2['xR'] + dx)
                                N2.update(yb=N2['base0'], xRext=max(N2['xR'], P['xRext']))
                                flow_layout(items, list(u_), en_of[u_], N2)
                                relaid[u_] = N2
                        moved_notes = fits([mv(g['rect']) for g in grp] + plain_notes +
                                           [b for N2 in relaid.values() for b in N2['lines']])
                    if not moved_notes:
                        # note stays: stretch the leader so its start follows the words and its end (the side
                        # facing the note) stays where it was
                        nb = pymupdf.Rect(items[notes[0]]['bbox']) if notes else None
                        for j in notes[1:]:
                            nb |= pymupdf.Rect(items[j]['bbox'])
                        side = lambda lo, hi, a0, a1: hi if nb is None or a0 >= hi - 2 else (lo if a1 <= lo + 2 else (lo + hi) / 2)
                        ex = side(area.x0, area.x1, nb.x0 if nb else 0, nb.x1 if nb else 0)
                        ey = side(area.y0, area.y1, nb.y0 if nb else 0, nb.y1 if nb else 0)
                        w_ = lambda e, a0, p, d: 1.0 if abs(e - a0) < 0.5 else min(1.0, max(0.0, (e - p) / (e - a0)))
                        T = lambda p, ex=ex, ey=ey, cx=cx, y=y, dx=dx, dy=dy: pymupdf.Point(
                            p.x + dx * w_(ex, cx, p.x, dx), p.y + dy * w_(ey, y, p.y, dy))
                        # the words moved past the note, or the stretched leader would cross text: a detached
                        # leader points at nothing, so it moves with the words and its note even if tight
                        if (ex - cx) * (ex - nx) < 0 or (ey - y) * (ey - ny) < 0 or \
                                not fits([pymupdf.Rect(T(g['rect'].tl), T(g['rect'].br)).normalize() for g in grp]):
                            moved_notes = True
                            print(f'p{n}: leader and note moved with their words into a tight spot; check it')
                    if moved_notes:
                        T = lambda p, dx=dx, dy=dy: pymupdf.Point(p.x + dx, p.y + dy)
                    taken |= {id(g) for g in grp}
                    paint_ = []
                    for g in grp:
                        r = g['rect'] + (-0.8, -0.8, 0.8, 0.8)
                        paint_.append((r, np.median(ring(img, r, 3), 0)))
                    carried.append((grp, T, paint_))
                    for u_ in note_units if moved_notes else []:
                        if u_ in relaid:
                            flows[u_] = relaid[u_]
                        elif u_ in flows:  # re-flowed note: move its placed words and line boxes
                            N = flows[u_]
                            N['placed'] = [(k, s, m, x + dx, w2, b + dy) for k, s, m, x, w2, b in N['placed']]
                            N['lines'] = [r + (dx, dy, dx, dy) for r in N['lines']]
                            for a2 in N['anchors'].values():
                                for sp in a2.get('spans', []):
                                    sp[0] += dx; sp[1] += dx; sp[2] += dy
                        else:
                            shifts[u_] = (dx, dy)
        paint, probes_ = [], []
        for br in moving:
            # a tiny "touched" redaction inside the box deletes just that path, unless a larger shape (page
            # background, panel) also lies under that point; then paint the surrounding colour over it instead
            pr = pymupdf.Rect(br.x0 + 1, br.y0 + br.height / 2 - 0.1, br.x0 + 1.2, br.y0 + br.height / 2 + 0.1)
            if any(g['rect'].intersects(pr) and not (br + (-0.8, -0.8, 0.8, 0.8)).contains(g['rect']) for g in src_paths):
                rg = ring(img, br + (-1, -1, 1, 1), 4)
                paint.append((br + (-1, -1, 1, 1), np.median(rg, 0)))
            else:
                probes_.append(pr)
        for pr in probes_:
            page.add_redact_annot(pr, fill=False)
        if probes_:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                  graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                                  text=pymupdf.PDF_REDACT_TEXT_NONE)
        for r, bg in paint + [p_ for *_, ps in carried for p_ in ps]:
            page.draw_rect(r, color=None, fill=tuple(float(v) / 255 for v in bg), overlay=True)
        inp = cv2.inpaint(img, mask, 4, cv2.INPAINT_TELEA) if mask.any() else None
        for *_, covers in plans:
            for r, bg, kind in covers:
                if kind == 'solid':
                    page.draw_rect(r, color=None, fill=tuple(float(v) / 255 for v in bg), overlay=True)
                elif kind == 'inpaint':
                    x0, y0, x1, y1 = px(r)
                    _, buf = cv2.imencode('.png', cv2.cvtColor(inp[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
                    page.insert_image(pymupdf.Rect(x0 / K, y0 / K, x1 / K, y1 / K), stream=buf.tobytes(), overlay=True)
        for ids, en, boxes, vert, pdf, covers in plans:  # pass 3: English
            if not en.strip():
                continue
            it0 = items[ids[0]]
            pl = untag(en)  # measuring text: style tags draw nothing
            if pdf and it0.get('rot') == -1:
                write_rotated(page, it0, pl, it0['color'])
                continue
            if tuple(ids) in flows:
                draw_flow(page, flows[tuple(ids)], gfx_paths)
                continue
            bg0 = covers[0][1]
            sd = shifts.get(tuple(ids))
            if sd:  # a note travelling with the words its leader points at
                boxes = [b + (sd[0], sd[1], sd[0], sd[1]) for b in boxes]
            if pdf:  # exact style from the text layer; box at least one line tall
                union = pymupdf.Rect(boxes[0])
                for b in boxes[1:]:
                    union |= b
                fs = float(np.median([items[i]['size'] for i in ids]))
                color = seen_color(img, it0, bg0) if not it0.get('styles') else tuple(it0['color'])
                bold = it0.get('bold', False)
                if len(ids) == 1:
                    if vert:
                        cx, hw = (union.x0 + union.x1) / 2, max(union.width, fs * 1.25) / 2
                        union = pymupdf.Rect(cx - hw, union.y0, cx + hw, union.y1)
                    else:
                        cy, hh = (union.y0 + union.y1) / 2, max(union.height, fs * 1.25) / 2
                        union = pymupdf.Rect(union.x0, cy - hh, union.x1, cy + hh)
            else:
                union = pymupdf.Rect(covers[0][0])
                for cv_ in covers[1:]:
                    union |= cv_[0]
                color, bold = text_color(img, boxes[0], bg0), False
                th = sorted((b.width if vert else b.height) for b in boxes)[len(boxes) // 2]
                sim = [v for v in thick_all if 0.75 * th <= v <= 1.33 * th]
                fs = max(4.0, (float(np.median(sim)) if sim else th) * 0.72)
            lay = LAYOUT.get(str(n), {}).get(','.join(map(str, ids)), {})
            wrapped = False
            rect = pymupdf.Rect(lay['box']) if lay.get('box') else pymupdf.Rect(union)  # QA "FIT": exact free area
            # alignment: QA override, else centred when the source sits centred between graphics (box, frame)
            align = lay.get('align')
            if not align and pdf and len(ids) == 1 and not vert and it0.get('rot') == 0:
                align = column_align(ids[0], items)
            if not align and pdf and len(ids) == 1 and not vert and it0.get('rot') == 0:
                gl, gr_ = enclosure_gaps(img, boxes[0], bg0)
                if gl is not None and gr_ is not None and abs(gl - gr_) <= max(1.5, 0.2 * (gl + gr_)):
                    align = 'center'
            if len(boxes) == 1 and not lay.get('box'):  # grow single-line labels into blank space instead of shrinking
                f = fonts[bold]
                wide = sum(1 for ch in pl if ord(ch) >= 0x2190)
                need = f.text_length(''.join(ch for ch in pl if ord(ch) < 0x2190), fs) + wide * fs + 2
                have = union.height if vert else union.width
                others = [o for o in all_boxes if o != boxes[0]]
                if need > have:
                    deficit, b0, best = need - have, boxes[0], rect
                    if pdf:  # tightly stacked labels: test only the glyph core, not the line box
                        b0 = b0 + ((b0.width * 0.2, 0, -b0.width * 0.2, 0) if vert else (0, b0.height * 0.2, 0, -b0.height * 0.2))
                    modes = [(0, 1)] if len(it0['text']) >= 9 else [(0.5, 0.5), (0, 1), (1, 0)]
                    if align:  # keep the anchor: centred text grows both ways, right-aligned grows left
                        modes = {'center': [(0.5, 0.5)], 'right': [(1, 0)], 'left': [(0, 1)]}[align]
                    for fa, fz in modes:
                        got = pymupdf.Rect(union)
                        for step in range(1, 41):
                            g = deficit * step / 40
                            lo, hi = g * fa, g * fz
                            if vert:
                                cand = pymupdf.Rect(union.x0, union.y0 - lo, union.x1, union.y1 + hi)
                                # new area only, 0.6pt clear of the source glyphs (their anti-aliased edges are ink)
                                ext = [pymupdf.Rect(b0.x0 if pdf else cand.x0, cand.y0, b0.x1 if pdf else cand.x1, union.y0 - 0.6),
                                       pymupdf.Rect(b0.x0 if pdf else cand.x0, union.y1 + 0.6, b0.x1 if pdf else cand.x1, cand.y1)]
                                probes = [pymupdf.Rect(b0.x0, cand.y0, b0.x1, union.y0), pymupdf.Rect(b0.x0, union.y1, b0.x1, cand.y1)]
                            else:
                                cand = pymupdf.Rect(union.x0 - lo, union.y0, union.x1 + hi, union.y1)
                                ext = [pymupdf.Rect(cand.x0, b0.y0 if pdf else cand.y0, union.x0 - 0.6, b0.y1 if pdf else cand.y1),
                                       pymupdf.Rect(union.x1 + 0.6, b0.y0 if pdf else cand.y0, cand.x1, b0.y1 if pdf else cand.y1)]
                                probes = [pymupdf.Rect(cand.x0, b0.y0, union.x0, b0.y1), pymupdf.Rect(union.x1, b0.y0, cand.x1, b0.y1)]
                            if not room.contains(cand) or any(not p.is_empty and p.intersects(o) for p in probes for o in others):
                                break
                            if any(not e.is_empty and ink_fraction(img, e, bg0) > 0.002 for e in ext):
                                break
                            got = cand
                        if (got.height if vert else got.width) > (best.height if vert else best.width) + 0.5:
                            best = got
                        if (best.height if vert else best.width) >= need:
                            break
                    rect = best
                    if not vert and rect.width < need * 0.85 and ' ' in pl.strip():
                        # still too small on one line: try two lines in blank space above, below or around it
                        wds = pl.split()
                        w2 = min(max(width(' '.join(wds[:k]), fs, f), width(' '.join(wds[k:]), fs, f))
                                 for k in range(1, len(wds))) + 2
                        h2 = fs * 1.18 * 2 + 1
                        cw = max(w2, union.width)
                        # text-layer lines are left-aligned unless shown otherwise: keep the left edge first
                        order = [align] if align else (['left', 'center', 'right'] if pdf else ['center', 'left', 'right'])
                        xo_by = {'center': (union.x0 + union.x1 - cw) / 2, 'left': union.x0, 'right': union.x1 - cw}
                        for yo in (union.y1 - h2, union.y0, (union.y0 + union.y1 - h2) / 2):
                            for side in order:
                                xo = xo_by[side]
                                cand = pymupdf.Rect(xo, yo, xo + cw, yo + h2)
                                core = cand + (0.4, 0.4, -0.4, -0.4)  # text boxes of neighbouring rows touch
                                if wrapped or not room.contains(cand) or any(core.intersects(o) for o in others):
                                    continue
                                # new area only, kept 0.6pt clear of the source glyphs and of neighbours' edges
                                outside = [pymupdf.Rect(cand.x0, cand.y0, cand.x1, union.y0 - 0.6),
                                           pymupdf.Rect(cand.x0, union.y1 + 0.6, cand.x1, cand.y1),
                                           pymupdf.Rect(cand.x0, union.y0, union.x0 - 0.6, union.y1),
                                           pymupdf.Rect(union.x1 + 0.6, union.y0, cand.x1, union.y1)]
                                if any(r_.x1 > r_.x0 and r_.y1 > r_.y0 and
                                       ink_fraction(img, r_ + (0.3, 0.3, -0.3, -0.3), bg0) > 0.003 for r_ in outside):
                                    continue
                                rect, wrapped, wside = cand, True, side
                        if wrapped and not align:
                            align = wside
            if not align:
                align = 'left' if (pdf or len(boxes) > 1 or (rect.height if vert else rect.width) > 100) else 'center'
            two = lay.get('box') and rect.height > fs * 1.18 * 1.9  # QA gave a box tall enough to wrap
            wrap = 'normal' if (len(ids) > 1 or two or wrapped) else 'nowrap'
            rot = it0.get('rot', 270 if vert else 0) if pdf else (270 if vert else 0)
            italic = bool(pdf and it0.get('italic'))
            under = bool(pdf and it0.get('underline'))
            css = (f"* {{font-family: sans-serif; font-size: {fs:.2f}px; line-height: 1.18; color: #{'%02x%02x%02x' % tuple(color)};"
                   f" font-weight: {'bold' if bold else 'normal'}; font-style: {'italic' if italic else 'normal'};"
                   f" text-decoration: {'underline' if under else 'none'};"
                   f" text-align: {align}; white-space: {wrap}; margin: 0; padding: 0;}}")
            # runs styled unlike the line (<sN> tags from the translation) keep their weight, slant, colour, size
            base_st = {'bold': bold, 'italic': italic, 'color': list(color), 'size': fs}
            S = {k: v for i in ids for k, v in items[i].get('styles', {}).items()} if pdf else {}
            body = ''
            for t, st in style_runs(re.sub(r'(?<= ) ', '\u00a0', en), base_st, S):  # HTML would collapse runs
                e = html.escape(t).replace('\n', '<br>')
                if st is not base_st:
                    sc = [f"font-weight: {'bold' if st['bold'] else 'normal'}",
                          f"font-style: {'italic' if st.get('italic') else 'normal'}",
                          f"text-decoration: {'underline' if st.get('underline') else 'none'}",
                          'color: #%02x%02x%02x' % tuple(st['color'])]
                    if abs(st['size'] / fs - 1) > 0.05:
                        sc.append(f"font-size: {st['size'] / fs:.2f}em")
                    e = f'<span style="{"; ".join(sc)}">{e}</span>'
                body += e
            symc = {k: v for i in ids for k, v in items[i].get('symcolors', {}).items()} if pdf else {}
            for sym, col in symc.items():
                body = body.replace(html.escape(sym), f"<span style=\"color:#{'%02x%02x%02x' % tuple(col)}\">{html.escape(sym)}</span>")
            page.insert_htmlbox(rect, body, css=css, scale_low=0,
                                rotate=rot if rot in (0, 90, 180, 270) else 0, overlay=True)
        for grp, T, _ in carried:  # attached leaders, following the words they point from
            copy_paths(page, grp, T)
        for r in RESTORE.get(str(n), []):  # pass 4: bring back clipped original artwork
            page.show_pdf_page(pymupdf.Rect(r), pristine, n - 1, clip=pymupdf.Rect(r), overlay=True)
    if skipped:
        print('kept source text (English still has source letters; fix via fixes.json):', ' '.join(skipped))
    keep = [p - 1 for p in pages]
    if len(keep) < doc.page_count:
        doc.select(keep)
    meta = doc.metadata
    title = meta.get('title') or ''
    if not title or letters.search(title):  # source-script title: use the file name instead
        title = os.path.splitext(os.path.basename(c['src']))[0]
    meta['title'] = title + ' (English)'
    doc.set_metadata(meta)
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    return doc.page_count


def cmd_render(a):
    c = cfg(a.work)
    pages = pages_arg(a.pages, c['pages'])
    out = a.out or os.path.join(a.work, 'preview.pdf' if a.pages else 'out.pdf')
    n = _render(a, pages, out)
    pmap = out + '.pages.json'
    if n < c['pages']:  # subset render: remember which book page each PDF page is
        jsave(pmap, pages)
    elif os.path.exists(pmap):
        os.remove(pmap)
    print(f'{out} ({n} pages, {os.path.getsize(out) // 1048576} MB)')
    if a.png:
        _png(out, range(1, n + 1), a.png, os.path.join(a.work, 'png'), pages)


def _png(pdf, idx, dpi, d, names):
    import pymupdf
    os.makedirs(d, exist_ok=True)
    doc = pymupdf.open(pdf)
    paths = []
    for i, name in zip(idx, names):
        p = os.path.join(d, f'p{name:03d}.png')
        doc[i - 1].get_pixmap(dpi=dpi).save(p)
        paths.append(p)
    print(' '.join(paths))


def _translated_src(a, page):
    """latest render holding the page: `render -p` writes preview.pdf (+ page map), a full render out.pdf"""
    if getattr(a, 'pdf', None):
        return a.pdf, page - 1
    src, prev = os.path.join(a.work, 'out.pdf'), os.path.join(a.work, 'preview.pdf')
    pmap = jload(prev + '.pages.json', None)
    if pmap and page in pmap and os.path.exists(prev) and \
            (not os.path.exists(src) or os.path.getmtime(prev) > os.path.getmtime(src)):
        return prev, pmap.index(page)
    return src, page - 1


def cmd_crop(a):
    """Zoom into a page: by default one image with the translation on top and the original below (6px red rule
    between, same scale), so a comparison costs one image. Images are cached in W/crops per page, side and rect
    and reused while newer than their sources, so the main agent opens the files QA subagents cite."""
    import pymupdf
    from PIL import Image
    c = cfg(a.work)
    r = pymupdf.Rect([float(v) for v in a.rect.split(',')])
    tsrc, tidx = _translated_src(a, a.page)
    side = 'o' if a.orig else ('t' if a.trans else 'p')
    srcs = {'o': [(c['src'], a.page - 1)], 't': [(tsrc, tidx)], 'p': [(tsrc, tidx), (c['src'], a.page - 1)]}[side]
    tag = '_'.join(str(int(v)) for v in r) + ('' if a.dpi == 250 else f'_{a.dpi}dpi')
    p = os.path.join(a.work, 'crops', f"p{a.page}_{side}_{tag}.png")
    if not (os.path.exists(p) and all(os.path.getmtime(p) >= os.path.getmtime(s) for s, _ in srcs)):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        ims = []
        for s, i in srcs:
            pm = pymupdf.open(s)[i].get_pixmap(dpi=a.dpi, clip=r)
            ims.append(Image.frombytes('RGB', (pm.width, pm.height), pm.samples) if pm.n == 3 else
                       Image.frombytes('RGBA', (pm.width, pm.height), pm.samples).convert('RGB'))
        out = ims[0]
        if len(ims) == 2:
            out = Image.new('RGB', (max(i.width for i in ims), ims[0].height + 6 + ims[1].height), (230, 0, 0))
            out.paste(ims[0], (0, 0))
            out.paste(ims[1], (0, ims[0].height + 6))
        out.save(p)
    print(p)


def cmd_ink(a):
    """Tight bbox (pt) of the ink inside each rect, against the rect's own border colour: measures rendered
    text or artwork without hand-written pixel scripts."""
    import numpy as np, pymupdf
    c = cfg(a.work)
    src, idx = (c['src'], a.page - 1) if a.orig else _translated_src(a, a.page)
    page = pymupdf.open(src)[idx]
    for spec in a.rects:
        r = pymupdf.Rect([float(v) for v in spec.split(',')])
        pm = page.get_pixmap(dpi=300, clip=r)
        img = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3].astype(int)
        ring = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
        ink = np.abs(img - np.median(ring, 0)).max(2) > 60
        ys, xs = np.where(ink)
        if not len(xs):
            print(f'{spec}: blank')
            continue
        s = 72 / 300
        print(f'{spec}: ink {r.x0 + xs.min() * s:.1f},{r.y0 + ys.min() * s:.1f},{r.x0 + (xs.max() + 1) * s:.1f},'
              f'{r.y0 + (ys.max() + 1) * s:.1f}')


def cmd_units(a):
    """Units of a page as the renderer sees them: ids | source bbox | size, weight | source (style-tagged) =>
    current English (`*` = from fixes.json, `L` = layout override). Saves reading ocr/tr/fixes by hand."""
    FIX = jload(os.path.join(a.work, 'fixes.json'), {})
    LAY = jload(os.path.join(a.work, 'layout.json'), {})
    for n in a.pages:
        items = jload(os.path.join(a.work, 'ocr', f'p{n:03d}.json'), [])
        units = jload(os.path.join(a.work, 'tr', f'p{n:03d}.json'), {'units': []})['units']
        if not units:  # not translated yet: one row per translatable line, so the source can be read the same way
            units = [{'ids': [i], 'en': '(untranslated)'} for i, x in enumerate(items) if x.get('tl')]
        print(f'== page {n}')
        styles = {k: v for x in items for k, v in x.get('styles', {}).items()}
        if styles:
            print('styles: ' + '; '.join(f's{k} = {_style_desc(v)}' for k, v in sorted(styles.items(), key=lambda kv: int(kv[0]))))
        for u in units:
            if not u['ids']:
                continue
            key = ','.join(map(str, u['ids']))
            fx = FIX.get(str(n), {})
            en = fx.get(key, u['en'])
            bx = [items[i]['bbox'] for i in u['ids']]
            b = [round(min(v[0] for v in bx)), round(min(v[1] for v in bx)), round(max(v[2] for v in bx)),
                 round(max(v[3] for v in bx))]
            it = items[u['ids'][0]]
            st = f"{it.get('size', 0):g}pt{' B' if it.get('bold') else ''}" if it.get('src') == 'pdf' else 'ocr'
            src = ' '.join(items[i].get('styled', items[i]['text']) for i in u['ids'])
            flag = ('*' if key in fx else '') + ('L' if key in LAY.get(str(n), {}) else '')
            print(f"{key} | {b[0]},{b[1]},{b[2]},{b[3]} | {st} | {src} => {en}{' ' + flag if flag else ''}")


def cmd_width(a):
    """Width (pt) of English drawn in a unit's font, size and style runs, next to the source line's width:
    tells whether a shorter wording fits before proposing it."""
    import pymupdf
    items = jload(os.path.join(a.work, 'ocr', f'p{a.page:03d}.json'), [])
    ids = [int(v) for v in a.ids.split(',')]
    it = items[ids[0]]
    S = {k: v for i in ids for k, v in items[i].get('styles', {}).items()}
    fonts = {(False, False): pymupdf.Font('helv'), (True, False): pymupdf.Font('hebo'),
             (False, True): pymupdf.Font('heit'), (True, True): pymupdf.Font('hebi')}
    base = {'bold': it.get('bold', False), 'italic': it.get('italic', False), 'size': it.get('size', 10)}
    for text in a.texts:
        w, pos = 0.0, 0
        for m in list(re.finditer(r'<s(\d+)>(.*?)</s\1>', text)) + [None]:
            seg = text[pos:m.start()] if m else text[pos:]
            w += fonts[base['bold'], base['italic']].text_length(untag(seg), base['size'])
            if m:
                st = dict(base, **S.get(m.group(1), {}))
                w += fonts[bool(st['bold']), bool(st.get('italic'))].text_length(untag(m.group(2)), st['size'])
                pos = m.end()
        src_w = max(items[i]['bbox'][2] - items[i]['bbox'][0] for i in ids)
        print(f"{w:.1f}pt at {base['size']:g}pt{' bold' if base['bold'] else ''} (source line {src_w:.1f}pt): {text}")


# ---------------------------------------------------------------- residual scan / findings
def _resid_worker(args):
    pdf, pages, lg, use_ocr = args
    import numpy as np, pymupdf
    L = LANGS[lg]
    src_re = re.compile(L['src'])
    eng = ocr_engine(L) if use_ocr else None
    doc = pymupdf.open(pdf)
    out = []
    for n in pages:
        page = doc[n - 1]
        seen = []
        for b in page.get_text('dict')['blocks']:  # exact: text layer
            for l in b.get('lines', []):
                t = ''.join(s['text'] for s in l['spans'])
                if src_re.search(t):
                    bb = [round(v) for v in l['bbox']]
                    seen.append(bb)
                    out.append(f'{n} | {bb[0]},{bb[1]},{bb[2]},{bb[3]} | {t.strip()} [text]')
        for dpi in ((300, 200) if eng else ()):  # pixels: outlines, scans, text inside images
            pix = page.get_pixmap(dpi=dpi)
            r = eng(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy())
            if getattr(r, 'boxes', None) is None:
                continue
            for b, t, s in zip(r.boxes, r.txts, r.scores):
                real = [ch for ch in src_re.findall(t) if ch not in L['sym']]
                if not real or s < 0.6 or (lg == 'ko' and (len(real) < 2 or s < 0.75)):
                    continue
                bb = [round(min(p[0] for p in b) * 72 / dpi), round(min(p[1] for p in b) * 72 / dpi),
                      round(max(p[0] for p in b) * 72 / dpi), round(max(p[1] for p in b) * 72 / dpi)]
                if any(cover(bb, x) > 0.5 for x in seen):
                    continue
                seen.append(bb)
                out.append(f'{n} | {bb[0]},{bb[1]},{bb[2]},{bb[3]} | {t}')
    return out


def cmd_resid(a):
    """List source-language text still in the translated PDF: text layer (exact) + OCR of pixels."""
    from concurrent.futures import ProcessPoolExecutor
    import pymupdf
    lg = cfg(a.work)['lang']
    pdf = a.pdf or os.path.join(a.work, 'out.pdf')
    pages = pages_arg(a.pages, pymupdf.open(pdf).page_count)
    nw = max(1, min(a.jobs, len(pages)))
    with ProcessPoolExecutor(nw) as ex:
        res = [l for r in ex.map(_resid_worker, [(pdf, pages[i::nw], lg, not a.no_ocr) for i in range(nw)]) for l in r]
    pmap = jload(pdf + '.pages.json', None)
    if pmap:  # report book page numbers, not subset indices
        res = [f"{pmap[int(l.split(' |')[0]) - 1]} |{l.split(' |', 1)[1]}" for l in res]
    print('\n'.join(sorted(res, key=lambda s: int(s.split(' |')[0]))) or f"no residual {LANGS[lg]['name']}")


def cmd_apply(a):
    """Apply QA findings lines (see prompts/qa-agent.md) to fixes/manual/restore/layout."""
    w = a.work
    FIX, MANUAL, RESTORE, LAYOUT = (jload(os.path.join(w, f), {}) for f in
                                    ('fixes.json', 'manual.json', 'restore.json', 'layout.json'))

    def unit_at(n, bb, quoted=None):
        """unit under bbox; with `quoted` (current English from the finding) prefer units that contain it,
        so a slightly off bbox cannot hit the neighbouring label"""
        items = jload(os.path.join(w, 'ocr', f'p{int(n):03d}.json'), [])
        units = jload(os.path.join(w, 'tr', f'p{int(n):03d}.json'), {'units': []})['units']
        for m in MANUAL.get(n, []):  # QA-added labels are units too (render appends them in this order)
            items.append({'bbox': m['bbox']})
            units.append({'ids': [len(items) - 1], 'en': m['en']})
        cands = []
        for u in units:
            if not u['ids']:
                continue
            bx = [items[i]['bbox'] for i in u['ids']]
            ub = [min(b[0] for b in bx), min(b[1] for b in bx), max(b[2] for b in bx), max(b[3] for b in bx)]
            ov = cover(ub, bb) * (ub[2] - ub[0]) * (ub[3] - ub[1])
            sc = ov / max(1e-6, min((ub[2] - ub[0]) * (ub[3] - ub[1]), (bb[2] - bb[0]) * (bb[3] - bb[1])))
            cur = FIX.get(n, {}).get(','.join(map(str, u['ids'])), u['en']) or ''
            # centre distance breaks ties when the finding's bbox misses the source box
            d = abs((ub[0] + ub[2]) - (bb[0] + bb[2])) + abs((ub[1] + ub[3]) - (bb[1] + bb[3]))
            cands.append((sc, d, u, bool(quoted) and quoted.strip().lower() in untag(cur).lower()))
        hit = [c for c in cands if c[3]]
        if hit:
            return min(hit, key=lambda c: (-c[0], c[1]))[2]
        best = max(cands, key=lambda c: c[0], default=None)
        return best[2] if best and best[0] > 0.3 else None
    text = '\n'.join(sys.stdin.read() if f == '-' else open(f).read() for f in a.files)  # files may lack a final newline
    review = 0
    for line in text.splitlines():
        # trailing fields naming the crop images that show the problem are evidence for the reader, not data
        p = [s.strip().strip('`') for s in line.split('|')]
        p = [s for s in p if not s.lower().endswith('.png')]
        if len(p) < 3 or not p[0].isdigit():
            continue
        n, kind = p[0], p[1].upper()
        try:
            bb = [float(v) for v in p[2].split(',')]
        except ValueError:
            continue
        if kind in ('SRC', 'JP') and len(p) >= 6:
            MANUAL.setdefault(n, []).append({'bbox': bb, 'v': p[3].upper() == 'V', 'text': p[4], 'en': p[5],
                                             'solid': not p[5]})
        elif kind == 'DAMAGE':
            RESTORE.setdefault(n, []).append(bb)
        elif kind == 'MERGE' and len(p) >= 4:
            # one paragraph re-translated as a single unit: units (and loose inline graphics) inside the bbox are replaced
            trf = os.path.join(w, 'tr', f'p{int(n):03d}.json')
            items = jload(os.path.join(w, 'ocr', f'p{int(n):03d}.json'), [])
            tr = jload(trf, {'units': []})
            ubox = lambda ids: [min(items[i]['bbox'][0] for i in ids), min(items[i]['bbox'][1] for i in ids),
                                max(items[i]['bbox'][2] for i in ids), max(items[i]['bbox'][3] for i in ids)]
            inside = [u for u in tr['units'] if u['ids'] and all(i < len(items) for i in u['ids'])
                      and cover(ubox(u['ids']), bb) > 0.6]
            used = {i for u in tr['units'] for i in u['ids']}
            loose = [i for i, x in enumerate(items) if x.get('inline') and i not in used and cover(x['bbox'], bb) > 0.6]
            ids = [i for u in inside for i in u['ids']] + loose
            nb = sum(1 for i in ids if items[i].get('inline'))
            if not inside or nb != p[3].count('[['):
                print(f'REVIEW merge (units {len(inside)}, inline graphics {nb} vs [[ ]] {p[3].count("[[")}):', line); review += 1
                continue
            ids.sort(key=lambda i: (round(items[i].get('origin', items[i]['bbox'][:2])[1]), items[i]['bbox'][0]))
            for u in inside:
                FIX.get(n, {}).pop(','.join(map(str, u['ids'])), None)
            tr['units'] = [u for u in tr['units'] if u not in inside] + [{'ids': ids, 'en': p[3]}]
            jsave(trf, tr)
        elif kind == 'STYLE' and len(p) >= 4:
            # whole-unit weight / slant / colour, e.g. "bold #808080" or "regular upright"
            u = unit_at(n, bb)
            words = ' '.join(p[3:]).lower().split()
            sty = {}
            for wd in words:
                if wd in ('bold', 'regular', 'normal'):
                    sty['bold'] = wd == 'bold'
                elif wd in ('italic', 'upright'):
                    sty['italic'] = wd == 'italic'
                elif re.fullmatch(r'#?[0-9a-f]{6}', wd):
                    h = wd.lstrip('#')
                    sty['color'] = [int(h[k:k + 2], 16) for k in (0, 2, 4)]
            if not u or not sty:
                print('REVIEW style:', line); review += 1; continue
            LAYOUT.setdefault(n, {}).setdefault(','.join(map(str, u['ids'])), {}).setdefault('style', {}).update(sty)
        elif kind in ('ALIGN', 'FIT') and len(p) >= 4:
            u = unit_at(n, bb)
            if not u:
                print('REVIEW no-unit:', line); review += 1; continue
            lay = LAYOUT.setdefault(n, {}).setdefault(','.join(map(str, u['ids'])), {})
            if kind == 'ALIGN' and p[3].lower() in ('left', 'center', 'right'):
                lay['align'] = p[3].lower()
            elif kind == 'FIT':
                try:
                    lay['box'] = [float(v) for v in p[3].split(',')]
                except ValueError:
                    print('REVIEW bad box:', line); review += 1
        elif kind == 'WRONG' and len(p) >= 5:
            if p[3] and p[4].startswith(p[3]) and re.fullmatch(r'\s*\[[^\[\]]*\]\s*', p[4][len(p[3]):]):
                # the "correction" only appends a bracketed remark (e.g. a note about style): never print it
                print('REVIEW remark, not text:', line); review += 1; continue
            if p[3] == p[4]:
                # same text: the problem is not the wording (colour, size, placement) — needs STYLE/FIT/ALIGN
                print('REVIEW no change in text:', line); review += 1; continue
            best = unit_at(n, bb, p[3])
            if not best:
                print('REVIEW no-unit:', line); review += 1; continue
            key = ','.join(map(str, best['ids']))
            cur = FIX.get(n, {}).get(key, best['en']) or ''
            # multi-line unit: only patch the quoted part — unless the correction is the whole unit again (QA
            # quoted a fragment but rewrote everything), which would otherwise be pasted into the middle of itself
            whole = len(untag(p[4])) >= 0.8 * len(untag(cur))
            if len(untag(cur)) > len(p[3]) * 1.6 + 10 and not whole:
                if p[3] and p[3] in cur:
                    FIX.setdefault(n, {})[key] = cur.replace(p[3], p[4], 1)
                elif p[3] and p[3] in untag(cur):  # quote crosses a style tag: patch the untagged text
                    FIX.setdefault(n, {})[key] = untag(cur).replace(p[3], p[4], 1)
                else:
                    print(f'REVIEW long-unit p{n} ids={key}: {cur[:90]!r} | {p[3]} -> {p[4]}'); review += 1
                    continue
            else:
                FIX.setdefault(n, {})[key] = p[4]
    for f, o in (('fixes.json', FIX), ('manual.json', MANUAL), ('restore.json', RESTORE), ('layout.json', LAYOUT)):
        jsave(os.path.join(w, f), o)
    print(f'applied; {review} need review')


def cmd_validate(a):
    """Check W/tr against W/prompts: JSON shape, every T id covered exactly once. Bad files are removed so the
    page shows up as untranslated again."""
    bad, ok = [], 0
    for f in sorted(glob.glob(os.path.join(a.work, 'prompts', 'p*.txt'))):
        name = os.path.basename(f)[:-4]
        n = int(name[1:])
        tids = {int(l.split('\t')[0]) for l in open(f).read().splitlines()[2:] if l.split('\t')[1:2] == ['T']}
        tf = os.path.join(a.work, 'tr', name + '.json')
        try:
            raw = open(tf).read().strip()
            raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)  # tolerate fenced output
            units = json.loads(raw)['units']
            got = [int(i) for u in units for i in u['ids']]
            assert all('en' in u for u in units), 'unit without en'
            if not tids <= set(got) or len(got) != len(set(got)):
                raise ValueError(f'missing {sorted(tids - set(got))[:5]} dup {len(got) - len(set(got))}')
            jsave(tf, {'units': units})
            ok += 1
        except FileNotFoundError:
            bad.append(f'{n}:missing')
        except Exception as e:
            bad.append(f'{n}:{str(e)[:60]}')
            os.remove(tf)
    print(f'valid {ok}; redo: {" ".join(bad) or "-"}')


def cmd_unify(a):
    """Same source label on several pages (legends, headings, notes) -> same English: majority wins.
    Writes fixes.json only for outliers, never over an existing fix."""
    import collections
    c = cfg(a.work)
    FIX = jload(os.path.join(a.work, 'fixes.json'), {})
    seen = collections.defaultdict(list)  # source text -> [(page, key, english)]
    for n in range(1, c['pages'] + 1):
        items = jload(os.path.join(a.work, 'ocr', f'p{n:03d}.json'), [])
        for u in jload(os.path.join(a.work, 'tr', f'p{n:03d}.json'), {'units': []})['units']:
            if len(u['ids']) != 1 or not u['en']:
                continue
            key = str(u['ids'][0])
            en = FIX.get(str(n), {}).get(key, u['en'])
            if untag(en) != en:  # style tag numbers are per page: never copy them to another page
                continue
            seen[re.sub(r'\s+', ' ', items[u['ids'][0]]['text']).strip()].append((str(n), key, en))
    changed = texts = 0
    for occ in seen.values():
        votes = collections.Counter(en for _, _, en in occ)
        if len(votes) < 2:
            continue
        best, cnt = votes.most_common(1)[0]
        if cnt * 2 <= len(occ):  # no clear majority: leave for QA
            continue
        texts += 1
        for n, key, en in occ:
            if en != best and key not in FIX.get(n, {}):
                FIX.setdefault(n, {})[key] = best
                changed += 1
    jsave(os.path.join(a.work, 'fixes.json'), FIX)
    print(f'unified {changed} units across {texts} repeated labels')


def cmd_status(a):
    c = cfg(a.work)
    cnt = lambda d: len(glob.glob(os.path.join(a.work, d, 'p*.*')))
    todo = [int(os.path.basename(f)[1:4]) for f in glob.glob(os.path.join(a.work, 'prompts', 'p*.txt'))
            if not os.path.exists(os.path.join(a.work, 'tr', os.path.basename(f)[:-4] + '.json'))]
    print(f"lang={c['lang']} pages={c['pages']} ocr={cnt('ocr')} prompts={cnt('prompts')} tr={cnt('tr')} "
          f"untranslated={','.join(map(str, sorted(todo))) or '-'}")


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # source text is printed; C/POSIX locales are ASCII
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `kpt units … | head` ends quietly
    ap = argparse.ArgumentParser(prog='kpt')
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('init'); s.add_argument('pdf'); s.add_argument('-w', '--work', required=True)
    s.add_argument('-l', '--lang', choices=['auto', *LANGS], default='auto', help='source language (auto: text layer)')
    s = sp.add_parser('ocr'); s.add_argument('--fast', action='store_true', help='skip tiled OCR passes')
    s.add_argument('-m', '--mode', choices=['auto', 'text', 'ocr'], default='auto',
                   help='auto: text layer where the page has source text, else OCR')
    s.add_argument('-j', '--jobs', type=int, default=max(1, (os.cpu_count() or 4) // 4))
    s = sp.add_parser('prompts'); s.add_argument('-g', '--glossary', default='knitting')
    sp.add_parser('qa-sheet')
    s = sp.add_parser('correct'); s.add_argument('specs', nargs='+', help='P:ID=TEXT or P:ID=-')
    sp.add_parser('checked')
    s = sp.add_parser('render'); s.add_argument('-o', '--out'); s.add_argument('--png', type=int, default=0)
    s = sp.add_parser('crop'); s.add_argument('page', type=int); s.add_argument('rect')
    s.add_argument('--orig', action='store_true', help='original only'); s.add_argument('--trans', action='store_true',
                                                                                         help='translation only')
    s.add_argument('--pdf'); s.add_argument('--dpi', type=int, default=250)
    s = sp.add_parser('ink'); s.add_argument('page', type=int); s.add_argument('rects', nargs='+')
    s.add_argument('--orig', action='store_true'); s.add_argument('--pdf')
    s = sp.add_parser('units'); s.add_argument('pages', type=int, nargs='+')
    s = sp.add_parser('width'); s.add_argument('page', type=int); s.add_argument('ids'); s.add_argument('texts', nargs='+')
    s = sp.add_parser('resid'); s.add_argument('--pdf'); s.add_argument('-j', '--jobs', type=int, default=4)
    s.add_argument('--no-ocr', action='store_true', help='text layer only (fast)')
    s = sp.add_parser('apply'); s.add_argument('files', nargs='+', help="findings files (each agent's reply) or '-'")
    sp.add_parser('status')
    sp.add_parser('unify')
    sp.add_parser('validate')
    for name, p in sp.choices.items():
        if name != 'init':
            p.add_argument('-w', '--work', required=True)
        if name in ('ocr', 'prompts', 'qa-sheet', 'checked', 'render', 'resid'):
            p.add_argument('-p', '--pages', help='e.g. 5,9,38-42 (default all)')
    a = ap.parse_args()
    globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
