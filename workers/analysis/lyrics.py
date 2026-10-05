"""Lyric lab: themes, rhyme scheme, similes, metaphors, meaning summary.

Pattern-based (no heavy NLP models required). Every figurative-language hit
carries its source line so results stay checkable.
"""
from __future__ import annotations

import re
from collections import Counter

THEMES = {
    "Love & romance": ["love", "heart", "kiss", "hold", "baby", "darling", "romance", "tender", "adore"],
    "Heartbreak & loss": ["goodbye", "tears", "cry", "broken", "gone", "miss", "lonely", "hurt", "leave", "leaving", "goodbye"],
    "Ambition & hustle": ["grind", "hustle", "money", "rich", "dream", "chase", "boss", "empire", "million"],
    "Struggle & pain": ["pain", "struggle", "demon", "fight", "battle", "scar", "heavy", "dark", "bleed"],
    "Confidence & bravado": ["king", "queen", "legend", "crown", "flex", "unstoppable", "greatest"],
    "Party & celebration": ["party", "dance", "club", "tonight", "shots", "celebrate", "vibe"],
    "Faith & hope": ["pray", "faith", "god", "heaven", "hope", "bless", "angel"],
    "Nostalgia & memory": ["remember", "memory", "yesterday", "childhood", "used to", "back then"],
    "Identity & self": ["myself", "mirror", "mask", "soul", "mind", "who i am"],
}

POSITIVE = {"love", "happy", "joy", "free", "alive", "shine", "bright", "hope", "dream", "heaven", "bless", "win", "celebrate"}
NEGATIVE = {"pain", "hurt", "cry", "tears", "broken", "dark", "lonely", "goodbye", "die", "dead", "fear", "hate", "lost", "bleed"}

_WORD = re.compile(r"[a-z']+")


def _words(text):
    return _WORD.findall(text.lower())


def detect_themes(lines):
    scored = []
    for theme, keywords in THEMES.items():
        hits = []
        for i, line in enumerate(lines):
            wl = _words(line)
            matched = [k for k in keywords if k in wl or k in line.lower()]
            if matched:
                hits.append({"line_no": i + 1, "line": line, "keywords": sorted(set(matched))})
        if hits:
            scored.append({"theme": theme, "score": len(hits), "evidence": hits[:3]})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:4]


def lyrical_mood(lines):
    pos = neg = 0
    for line in lines:
        wl = set(_words(line))
        pos += len(wl & POSITIVE)
        neg += len(wl & NEGATIVE)
    if pos + neg == 0:
        return {"mood": "Neutral / ambiguous", "positive": pos, "negative": neg}
    ratio = pos / (pos + neg)
    label = "Uplifting" if ratio >= 0.65 else "Melancholy" if ratio <= 0.35 else "Bittersweet / mixed"
    return {"mood": label, "positive": pos, "negative": neg}


def _rhyme_key(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    return w[-3:] if len(w) >= 3 else w


def rhyme_scheme(lines):
    """Approximate suffix-based rhyme scheme per stanza (blank line = break)."""
    stanzas, cur = [], []
    for line in lines:
        if line.strip():
            cur.append(line)
        elif cur:
            stanzas.append(cur)
            cur = []
    if cur:
        stanzas.append(cur)
    out = []
    for si, st in enumerate(stanzas, 1):
        last_words = []
        for line in st:
            toks = re.findall(r"[A-Za-z']+", line)
            last_words.append(toks[-1] if toks else "")
        keys = [_rhyme_key(w) for w in last_words]
        mapping, scheme = {}, []
        nxt = ord("A")
        for k in keys:
            if k not in mapping:
                mapping[k] = chr(nxt)
                nxt += 1
            scheme.append(mapping[k])
        out.append(
            {
                "stanza": si,
                "scheme": "".join(scheme),
                "line_endings": last_words,
                "note": "approximate (suffix-based, not phonetic)",
            }
        )
    return out


def find_similes(lines):
    hits = []
    like_pat = re.compile(r"\blike\s+(a|an|the)?\s*([a-zA-Z' ]{1,32}?)(?=[,.;!?]|$)")
    as_pat = re.compile(r"\bas\s+([a-zA-Z' ]{1,24}?)\s+as\b")
    for i, line in enumerate(lines):
        for m in like_pat.finditer(line):
            phrase = m.group(0).strip().rstrip(",.;!?")
            if len(phrase.split()) >= 2:
                hits.append(
                    {
                        "type": "simile",
                        "line_no": i + 1,
                        "line": line,
                        "phrase": phrase,
                        "note": "explicit comparison using 'like'",
                    }
                )
        for m in as_pat.finditer(line):
            phrase = m.group(0).strip()
            hits.append(
                {
                    "type": "simile",
                    "line_no": i + 1,
                    "line": line,
                    "phrase": phrase,
                    "note": "explicit comparison using 'as ... as'",
                }
            )
    return hits


_NON_METAPHOR_STARTERS = {
    "good", "bad", "great", "nice", "ok", "okay", "fine", "sorry", "ready",
    "here", "there", "gone", "over", "done", "tired", "sick", "wrong",
}


def find_metaphors(lines):
    hits = []
    copula_pat = re.compile(
        r"\b(i am|i'm|you are|you're|he is|he's|she is|she's|it is|it's|"
        r"we are|we're|they are|they're|this is|that is)\s+"
        r"(a|an|the)\s+([^,.;!?]{2,42})",
        re.IGNORECASE,
    )
    my_pat = re.compile(
        r"\bmy\s+([a-zA-Z]+)\s+(is|are)\s+(a|an|the)?\s*([^,.;!?]{2,42})",
        re.IGNORECASE,
    )
    for i, line in enumerate(lines):
        for m in copula_pat.finditer(line):
            phrase = m.group(0).strip()
            tail = m.group(3).strip().lower()
            if "like" in tail or tail.split()[:1][0] in _NON_METAPHOR_STARTERS:
                continue
            hits.append(
                {
                    "type": "possible metaphor",
                    "line_no": i + 1,
                    "line": line,
                    "phrase": phrase,
                    "note": "'X is a Y' equates two unlike things — check the line in context",
                }
            )
        for m in my_pat.finditer(line):
            phrase = m.group(0).strip()
            if "like" in phrase.lower():
                continue
            hits.append(
                {
                    "type": "possible metaphor",
                    "line_no": i + 1,
                    "line": line,
                    "phrase": phrase,
                    "note": "'my X is Y' framing — likely figurative, verify in context",
                }
            )
    # de-dupe identical (line_no, phrase)
    seen, uniq = set(), []
    for h in hits:
        k = (h["line_no"], h["phrase"].lower())
        if k not in seen:
            seen.add(k)
            uniq.append(h)
    return uniq


def meaning_summary(lines, themes, mood, figurative):
    if not lines:
        return "No lyrics to interpret yet."
    top = [t["theme"] for t in themes[:2]]
    n_fig = len(figurative)
    parts = []
    if top:
        parts.append(f"thematically centered on {' and '.join(top).lower()}")
    parts.append(f"with a {mood['mood'].lower()} lyrical tone")
    key_lines = [f["line"].strip() for f in figurative[:2]]
    summary = "This song reads as " + ", ".join(parts) + "."
    if key_lines:
        summary += " Key figurative moments: " + " / ".join(f'"{l}"' for l in key_lines) + "."
    # narrative arc: dominant theme per third
    thirds = [lines[: len(lines) // 3 or 1], lines[len(lines) // 3: 2 * len(lines) // 3], lines[2 * len(lines) // 3:]]
    arc = []
    for name, chunk in zip(("opening", "middle", "closing"), thirds):
        c = Counter()
        for line in chunk:
            wl = set(_words(line))
            for theme, kws in THEMES.items():
                if wl & set(kws):
                    c[theme] += 1
        if c:
            arc.append(f"{name}: {c.most_common(1)[0][0].lower()}")
    if arc:
        summary += " Arc — " + "; ".join(arc) + "."
    return summary


def analyze_lyrics(lines, audio_mood=None):
    lines = [l for l in (ln.strip() for ln in lines) if l]
    themes = detect_themes(lines)
    mood = lyrical_mood(lines)
    similes = find_similes(lines)
    metaphors = find_metaphors(lines)
    figurative = similes + metaphors
    figurative.sort(key=lambda h: h["line_no"])
    return {
        "line_count": len(lines),
        "themes": themes,
        "lyrical_mood": mood,
        "rhyme_scheme": rhyme_scheme(lines),
        "similes": similes,
        "metaphors": metaphors,
        "figurative_count": len(figurative),
        "meaning_summary": meaning_summary(lines, themes, mood, figurative),
        "audio_mood": audio_mood,
        "method_notes": (
            "keyword themes, suffix-based rhyme approximation, regex simile/metaphor "
            "candidates. Treat metaphor hits as candidates to verify in context."
        ),
    }
