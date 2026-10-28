"""Replay a deixis-v2 chat (templated lines) to the final who-holds-what state, and classify a picked object."""
import re
def replay(prompt):
    lines = prompt.rstrip().split("\n")
    people = re.findall(r"A chat between (.+?)\.$", lines[0])[0].replace(" and ", ", ").split(", ")
    hold = {}; last_obj = None; others = lambda s: [p for p in people if p != s]
    for ln in lines[1:-1]:
        m = re.match(r"(\w+): (.*)$", ln)
        if not m: continue
        s, t = m.groups()
        objs = re.findall(r"\bthe (\w+)|\bmy (\w+)", t)
        for a, b in objs: last_obj = a or b
        r = (re.match(r"I have the (\w+)\.", t) or re.match(r"The (\w+) is mine\.", t))
        if r: hold[r.group(1)] = s; continue
        r = re.match(r"(?:Here, )?(\w+), take my (\w+)\.", t) or re.match(r"Here, (\w+), take my (\w+)\.", t)
        if r: hold[r.group(2)] = r.group(1); continue
        r = re.match(r"(\w+), my (\w+) is yours now\.", t)
        if r: hold[r.group(2)] = r.group(1); continue
        r = re.match(r"(\w+), I'm giving you my (\w+)\.", t) or re.match(r"I'm giving you my (\w+)\.", t)
        if r:
            if r.lastindex == 2: hold[r.group(2)] = r.group(1)
            else: o = others(s); hold[r.group(1)] = o[0] if len(o) == 1 else None
            continue
        r = re.match(r"Here, take my (\w+)\.", t) or re.match(r"My (\w+) is yours now\.", t)
        if r:
            o = others(s); hold[r.group(1)] = o[0] if len(o) == 1 else None; continue
    q = lines[-1]; qs = q.split(":")[0]
    addr = re.match(r"\w+: (\w+), right now you're holding the", q)
    addressee = addr.group(1) if addr else (others(qs)[0] if len(others(qs)) == 1 else None)
    return hold, qs, addressee, last_obj


def classify(prompt: str, pick: str, ok: bool, in_cands: bool) -> str:
    """speaker_confusion = picked the speaker's own object (you→I); other_participant = a third person's object;
    other_object = the last-mentioned object; format = not a candidate; other = anything else."""
    if ok:
        return "correct"
    if not in_cands:
        return "format"
    h, qs, ad, lo = replay(prompt)
    if h.get(pick) == qs:
        return "speaker_confusion"
    if h.get(pick) and h.get(pick) != ad:
        return "other_participant"
    if pick == lo:
        return "other_object"
    return "other"
