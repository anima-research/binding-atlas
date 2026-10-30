"""Neuron Atlas data: the compendium map, its layouts and the binding attribution → data/built/neurons/.

Every FFN neuron of kotodama-tsumugi at every effective position (72 × 4096 = 294,912) is one row, indexed
i = position * 4096 + neuron. Writes:
  neurons.json        manifest: columns and their byte offsets in neurons.bin, layouts, families, categories, clusters
                      (draft names), per-neuron draft labels, the hand-validated neurons, quantisation scales
  neurons.bin         columns, little-endian: fam u8, cluster i16, binding attribution i8 per family (signed sqrt
                      scale), binding-set bits u8
  layout-<key>.bin    x u16[N] then y u16[N], one shared scale per layout (aspect kept; 0.2 / 99.8 percentile clip)

Inputs (all from the compendium build; paths are arguments, nothing machine-specific is tracked):
  --map-v1 <dir>    the named meaning map: the neuron rows, the draft compendium, the taxonomy, the known neurons
  --map-v2 <dir>    the meaning map with the firing token masked (its neuron rows)
  --weights <file>  the weight-space layouts (optional; a JSON sibling adds the PCA statistics)
  --binding <file>  the mean binding attribution per family, shape (72, 4096)
  --sets <file>     the causal supporter / interferer sets

  python build/build_neurons.py --map-v1 <d1> --map-v2 <d2> --weights <w> --binding <b> --sets <s> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FAM_ORDER = ["Narration, speech & the Other", "Meta-layer: text about text", "Genre & register", "Discourse & reasoning",
             "Grammar & function", "Time, number & measurement", "Entities & domains", "Machine text & token-level",
             "Artefact & unassigned"]
BIND_FAMS = ["attr", "dialogue", "deixis", "adjacency", "answer", "codevar", "codevarmut", "setbind"]
SHOWN_BIND = ["attr", "dialogue", "deixis"]          # own columns; "all" = mean over the eight, each scaled to unit std
SET_BITS = ["attr_pos100", "attr_neg100", "dialogue_pos100", "dialogue_neg100", "deixis_pos100", "deixis_neg100", "core28"]
WEIGHT_LAYOUTS = {"write_centered_umap": "weights-write", "write_raw_pca": "weights-write-pca",
                  "read_centered_umap": "weights-read"}
N_POS, F = 72, 4096
N = N_POS * F


def read_map(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros(N, np.float32)
    y = np.zeros(N, np.float32)
    cl = np.full(N, -1, np.int16)
    seen = np.zeros(N, bool)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            i = r["L"] * F + r["n"]
            x[i], y[i], cl[i], seen[i] = r["x"], r["y"], r["cluster"], True
    if not seen.all():
        raise SystemExit(f"{path}: {int((~seen).sum())} neurons missing")
    return x, y, cl


def quantise(x: np.ndarray, y: np.ndarray, keep_aspect: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """u16 coordinates over the full range of the data (nothing clipped), plus the box the page frames: the 0.2 / 99.8
    percentile region, as [x0, y0, x1, y1] in stored units / 65535, widened on one axis to keep the aspect when
    keep_aspect (aspect = height / width of that box; 0 when each axis fills its own range, as for depth). The meta
    maps a data coordinate v to the stored one: (v - lo) / span, times 65535."""
    lo = np.array([x.min(), y.min()], np.float64)
    span = np.array([x.max(), y.max()], np.float64) - lo
    span[span == 0] = 1.0
    qx = np.round((x - lo[0]) / span[0] * 65535).astype(np.uint16)
    qy = np.round((y - lo[1]) / span[1] * 65535).astype(np.uint16)
    p_lo = np.array([np.percentile(x, 0.2), np.percentile(y, 0.2)])
    p_hi = np.array([np.percentile(x, 99.8), np.percentile(y, 99.8)])
    if keep_aspect:
        mid, half = (p_lo + p_hi) / 2, (p_hi - p_lo) / 2
        aspect = float(half[1] / half[0])
    else:
        p_lo, p_hi, aspect = lo, lo + span, 0.0
    fit = [float(v) for v in np.concatenate([(p_lo - lo) / span, (p_hi - lo) / span])]
    meta = {"aspect": round(aspect, 4), "lo": [round(float(v), 5) for v in lo], "span": [round(float(v), 5) for v in span],
            "fit": [round(v, 5) for v in fit]}
    return qx, qy, meta


def depth_layout(cl: np.ndarray, cluster_order: list[int], mx: np.ndarray, my: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x = effective position (with jitter inside its column); y = the cluster's band in family/category order, and
    inside the band the neuron's rank along the meaning map's main axis."""
    pos = np.repeat(np.arange(N_POS), F).astype(np.float64)
    rng = np.random.default_rng(0)
    x = pos + 0.12 + 0.76 * rng.random(N)
    sizes = np.bincount(cl.astype(np.int64), minlength=len(cluster_order))
    y = np.zeros(N)
    start = 0.0
    axis = mx + 0.25 * my
    for k in cluster_order:
        idx = np.where(cl == k)[0]
        h = float(sizes[k])
        r = np.argsort(np.argsort(axis[idx])) / max(1, len(idx) - 1) if len(idx) > 1 else np.zeros(len(idx))
        y[idx] = start + 0.06 * h + 0.88 * h * r
        start += h
    return x, -y                                   # first family at the top


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map-v1", type=Path, required=True)
    ap.add_argument("--map-v2", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--binding", type=Path, required=True)
    ap.add_argument("--sets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tax = json.loads((a.map_v1 / "taxonomy.json").read_text())
    draft = json.loads((a.map_v1 / "compendium-draft.json").read_text())
    known = json.loads((a.map_v1 / "known_v1.json").read_text())
    if {f["family"] for f in tax["families"]} != set(FAM_ORDER):
        raise SystemExit(f"taxonomy families changed: {[f['family'] for f in tax['families']]}")
    cats, c2cat, c2fam = [], {}, {}
    for f in sorted(tax["families"], key=lambda f: FAM_ORDER.index(f["family"])):
        for c in f["categories"]:
            ci = len(cats)
            cats.append({"name": c["category"], "family": FAM_ORDER.index(f["family"]), "desc": c["description"],
                         "n": c["n_neurons"], "high": round(c["share_high_agreement"], 3)})
            for k in c["clusters"]:
                c2cat[k], c2fam[k] = ci, FAM_ORDER.index(f["family"])

    x1, y1, cl1 = read_map(a.map_v1 / "neurons_v1.jsonl")
    x2, y2, cl2 = read_map(a.map_v2 / "neurons_v1.jsonl")
    n_cl = int(cl1.max()) + 1
    missing = [k for k in range(n_cl) if k not in c2cat]
    if missing:
        raise SystemExit(f"clusters without a category: {missing[:10]}")
    fam = np.array([c2fam[int(c)] for c in cl1], np.uint8)

    # layouts
    layouts: dict[str, tuple[np.ndarray, np.ndarray]] = {"meaning": (x1, y1), "masked": (x2, y2)}
    dc = {c["cluster"]: c for c in draft["clusters"]}
    meaning_x = np.array([np.median(x1[cl1 == k]) for k in range(n_cl)])
    order = sorted(range(n_cl), key=lambda k: (c2fam[k], c2cat[k], meaning_x[k]))
    layouts["depth"] = depth_layout(cl1, order, x1, y1)
    if a.weights:
        w = np.load(a.weights)
        for key, name in WEIGHT_LAYOUTS.items():
            if key in w.files:
                layouts[name] = (w[key][:, 0].astype(np.float32), w[key][:, 1].astype(np.float32))
    lay_meta = {}
    for name, (lx, ly) in layouts.items():
        qx, qy, meta = quantise(lx, ly, keep_aspect=name != "depth")
        (a.out / f"layout-{name}.bin").write_bytes(qx.astype("<u2").tobytes() + qy.astype("<u2").tobytes())
        lay_meta[name] = {"file": f"layout-{name}.bin", **meta}
    weight_stats = {}
    if a.weights and a.weights.with_suffix(".json").exists():
        wj = json.loads(a.weights.with_suffix(".json").read_text())
        weight_stats = {k: wj[k] for k in ("write_raw_explained", "read_raw_explained", "write_depth_share",
                                           "read_depth_share", "unique") if k in wj}

    # binding attribution: signed sqrt scale to i8, per family; "all" = mean of the eight at unit std
    bm = np.load(a.binding)
    cols: dict[str, np.ndarray] = {"fam": fam, "cluster": cl1}
    scales = {}
    std_sum = np.zeros(N, np.float64)
    for f in BIND_FAMS:
        v = bm[f"mean_{f}"].astype(np.float64).reshape(-1)
        std_sum += v / v.std()
        if f in SHOWN_BIND:
            cols[f"bind_{f}"], scales[f] = q8(v)
    cols["bind_all"], scales["all"] = q8(std_sum / len(BIND_FAMS))
    sets = json.loads(a.sets.read_text())
    bits = np.zeros(N, np.uint8)
    for b, name in enumerate(SET_BITS):
        for L, n in sets[name]:
            bits[L * F + n] |= 1 << b
    cols["sets"] = bits

    blob, col_meta, off = bytearray(), {}, 0
    for name, arr in cols.items():
        dt = {np.dtype(np.uint8): "u8", np.dtype(np.int16): "i16", np.dtype(np.int8): "i8"}[arr.dtype]
        raw = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
        while off % 2:                              # i16 columns start on an even offset
            blob.append(0)
            off += 1
        col_meta[name] = {"type": dt, "offset": off}
        blob += raw
        off += len(raw)
    (a.out / "neurons.bin").write_bytes(bytes(blob))

    labels = [[n["position"] * F + n["neuron"], n["label"], n.get("pole") or "", n["confidence"]]
              for n in draft["neurons"] if n.get("confidence") in ("high", "medium") and n.get("label")]
    clusters = []
    for k in range(n_cl):
        c = dc[k]
        clusters.append({"id": k, "name": c["name"], "desc": c["description"], "kind": c["kind"],
                         "agree": c["view_agreement"], "themes": c.get("themes") or [], "cat": c2cat[k],
                         "fam": c2fam[k], "size": int((cl1 == k).sum())})
    manifest = {
        "n": N, "positions": N_POS, "per_position": F, "families": FAM_ORDER, "categories": cats, "clusters": clusters,
        "clusters_masked": int(cl2.max()) + 1, "summary": tax["summary"], "layouts": lay_meta, "weights": weight_stats,
        "columns": col_meta, "bind_scales": scales, "set_bits": SET_BITS, "labels": labels,
        "known": [{"id": k["id"], "i": k["L"] * F + k["n"], "name": k["name"], "cluster": k["cluster"],
                   "with": k["shares_cluster_with"]} for k in known],
    }
    (a.out / "neurons.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {a.out}: {len(layouts)} layouts {sorted(layouts)}; {len(labels)} labels; neurons.bin {len(blob)/1e6:.2f} MB",
          file=sys.stderr)


def q8(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Signed sqrt quantisation to int8: q = 127 · sign(v) · sqrt(|v| / m), m = the 99.9th percentile of |v|."""
    m = float(np.percentile(np.abs(v), 99.9)) or 1.0
    q = np.clip(np.round(127 * np.sign(v) * np.sqrt(np.abs(v) / m)), -127, 127).astype(np.int8)
    return q, m


if __name__ == "__main__":
    main()
