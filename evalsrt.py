"""Compara SOLO tiempos de un SRT generado contra uno de referencia (el idioma puede diferir).

Uso: python evalsrt.py ref.srt hyp.srt [--max SEG]
"""
import re, sys, argparse
import numpy as np

TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def load(path, tmax=None):
    cues = []
    for m in TS.finditer(open(path, "rb").read().decode("utf-8", "replace")):
        g = list(map(int, m.groups()))
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        if tmax is None or s < tmax:
            cues.append((s, min(e, tmax) if tmax else e))
    return cues


def mask(cues, n, res):
    m = np.zeros(n, bool)
    for s, e in cues:
        m[int(s / res):int(e / res)] = True
    return m


def evaluate(ref, hyp, tmax, res=0.01):
    n = int(tmax / res) + 1
    R, H = mask(ref, n, res), mask(hyp, n, res)
    tp = (R & H).sum()
    prec, rec = tp / max(H.sum(), 1), tp / max(R.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    hs = np.array([s for s, _ in hyp]); he = np.array([e for _, e in hyp])
    ds = np.array([np.min(np.abs(hs - s)) for s, _ in ref]) if len(hyp) else np.array([np.inf])
    de = np.array([np.min(np.abs(he - e)) for _, e in ref]) if len(hyp) else np.array([np.inf])
    long = sum(1 for s, e in hyp if e - s > 7)
    return dict(n_ref=len(ref), n_hyp=len(hyp), precision=prec, recall=rec, f1=f1,
                start_med=np.median(ds), start_05=(ds <= 0.5).mean(), start_1=(ds <= 1).mean(),
                end_med=np.median(de), end_1=(de <= 1).mean(), largos_7s=long)


def fmt(name, r):
    return (f"{name:28s} cues={r['n_hyp']:4d}/{r['n_ref']:<4d} F1={r['f1']:.3f} "
            f"(P={r['precision']:.2f} R={r['recall']:.2f}) | inicio: mediana={r['start_med']:.2f}s "
            f"≤0.5s={r['start_05']:.0%} ≤1s={r['start_1']:.0%} | fin: mediana={r['end_med']:.2f}s "
            f"≤1s={r['end_1']:.0%} | >7s={r['largos_7s']}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("hyp", nargs="+")
    ap.add_argument("--max", type=float, default=None, help="evaluar solo hasta este segundo")
    a = ap.parse_args()
    tmax = a.max or max(e for _, e in load(a.ref))
    ref = load(a.ref, tmax)
    for h in a.hyp:
        print(fmt(h.split("/")[-1], evaluate(ref, load(h, tmax), tmax)))


if __name__ == "__main__":
    main()
