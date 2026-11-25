import sys, csv, math
import numpy as np

NUM = ["wet_cells","fluid_cells","iface_cells","wet_mass","ceiling","max_abs_u",
       "rho_min","rho_max","eps_min","eps_max","neg_f","neg_cells",
       "neg_porous_cells","min_temp","nonfinite","eps_fluid_min",
       "eps_fluid_max","eps_filter_max","eps_filternb_max","u_p99","u_med"]

def load(path):
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    hdr = lines[0].rstrip("\n").split(",")
    n = len(hdr)
    rows = []
    for l in lines[1:]:
        parts = l.rstrip("\n").split(",")
        if len(parts) < n:
            continue
        d = dict(zip(hdr[:-1], parts[:n-1]))
        d[hdr[-1]] = ",".join(parts[n-1:])
        rows.append(d)
    return hdr, rows

def f(r, k):
    v = r.get(k, "")
    try: return float(v)
    except (TypeError, ValueError): return float("nan")

def summarize(path, label):
    hdr, rows = load(path)
    print(f"\n===== {label}  ({path})  {len(rows)} ledger rows, last step "
          f"{rows[-1]['step'] if rows else '-'} =====")
    if not rows: return
    for k in NUM:
        if k not in hdr: continue
        v = np.array([f(r,k) for r in rows])
        v = v[np.isfinite(v)]
        if not v.size:
            print(f"  {k:20s} all non-finite/absent"); continue
        print(f"  {k:20s} first={v[0]:.6g} last={v[-1]:.6g} min={v.min():.6g} "
              f"max={v.max():.6g} p99={np.percentile(v,99):.6g} med={np.median(v):.6g}")
    wm = np.array([f(r,"wet_mass") for r in rows])
    d = np.diff(wm)
    bad = np.nonzero(d < -1e-9)[0]
    print(f"  wet_mass monotonic: {'YES' if bad.size==0 else 'NO'} "
          f"({bad.size} decreases; worst {d.min() if d.size else 0:.4g})")
    if bad.size:
        for i in bad[:8]:
            print(f"     step {rows[i]['step']}->{rows[i+1]['step']}: "
                  f"{wm[i]:.4g} -> {wm[i+1]:.4g}")
    nf = [r for r in rows if f(r,"nonfinite") > 0]
    print(f"  nonfinite rows: {len(nf)}" + (f" first at step {nf[0]['step']}" if nf else ""))
    return hdr, rows

def parse_prof(s, part):
    key = part + "["
    i = s.find(key)
    if i < 0: return {}
    j = s.find("]", i)
    body = s[i+len(key):j]
    out = {}
    for tok in body.split(";"):
        if ":" in tok:
            z, n = tok.split(":")
            out[int(z)] = int(n)
    return out

def riser_check(path, label, riser="riser", below=("cast",)):
    hdr, rows = load(path)
    if "wetz_by_type" not in hdr:
        print(f"\n{label}: no wetz_by_type column"); return
    print(f"\n----- {label}: riser wetting order / contiguity -----")
    firstr = None
    noncontig = []
    above_front = []
    for r in rows:
        s = r["wetz_by_type"]
        rp = parse_prof(s, riser)
        if not rp: continue
        zs = sorted(rp)
        if firstr is None:
            firstr = (r["step"], zs[0], zs[-1], dict(sorted(rp.items())))
        gaps = [z for z in range(zs[0], zs[-1]+1) if z not in rp]
        if gaps:
            noncontig.append((r["step"], zs[0], zs[-1], gaps[:12], len(gaps)))
    if firstr:
        print(f"  riser first wet at step {firstr[0]}: z {firstr[1]}..{firstr[2]}")
        print(f"    profile: {firstr[3]}")
    else:
        print("  riser never wet")
    print(f"  rows with a z-gap inside the wet riser span: {len(noncontig)}")
    for e in noncontig[:6]:
        print(f"    step {e[0]}: span {e[1]}..{e[2]}, {e[4]} empty planes inside, e.g. {e[3]}")
    print("  --- per-part wet z span over the run (first / last row) ---")
    for r in (rows[0], rows[-1]):
        s = r["wetz_by_type"]
        bits = []
        for part in ("cast","riser","sprue","runner","gate","filter","riser_post","chill","core","pour_cup"):
            p = parse_prof(s, part)
            if p: bits.append(f"{part} z{min(p)}-{max(p)} n{sum(p.values())}")
        print(f"    step {r['step']}: " + "; ".join(bits))

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sum":
        for p in sys.argv[2:]:
            summarize(p, p.split("/")[-1])
            riser_check(p, p.split("/")[-1])
    elif mode == "diff":
        a, b = sys.argv[2], sys.argv[3]
        ha, ra = load(a); hb, rb = load(b)
        print(f"A={a} rows={len(ra)}  B={b} rows={len(rb)}")
        common = [k for k in NUM if k in ha and k in hb]
        print(f"{'column':20s} {'A last':>12s} {'B last':>12s} {'A max':>12s} {'B max':>12s}")
        for k in common:
            va = np.array([f(r,k) for r in ra]); vb = np.array([f(r,k) for r in rb])
            va = va[np.isfinite(va)]; vb = vb[np.isfinite(vb)]
            print(f"{k:20s} {va[-1]:12.6g} {vb[-1]:12.6g} "
                  f"{(va.max() if va.size else float('nan')):12.6g} "
                  f"{(vb.max() if vb.size else float('nan')):12.6g}")
        da = {int(r['step']): r for r in ra}; db = {int(r['step']): r for r in rb}
        shared = sorted(set(da) & set(db))
        print(f"\nshared steps: {len(shared)} ({shared[0] if shared else '-'}..{shared[-1] if shared else '-'})")
        for k in ("wet_cells","wet_mass","max_abs_u","eps_max"):
            if k not in common: continue
            diffs = [(s, f(da[s],k), f(db[s],k)) for s in shared]
            rel = [abs(x-y)/max(abs(y),1e-12) for _,x,y in diffs]
            print(f"  {k:14s} identical={all(r<1e-12 for r in rel)} "
                  f"max rel diff={max(rel):.4g}")
        print("\n  sample rows (step, A wet_cells/mass, B wet_cells/mass):")
        for s in shared[::max(1,len(shared)//12)]:
            print(f"   {s:8d}  A {f(da[s],'wet_cells'):7.0f}/{f(da[s],'wet_mass'):9.4g}"
                  f"   B {f(db[s],'wet_cells'):7.0f}/{f(db[s],'wet_mass'):9.4g}")
