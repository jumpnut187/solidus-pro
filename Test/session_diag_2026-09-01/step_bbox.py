import re, sys, glob, os

for path in sorted(glob.glob("/Users/brendonjarett/Downloads/Test 2/*.step")):
    txt = open(path, errors="ignore").read()
    unit = "INCH" if re.search(r"'INCH'", txt, re.I) else ("MM" if re.search(r"SI_UNIT\(.MILLI", txt) else "?")
    pts = [tuple(float(x) for x in m.group(1).split(","))
           for m in re.finditer(r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]+)\)", txt)]
    pts = [p for p in pts if len(p) == 3]
    if not pts:
        print(f"{os.path.basename(path):45s} no points"); continue
    xs, ys, zs = zip(*pts)
    dims = sorted([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)])
    print(f"{os.path.basename(path):45s} unit={unit:4s} dims (sorted) = {dims[0]:8.2f} x {dims[1]:8.2f} x {dims[2]:8.2f}")
