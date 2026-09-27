"""Inspect and shrink a folder of PLY point clouds before uploading them.

Only needs numpy. Keeps every vertex property (xyz, colors, score, subject, ...) and the
folder structure; drops faces and other non-vertex elements.

1) Inspect only (fast, no files written):
       python shrink_pointclouds.py projpage_pointclouds --inspect

2) Shrink into a new folder, then zip that folder and upload it.
   --only rgb.ply processes just the rgb.ply in each folder (rgb/relevance/target share
   the same points and properties, so one is enough):
       python shrink_pointclouds.py projpage_pointclouds projpage_small --max-points 120000 --only rgb.ply

Downsampling: the voxel size is searched so that about --max-points occupied voxels remain,
one point is kept per voxel, and random sampling trims the rest. Subject points are thinned at
the same density as the scene (a small subject keeps at least 500 points).
A report.txt with each file's header summary is written next to the output.
"""
import argparse, os, sys
import numpy as np

TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
         "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
         "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
         "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_header(path):
    with open(path, "rb") as f:
        lines, size = [], 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no end_header")
            size += len(line)
            text = line.decode("ascii", "replace").strip()
            lines.append(text)
            if text == "end_header":
                break
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file")
    fmt, elements = None, []
    for l in lines:
        t = l.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            elements.append({"name": t[1], "count": int(t[2]), "props": []})
        elif t[0] == "property" and elements:
            if t[1] == "list":
                elements[-1]["props"].append((t[4], "list"))
            else:
                elements[-1]["props"].append((t[2], t[1]))
    return fmt, elements, size


def read_vertices(path):
    fmt, elements, hsize = read_header(path)
    if not elements or elements[0]["name"] != "vertex":
        raise ValueError("vertex must be the first element")
    v = elements[0]
    if any(t == "list" for _, t in v["props"]):
        raise ValueError("list properties on vertices are not supported")
    if fmt in ("binary_little_endian", "binary_big_endian"):
        end = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(n, end + TYPES[t]) for n, t in v["props"]])
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=hsize, shape=(v["count"] * dt.itemsize,))
        arr = np.frombuffer(mm, dtype=dt, count=v["count"])
    elif fmt == "ascii":
        dt = np.dtype([(n, TYPES[t]) for n, t in v["props"]])
        with open(path, "rb") as f:
            f.seek(hsize)
            raw = np.loadtxt(f, max_rows=v["count"], ndmin=2)
        arr = np.empty(len(raw), dtype=dt)
        for i, (n, _) in enumerate(v["props"]):
            arr[n] = raw[:, i]
    else:
        raise ValueError("unknown format " + str(fmt))
    return fmt, elements, arr


def describe(path, fmt, elements, arr=None):
    mb = os.path.getsize(path) / 1e6
    out = [f"{path}  ({mb:.1f} MB, {fmt})"]
    for e in elements:
        props = ", ".join(f"{n}:{t}" for n, t in e["props"])
        out.append(f"  element {e['name']} x {e['count']:,}: {props}")
    if arr is not None and len(arr):
        names = arr.dtype.names
        if all(k in names for k in ("x", "y", "z")):
            lo = [float(np.nanmin(arr[k])) for k in "xyz"]
            hi = [float(np.nanmax(arr[k])) for k in "xyz"]
            out.append("  bbox min " + " ".join(f"{a:.3f}" for a in lo) + "  max " + " ".join(f"{a:.3f}" for a in hi))
        for n in names:
            if n not in ("x", "y", "z", "nx", "ny", "nz"):
                col = np.asarray(arr[n])
                out.append(f"  {n}: min {col.min():.4g}  max {col.max():.4g}  mean {col.mean():.4g}")
    return "\n".join(out)


def voxel_pick(xyz, n_target, seed=0):
    """About n_target point indices spread evenly in space (one point per voxel).

    The voxel size is found by binary search on the actual number of occupied voxels,
    so it works for surfaces (scans) as well as filled volumes."""
    n = len(xyz)
    if n <= n_target:
        return np.arange(n)
    lo = xyz.min(0)
    ext = float(np.max(np.ptp(xyz, 0))) or 1.0
    small, large = ext / 4000.0, ext / 2.0
    best = None
    for _ in range(24):
        s = np.sqrt(small * large)
        k = np.floor((xyz - lo) / s).astype(np.int64)
        dims = k.max(0) + 1
        key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
        _, first = np.unique(key, return_index=True)
        if len(first) >= n_target:
            best, small = first, s          # enough voxels: try bigger ones
            if len(first) <= n_target * 1.1:
                break
        else:
            large = s                       # too few voxels: go smaller
    idx = np.sort(best) if best is not None else np.arange(n)
    if len(idx) > n_target:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n_target, replace=False))
    return idx


def downsample(arr, max_points, min_subject=500, seed=0):
    """Evenly spread max_points points over the whole scene.

    Subject points are thinned at the same density as everything else, so a large subject
    no longer crowds out the scene. A small subject keeps at least min_subject points."""
    names = arr.dtype.names
    xyz = np.stack([np.asarray(arr[k], np.float64) for k in "xyz"], 1)
    ok = np.flatnonzero(np.isfinite(xyz).all(1))
    if len(ok) <= max_points:
        return ok
    idx = ok[voxel_pick(xyz[ok], max_points, seed)]

    subj_key = next((k for k in ("subject", "is_subject") if k in names), None)
    if subj_key:
        subj = np.asarray(arr[subj_key]) > 0.5
        have = int(subj[idx].sum())
        want = min(int(subj[ok].sum()), min_subject)
        if have < want:
            pool = np.setdiff1d(ok[subj[ok]], idx)
            extra = np.random.default_rng(seed).choice(pool, min(want - have, len(pool)), replace=False)
            idx = np.sort(np.concatenate([idx, extra]))
    return idx


def write_ply(path, arr):
    le = []
    for n in arr.dtype.names:
        base = arr.dtype[n].base
        kind = base.kind + str(base.itemsize)
        if kind == "f8":
            kind = "f4"                       # doubles -> floats: half the size, plenty for display
        le.append((n, "<" + kind))
    out = np.empty(len(arr), dtype=np.dtype(le))
    for n in arr.dtype.names:
        out[n] = arr[n]
    inv = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
           "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(out)}"]
    header += [f"property {inv[out.dtype[n].str[1:]]} {n}" for n in out.dtype.names]
    header.append("end_header")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(out.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--max-points", type=int, default=120000)
    ap.add_argument("--only", default=None, help="only process files with this name, e.g. rgb.ply")
    a = ap.parse_args()
    if not a.inspect and not a.dst:
        sys.exit("give an output folder, or use --inspect")

    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.src) for f in fs
                   if f.lower().endswith(".ply") and (a.only is None or f == a.only))
    if not files:
        sys.exit("no .ply files found under " + a.src)
    report, total_in, total_out = [], 0, 0
    for p in files:
        total_in += os.path.getsize(p)
        try:
            fmt, elements, arr = read_vertices(p)
            if a.inspect:
                report.append(describe(p, fmt, elements, arr))
                print(report[-1], end="\n\n", flush=True)
                continue
            idx = downsample(arr, a.max_points)
            q = os.path.join(a.dst, os.path.relpath(p, a.src))
            write_ply(q, arr[idx])
            total_out += os.path.getsize(q)
            report.append(describe(p, fmt, elements, arr) + f"\n  -> {q}: {len(idx):,} points, {os.path.getsize(q)/1e6:.1f} MB")
        except Exception as e:
            try:
                fmt, elements, _ = read_header(p)
                report.append(describe(p, fmt, elements) + f"\n  !! skipped: {e}")
            except Exception:
                report.append(f"{p}\n  !! skipped: {e}")
        print(report[-1], end="\n\n", flush=True)

    summary = f"{len(files)} files, {total_in/1e6:.1f} MB in"
    if not a.inspect:
        summary += f", {total_out/1e6:.1f} MB out"
        with open(os.path.join(a.dst, "report.txt"), "w") as f:
            f.write("\n\n".join(report) + "\n\n" + summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
