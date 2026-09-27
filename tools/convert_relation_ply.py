"""Convert RelationVGGT visualization PLYs into the compact format the project page viewer loads.

Input : one PLY per example with per-vertex
        x y z (reference-camera frame, OpenCV axes), red green blue,
        relevance, is_subject, is_pred, is_gt  (conf / view are optional and dropped)
Output: static/pointclouds/<slug>.ply  (binary little endian)
        x y z (float, y up), red green blue (uchar), score (uchar, relevance*255),
        is_subject is_pred is_gt (uchar), and a "comment rv_ref_camera 0 0 0" line so the
        viewer can start from the reference viewpoint.

    python tools/convert_relation_ply.py SRC.ply static/pointclouds/hanging_on.ply --max-points 120000
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shrink_pointclouds import read_vertices, downsample  # noqa: E402


def convert(src, dst, max_points=120000, seed=0):
    _, _, a = read_vertices(src)
    names = a.dtype.names
    xyz = np.stack([np.asarray(a[k], np.float32) for k in "xyz"], 1)
    idx = downsample(a, max_points, seed=seed)   # even spacing; subject thinned at scene density

    p = xyz[idx].copy()
    p[:, 1] *= -1                           # OpenCV (y down, z forward) -> y up, camera looks down -z
    p[:, 2] *= -1
    rel = np.asarray(a["relevance"], np.float32)[idx] if "relevance" in names else np.zeros(len(idx), np.float32)

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
              ("score", "u1"), ("is_subject", "u1"), ("is_pred", "u1"), ("is_gt", "u1")]
    v = np.zeros(len(idx), dtype=fields)
    v["x"], v["y"], v["z"] = p[:, 0], p[:, 1], p[:, 2]
    for c in ("red", "green", "blue"):
        v[c] = np.asarray(a[c])[idx]
    v["score"] = np.clip(np.round(np.nan_to_num(rel) * 255), 0, 255).astype(np.uint8)
    for f in ("is_subject", "is_pred", "is_gt"):
        if f in names:
            v[f] = (np.asarray(a[f])[idx] > 0).astype(np.uint8)

    header = ["ply", "format binary_little_endian 1.0", "comment rv_ref_camera 0 0 0",
              f"element vertex {len(v)}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue",
              "property uchar score", "property uchar is_subject", "property uchar is_pred", "property uchar is_gt",
              "end_header"]
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(v.tobytes())
    return len(v), int(v["is_subject"].sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--max-points", type=int, default=120000)
    a = ap.parse_args()
    n, ns = convert(a.src, a.dst, a.max_points)
    print(f"{a.dst}: {n:,} points ({ns:,} subject), {os.path.getsize(a.dst)/1e6:.2f} MB")
