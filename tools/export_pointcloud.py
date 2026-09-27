"""Export one RelationVGGT prediction as a point cloud for the project page viewer.

The page looks for  static/pointclouds/<slug>.ply  for each explorer example, where <slug> is
one of: hanging_on, attached_to, resting_on, lying_on, surrounded_by, inserted_into, inside, on.
If a file is missing, the viewer shows a labelled placeholder scene instead.

Output: binary little-endian PLY with, per point,
    float x, y, z        position (y up; OpenCV camera axes are converted by default)
    uchar red, green, blue
    float score          relation relevance S_t(u; q) in [0, 1]  (0 for reference-view points)
    uchar subject        1 for subject points (the subject mask in the reference view), else 0

Typical inputs, all flattened per pixel:
    points  : (..., 3) point map in a shared world frame, e.g. from the Pi3 point decoder
    colors  : (..., 3) RGB of the same pixels, float in [0, 1] or uint8
    score   : (...)    sigmoid relevance for target-view pixels; use 0 for the reference view
    subject : (...)    bool, True on the subject mask pixels of the reference view
    conf    : (...)    optional confidence map used to drop unreliable points

Keep each file under ~80k points so the page stays light (the self-contained published
version embeds every file, and must stay under 16 MB in total).

Example
-------
    import numpy as np
    from export_pointcloud import export_ply
    export_ply("static/pointclouds/hanging_on.ply",
               points=pts, colors=rgb, score=relevance, subject=subject_mask, conf=conf,
               conf_quantile=0.2, max_points=80000)
"""
import numpy as np


def export_ply(path, points, colors, score=None, subject=None, conf=None,
               conf_quantile=0.2, conf_thresh=None, max_points=80000,
               opencv_to_gl=True, seed=0):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(pts)
    col = np.asarray(colors).reshape(-1, 3)
    if col.dtype != np.uint8:
        col = np.clip(np.asarray(col, dtype=np.float32), 0, 1)
        col = (col * 255 + 0.5).astype(np.uint8)
    sc = np.zeros(n, np.float32) if score is None else np.asarray(score, np.float32).reshape(-1)
    sub = np.zeros(n, np.uint8) if subject is None else np.asarray(subject).reshape(-1).astype(np.uint8)
    assert len(col) == len(sc) == len(sub) == n, "all inputs must describe the same pixels"

    keep = np.isfinite(pts).all(1)
    if conf is not None:
        c = np.asarray(conf, np.float32).reshape(-1)
        thr = conf_thresh if conf_thresh is not None else np.quantile(c[keep], conf_quantile)
        keep &= c >= thr
        keep |= sub.astype(bool) & np.isfinite(pts).all(1)   # never drop the subject
    idx = np.flatnonzero(keep)

    if max_points and len(idx) > max_points:
        rng = np.random.default_rng(seed)
        subj_idx = idx[sub[idx] == 1]
        rest = idx[sub[idx] == 0]
        budget = max(max_points - len(subj_idx), 0)
        idx = np.concatenate([subj_idx, rng.choice(rest, size=min(budget, len(rest)), replace=False)])

    pts, col, sc, sub = pts[idx].copy(), col[idx], np.clip(sc[idx], 0, 1), sub[idx]
    if opencv_to_gl:                      # OpenCV (x right, y down, z forward) -> y up, z toward viewer
        pts[:, 1] *= -1
        pts[:, 2] *= -1

    vertex = np.empty(len(idx), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                       ("score", "<f4"), ("subject", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = col[:, 0], col[:, 1], col[:, 2]
    vertex["score"], vertex["subject"] = sc, sub

    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertex)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property float score\nproperty uchar subject\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
    return len(vertex)
