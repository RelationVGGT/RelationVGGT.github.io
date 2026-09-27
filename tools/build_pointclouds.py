"""Build all explorer point clouds for the page from the original visualization exports.

    python tools/build_pointclouds.py /path/to/projpage_pointclouds --max-points 120000

Reads <root>/<example folder>/rgb.ply for each example below and writes
static/pointclouds/<slug>.ply in the viewer's compact format (about 19 bytes per point).
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_relation_ply import convert  # noqa: E402

EXAMPLES = {
    "hanging_on":    "scannetpp_rel_ver2/a24f64f7fb/pair_2",
    "attached_to":   "scannetpp_rel_ver2/a24f64f7fb/pair_1",
    "resting_on":    "scannetpp_rel_ver2/7831862f02/pair_4",
    "lying_on":      "scannetpp_rel_ver2/25f3b7a318/pair_4",
    "surrounded_by": "replica_rel/replica_office2/pair_3",
    "inserted_into": "replica_rel/replica_office0/pair_1",
    "inside":        "lerf_rel_easy/waldo_kitchen/pair_1",
    "on":            "lerf_rel_easy/ramen/pair_5",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "pointclouds"))
    ap.add_argument("--max-points", type=int, default=120000)
    ap.add_argument("--file", default="rgb.ply")
    a = ap.parse_args()
    for slug, rel in EXAMPLES.items():
        src = os.path.join(a.root, rel, a.file)
        if not os.path.exists(src):
            print(f"skip {slug}: {src} not found")
            continue
        dst = os.path.join(a.out, slug + ".ply")
        n, ns = convert(src, dst, a.max_points)
        print(f"{slug:14s} {n:>8,} points ({ns:,} subject)  {os.path.getsize(dst)/1e6:5.2f} MB")
