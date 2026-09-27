"""Bundle index.html + static/ into one self-contained HTML file (images and point clouds inlined).

    python tools/build_single_file.py out.html [--pc-dir DIR]

--pc-dir points at a folder of lighter point clouds for this copy (claude.ai artifacts must
stay under 16 MB; about 60k points per example fits). Defaults to static/pointclouds.
"""
import base64, glob, os, re, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
pc_dir = os.path.join(root, "static", "pointclouds")
if "--pc-dir" in args:
    i = args.index("--pc-dir"); pc_dir = args[i + 1]; del args[i:i + 2]
out = args[0] if args else os.path.join(root, "relationvggt_single_file.html")
html = open(os.path.join(root, "index.html"), encoding="utf-8").read()

def data_uri(m):
    path = os.path.join(root, m.group(0))
    return "data:image/webp;base64," + base64.b64encode(open(path, "rb").read()).decode()
html = re.sub(r"static/images/[a-z0-9_]+\.webp", data_uri, html)

pcs = {}
for f in sorted(glob.glob(os.path.join(pc_dir, "*.ply"))):
    pcs[os.path.splitext(os.path.basename(f))[0]] = base64.b64encode(open(f, "rb").read()).decode()
inline = "window.RV_PC_INLINE = {" + ",".join('"%s":"%s"' % kv for kv in pcs.items()) + "};"
html = html.replace("<body>", "<body>\n<script>" + inline + "</script>", 1)

open(out, "w", encoding="utf-8").write(html)
mb = len(html.encode("utf-8")) / 1e6
print(f"wrote {out}: {mb:.1f} MB, point clouds: {list(pcs) or 'none (placeholder scene)'}")
if mb > 16:
    print("warning: over 16 MB; reduce max_points in export_pointcloud.py")
