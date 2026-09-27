# RelationVGGT project page

Static page for GitHub Pages. No build step.

## Deploy
1. Put `index.html` and the `static/` folder at the root of a repo (e.g. `relationvggt.github.io` or a `gh-pages` branch).
2. Settings → Pages → deploy from branch.

## Updating links
Open `index.html` and edit the `LINKS` object near the bottom:

    var LINKS = { paper: "", arxiv: "", code: "", benchmark: "" };

An empty string shows the button as "soon"; a URL turns it into a live button.

## Images
All figures live in `static/images/`. Replace a file with the same name to update it.

## Light / dark mode
The page follows the visitor's system setting by default. The round button in the top-right corner switches modes, and the choice is remembered in the browser.

## 3D point cloud view
Under the reference view, "3D point cloud visualization" opens an interactive viewer
(drag to rotate, scroll or pinch to zoom, right-drag to pan, double-click to reset).
Color modes: Ours (predicted target), Relevance, RGB. The view starts from the reference camera.

Build the eight point clouds from the original visualization exports in one step:

    python tools/build_pointclouds.py /path/to/projpage_pointclouds --max-points 120000

This reads `<example>/rgb.ply` (x y z, rgb, relevance, is_subject, is_pred, is_gt), converts
OpenCV axes to y-up, spreads points evenly with a voxel grid (subject kept at similar density),
and writes `static/pointclouds/<slug>.ply` (about 19 bytes per point, ~2.3 MB at 120k).
The folder-to-example mapping is at the top of `tools/build_pointclouds.py`.

Examples without a file show a placeholder scene labelled "Placeholder scene, not a model output".
Browsers block `fetch` on `file://`, so preview with a local server (`python -m http.server`).

## Single-file version
`python tools/build_single_file.py out.html [--pc-dir DIR]` inlines all images and point clouds into one HTML file.
For claude.ai artifacts (16 MB limit), build lighter point clouds first, e.g.
`python tools/build_pointclouds.py ROOT --max-points 60000 --out pc60`, then pass `--pc-dir pc60`.
