import numpy as np
from pathlib import Path
from cate import astra as cate_astra
import cate.xray as xray
import yaml
import warnings
from cate.util import plot_projected_markers
from scripts.calib.util import (
    annotated_data,
    triangle_geom,
    triple_camera_circular_geometry,
    marker_optimization,
    markers_from_leastsquares_intersection)
# TODO find a way to avoid import * statements.


with open("./calib.yaml") as calib_yaml:
    calib = yaml.safe_load(calib_yaml)

# Extract physical setup fromyaml
SOURCE_RADIUS = calib["source_radius"]
DETECTOR_RADIUS = calib["detector_radius"]
DETECTOR_COLS = calib["detector_cols"]
DETECTOR_ROWS = calib["detector_rows"]
DETECTOR_COLS_SPEC = calib["detector_cols_spec"]
DETECTOR_ROWS_SPEC = calib["detector_rows_spec"]
DETECTOR_WIDTH_SPEC = calib["detector_width_spec"]
DETECTOR_HEIGHT_SPEC = calib["detector_height_spec"]
DETECTOR_WIDTH = DETECTOR_WIDTH_SPEC / DETECTOR_COLS_SPEC * DETECTOR_COLS       # cm
DETECTOR_HEIGHT = DETECTOR_HEIGHT_SPEC / DETECTOR_ROWS_SPEC * DETECTOR_ROWS     # cm
DETECTOR_PIXEL_WIDTH = DETECTOR_WIDTH / DETECTOR_COLS
DETECTOR_PIXEL_HEIGHT = DETECTOR_HEIGHT / DETECTOR_ROWS
DETECTOR_PIXEL_SPEC = calib['detector_pixel_spec']
if not DETECTOR_PIXEL_SPEC * 0.99 < DETECTOR_PIXEL_HEIGHT < DETECTOR_PIXEL_SPEC * 1.01:
    warnings.warn(f"\n\nCalculated pixel height ({DETECTOR_PIXEL_HEIGHT:.3e}) has"
                  f" more than 1% deviation with spec ({DETECTOR_PIXEL_SPEC:.3e})\n")
if not DETECTOR_PIXEL_SPEC * 0.99 < DETECTOR_PIXEL_WIDTH < DETECTOR_PIXEL_SPEC * 1.01:
    warnings.warn(f"\n\nCalculated pixel width ({DETECTOR_PIXEL_WIDTH:.3e}) has"
                  f" more than 1% deviation with spec ({DETECTOR_PIXEL_SPEC:.3e})\n")

detector = cate_astra.Detector(
    DETECTOR_ROWS, DETECTOR_COLS, DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT
)

""" 1. Extract calibration folder and settings from yaml."""
root = Path(calib["root"])
calib_folder = calib["calibration_folder"]
PROJS_PATH = root / calib_folder

proj_start = calib["rotation"]["start"]
proj_end = calib["rotation"]["stop"]
nr_projs = proj_end - proj_start
x = calib['frames']['start']
n = calib['frames']['n']
t_annotated = []
for i in range(n):
    t_annotated.append(int(x + i * nr_projs / n))

mirrored = calib["images_mirrored"]

for t in t_annotated:
    assert proj_start <= t < proj_end, f"{t} is not within proj start-end."


""" 2. Annotate the projections, for a description of markers, see `util.py`"""
# [ ] change folder to 00_calib
res_path = Path(root.parent / "calib" / calib_folder)
if not res_path.is_dir():
    res_path.mkdir(
        parents=True,       # Also make _all_ parent folders (no safety checks)
        exist_ok=True)      # Doesn't error if folder already exists
# res_path = Path(__file__).parent / "resources"
multicam_data = annotated_data(
    PROJS_PATH,
    t_annotated,
    fname="needles",
    resource_path=res_path,
    cameras=[1, 2, 3],
    open_annotator=True,    # set to `True` if images have not been annotated
    vmin=6.0,
    vmax=10.0,
)
cate_astra.pixels2coords(multicam_data, detector)  # convert to physical coords


""" 3. Set up a multi-camera geometry, where sources, detectors and angles are
the unknowns."""
pre_geoms = triangle_geom(SOURCE_RADIUS, DETECTOR_RADIUS,
                          rotation=False, shift=False, mirrored=mirrored)
srcs = [g.source for g in pre_geoms]
dets = [g.detector for g in pre_geoms]

if mirrored:
    angles = (np.array(t_annotated) - proj_start) / nr_projs * 2 * np.pi
else:
    angles = 2 * np.pi - ((np.array(t_annotated) - proj_start) / nr_projs * 2 * np.pi)
multicam_geom = triple_camera_circular_geometry(
    srcs, dets, angles=angles, optimize_rotation=True)
# Extract geometries from multicam_geom
multicam_geom_flat = []
for c in multicam_geom:
    for g in c:
        multicam_geom_flat.append(g)
# Extract data from multicam_data
multicam_data_flat = []
for c in multicam_data.values():
    for d in c:
        multicam_data_flat.append(d)

markers = marker_optimization(
    multicam_geom_flat,
    multicam_data_flat,
    plot=False,
    max_nfev=20,
    nr_iters=2
)

for cam in range(1, 4):
    for d1, d2 in zip(multicam_data[cam],
                      xray.xray_multigeom_project(multicam_geom[cam - 1], markers)):
        plot_projected_markers(d1, d2, det=detector, det_padding=1.2)

markers_from_leastsquares_intersection(
    multicam_geom_flat,
    multicam_data_flat,
    optimizable=False,
    plot=True)

np.save(f"{res_path}/markers.npy", markers)

# calib (export format)
rotation_0_geoms = {}
for key, val in zip(multicam_data.keys(), multicam_geom):
    rotation_0_geoms[key] = val[0]._g.asstatic()
np.save(f"{res_path}/geom.npy", [rotation_0_geoms])
np.save(f"{res_path}/multicam_geom.npy", multicam_geom)
print("Optimalization results saved.")
