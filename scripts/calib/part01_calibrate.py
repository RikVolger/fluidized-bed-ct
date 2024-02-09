import numpy as np
from pathlib import Path
from cate import astra as cate_astra
from scripts.calib.util import *
from scripts.settings import *

detector = cate_astra.Detector(
    DETECTOR_ROWS, DETECTOR_COLS, DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT
)


""" 1. Choose a directory, and find the range of motion in the projections."""
DATA_DIR = R"D:\XRay\2023-11-21 Rik"
MAIN_DIR = "preprocessed_Alignment_5 (needles)"
PROJS_PATH = f"{DATA_DIR}/{MAIN_DIR}"
POSTFIX = f"{MAIN_DIR}_calibrated_on_06feb2024"  # set this value

if MAIN_DIR == "pre_proc_Calibration_needle_phantom_30degsec_table474mm":
    # first frame before motion
    # 31-32 shows a very tiny bit of motion, but seems insignificant
    proj_start = 33
    # final state frame, img 806 equals 32, so the range should be without 806
    proj_end = 806
    nr_projs = proj_end - proj_start  # 773
    x = 50
    t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]
    ignore_cols = 0  # det width used is 550
elif MAIN_DIR == "pre_proc_Calibration_needle_phantom_30degsec_table534mm":
    # first frame before motion
    # 31-32 shows a very tiny bit of motion, but seems insignificant
    proj_start = 39
    proj_end = 813  # or 814, depending on who you ask
    nr_projs = proj_end - proj_start  # 774
    x = 50
    t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]
    ignore_cols = 0  # det width used is 550
elif MAIN_DIR == "pre_proc_VROI500_1000_Cal_20degsec":
    proj_start = 45
    proj_end = 1400
    t_annotated = [50, 501, 953]
    nr_projs = proj_end - proj_start
elif MAIN_DIR == "preprocessed_Alignment_5 (needles)":
    proj_start = 35
    proj_end = 1616
    nr_projs = proj_end - proj_start
    x = 50  # safety margin for start
    t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]
else:
    raise Exception()
for t in t_annotated:
    assert proj_start <= t < proj_end, f"{t} is not within proj start-end."


""" 2. Annotate the projections, for a description of markers, see `util.py`"""
res_path = Path(__file__).parent / "resources"
multicam_data = annotated_data(
    PROJS_PATH,
    t_annotated,
    fname=MAIN_DIR,
    resource_path=res_path,
    cameras=[1, 2, 3],
    open_annotator=False,  # set to `True` if images have not been annotated
    vmin=6.0,
    vmax=10.0,
)
cate_astra.pixels2coords(multicam_data, detector)  # convert to physical coords


""" 3. Set up a multi-camera geometry, where sources, detectors and angles are
the unknowns."""
pre_geoms = triangle_geom(SOURCE_RADIUS, DETECTOR_RADIUS, 
                          rotation=False, shift=False)
srcs = [g.source for g in pre_geoms]
dets = [g.detector for g in pre_geoms]
angles = (np.array(t_annotated) - proj_start) / nr_projs * 2 * np.pi
multicam_geom = triple_camera_circular_geometry(
    srcs, dets, angles=angles, optimize_rotation=True)


""" 4. Perform the optimization """
multicam_geom_flat = [g for c in multicam_geom for g in c]
multicam_data_flat = [d for c in multicam_data.values() for d in c]
markers = marker_optimization(
    multicam_geom_flat,
    multicam_data_flat,
    plot=True,
    max_nfev=10,
    nr_iters=2
)
np.save(f"{res_path}/markers_{POSTFIX}.npy", markers)

# calib (export format)
rotation_0_geoms = {}
for key, val in zip(multicam_data.keys(), multicam_geom):
    rotation_0_geoms[key] = val[0]._g.asstatic()
np.save(f"{res_path}/geom_{POSTFIX}.npy", [rotation_0_geoms])
np.save(f"{res_path}/multicam_geom_{POSTFIX}.npy", multicam_geom)
print("Optimalization results saved.")