import numpy as np
from pathlib import Path
from cate import astra as cate_astra
import cate.xray as xray
from cate.util import plot_projected_markers
from scripts.calib.util import *
from scripts.settings import *


detector = cate_astra.Detector(
    DETECTOR_ROWS, DETECTOR_COLS, DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT
)


""" 1. Choose a directory, and find the range of motion in the projections."""
DATA_DIR = R"d:\XRay\2024-11-14 Rik en Sam"
MAIN_DIR = "preprocessed_Rotation_needles_5degps_again"
PROJS_PATH = f"{DATA_DIR}/{MAIN_DIR}"
POSTFIX = f"{MAIN_DIR}_calibrated_on_14janc2025"  # set this value

t_annotated = None
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
elif MAIN_DIR == "preprocessed_Rotation_needles_5degps_again":
    proj_start = 27 #20
    proj_end = 1610 #1604
    nr_projs = proj_end - proj_start
    x = 50  # safety margin for start
    t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]
    
else:
    raise Exception()

if t_annotated is None:
    n_annotated = 6
    t_annotated = [int(x + n * nr_projs / n_annotated) for n in range(n_annotated)]

for t in t_annotated:
    assert proj_start <= t < proj_end, f"{t} is not within proj start-end."


""" 2. Annotate the projections, for a description of markers, see `util.py`"""
res_path = Path(PROJS_PATH) / "calibration"
# res_path = Path(__file__).parent / "resources"
multicam_data = annotated_data(
    PROJS_PATH,
    t_annotated,
    fname=MAIN_DIR,
    resource_path=res_path,
    cameras=[1, 2, 3],
    open_annotator=False, #True,  # set to `True` if images have not been annotated
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
angles = 2 * np.pi - ((np.array(t_annotated) - proj_start) / nr_projs * 2 * np.pi)
# angles = (np.array(t_annotated) - proj_start) / nr_projs * 2 * np.pi
multicam_geom = triple_camera_circular_geometry(
    srcs, dets, angles=angles, optimize_rotation=True)
multicam_geom_flat = [g for c in multicam_geom for g in c]
multicam_data_flat = [d for c in multicam_data.values() for d in c]

""" Inspect initial geometry """
markers = markers_from_leastsquares_intersection(
            multicam_geom_flat,
            multicam_data_flat,
            optimizable=False,
            plot=True)

for cam in range(1, 4):
    for d1, d2 in zip(multicam_data[cam],
                      xray.xray_multigeom_project(multicam_geom[cam - 1], markers)):
        plot_projected_markers(d1, d2, det=detector, det_padding=1.2)

for geom in multicam_geom_flat:
    for param in geom.parameters():
        if len(param) == 1:
            if (param.value > 1) or (param.value < -1):
                playroom = abs(param.value * 0.2)
                bounds = [
                    param.value - playroom,
                    param.value + playroom
                ]
                param.bounds = bounds
            else:
                param.bounds = [-1*np.pi, 1*np.pi]
        elif len(param) > 1:
            if (any(param.value > 1)) or (any(param.value < -1)):
                playroom = abs(param.value * 0.2)
                # Give z-coordinate (value_init = 0) 10 cm of play. 
                # Some other params might get extra play too, but 
                # c'est la vie. z-coordinate really needs to play.
                playroom[playroom < 0.2] = 10
                bounds = [
                    param.value - playroom,
                    param.value + playroom
                ]
                if any(bounds[0] > bounds[1]):
                    raise ValueError
                param.bounds = bounds
            else:
                min_bounds = [-1*np.pi] * len(param)
                max_bounds = [ 1*np.pi] * len(param)
                param.bounds = [min_bounds, max_bounds]

""" 4. Perform the optimization """
markers = marker_optimization(
    multicam_geom_flat,
    multicam_data_flat,
    plot=False,
    max_nfev=20,
    nr_iters=4
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

np.save(f"{res_path}/markers_{POSTFIX}.npy", markers)

# calib (export format)
rotation_0_geoms = {}
for key, val in zip(multicam_data.keys(), multicam_geom):
    rotation_0_geoms[key] = val[0]._g.asstatic()
np.save(f"{res_path}/geom_{POSTFIX}.npy", [rotation_0_geoms])
np.save(f"{res_path}/multicam_geom_{POSTFIX}.npy", multicam_geom)
print("Optimalization results saved.")