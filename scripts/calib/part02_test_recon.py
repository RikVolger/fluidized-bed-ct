import yaml
import warnings
import matplotlib.pyplot as plt
import pyqtgraph as pq
import numpy as np
from pathlib import Path

import cate.astra as cate_astra
import cate.xray as xray
from cate.util import geoms_from_interpolation, plot_projected_markers
from fbrct.reco import AstraReconstruction
from scripts.calib.util import *

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

recon_step = calib["reconstruction"]["step"]
t_range = range(proj_start, proj_end, recon_step)

calib_path = root / "calib" / calib_folder
# restore calibration
multicam_geom = np.load(calib_path / 'multicam_geom.npy', allow_pickle=True)
markers = np.load(calib_path / 'markers.npy', allow_pickle=True).item()

res_path = CALIB_FOLDER / "resources"
multicam_data = annotated_data(
    PROJS_PATH,
    t_annotated,
    fname="needles",
    resource_path=calib_path,
    cameras=[1, 2, 3],
    open_annotator=False,  # set to `True` if images have not been annotated
    vmin=6.0,
    vmax=10.0,
)
cate_astra.pixels2coords(multicam_data, detector)  # convert to physical coords

# for cam in range(1, 4):
#     for d1, d2 in zip(multicam_data[cam],
#                     xray.xray_multigeom_project(multicam_geom[cam - 1], markers)):
#         plot_projected_markers(d1, d2, det=detector, det_padding=1.2)


detector_cropped = cate_astra.crop_detector(detector, 0)
reco = AstraReconstruction(PROJS_PATH, detector_cropped.todict())

all_geoms = []
all_projs = []
for cam_id in range(1, 4):
    geoms_interp = geoms_from_interpolation(
        interpolation_geoms=multicam_geom[cam_id - 1],
        interpolation_nrs=t_range,
        interpolation_calibration_nrs=t_annotated,
        plot=False)
    all_geoms.extend(geoms_interp)

    projs = reco.load_sinogram(t_range=t_range, cameras=[cam_id],
                               ref_full=False) # ref_rotational = True?
    projs = prep_projs(projs)
    all_projs.append(projs)

if len(all_projs[0].shape) < 3:
    all_projs = np.array(all_projs).swapaxes(1, 2)
else:
    all_projs = np.concatenate(all_projs, axis=1).swapaxes(0, 1)

scaling = 1.5
vol_id, vol_geom = astra_reco_rotation_singlecamera(
    reco,
    all_projs,
    all_geoms,
    'fdk',
    [int(1500/scaling), int(1500/scaling), int(1500/scaling)],
    0.016 * scaling,
    max_constraint=1.0,
    r=int(20/2/(0.016*scaling)),
    iters=200
    )
x = reco.volume(vol_id)
x = np.transpose(x, (2, 1, 0))
print(x.shape)
pq.image(x)
plt.figure()
plt.imshow(x[300, :, :])
plt.show()

for res_cam_id in range(1, 4):
    projs_annotated = reco.load_sinogram(
        t_range=t_annotated,
        cameras=[res_cam_id])
    projs_annotated = prep_projs(projs_annotated)
    res = astra_residual(reco,
                         projs_annotated, vol_id, vol_geom,
                         multicam_geom[res_cam_id - 1])
    plot_projections(res, title='res')
    plot_projections(projs_annotated, title='projs')
    plot_projections(astra_project(
        reco, vol_id, vol_geom,
        multicam_geom[res_cam_id - 1]), title='reprojs')
    plt.show()

reco.clear()
