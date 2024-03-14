import matplotlib.pyplot as plt
import pyqtgraph as pq
import numpy as np
from pathlib import Path

import cate.astra as cate_astra
import cate.xray as xray
from cate.util import geoms_from_interpolation, plot_projected_markers
from fbrct.reco import AstraReconstruction
from scripts.calib.util import *
from scripts.settings import *

detector = cate_astra.Detector(DETECTOR_ROWS, DETECTOR_COLS,
                               DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)

CALIB_FOLDER = Path(__file__).parent

# directory of the calibration scan
DATA_DIR_CALIB = R"D:\XRay\2023-11-21 Rik"
MAIN_DIR_CALIB = "preprocessed_Alignment_5 (needles)"

# directory of a scan to reconstruct (can be different or same to calib)
DATA_DIR = R"D:\XRay\2023-11-21 Rik"
MAIN_DIR = "preprocessed_Alignment_5 (needles)"
PROJS_PATH = f'{DATA_DIR}/{MAIN_DIR}'

# configure which projection range to take
if MAIN_DIR == "pre_proc_3x10mm_foamballs_vertical_01":
    proj_start = 37
    proj_end = 1621
    ref_path = '/home/adriaan/ownCloud3/pre_proc_Full_30degsec_03'
    nr_projs = proj_end - proj_start
elif MAIN_DIR == "pre_proc_Calibration_needle_phantom_30degsec_table474mm":
    proj_start = 39
    proj_end = 813  # or 814, depending on who you ask
    ref_path = '/home/adriaan/ownCloud3/pre_proc_Brightfield'
    nr_projs = proj_end - proj_start
elif MAIN_DIR == "pre_proc_VROI500_1000_Cal_20degsec":
    proj_start = 45
    proj_end = 1400
    t_annotated = [50, 501, 953]
    nr_projs = proj_end - proj_start
    t_range = range(proj_start, proj_start + nr_projs, 6)
elif MAIN_DIR == "preprocessed_Alignment_5 (needles)":
    proj_start = 35
    proj_end = 1616
    nr_projs = proj_end - proj_start
    x = 50  # safety margin for start
    n_annotated = 6
    t_annotated = [int(x + n * nr_projs / n_annotated) for n in range(n_annotated)]
    t_range = range(proj_start, proj_end, 12)
    # t_range = np.linspace(proj_start, proj_end, 1, dtype=int)
else:
    raise Exception()

# postfix of stored claibration
POSTFIX = f'{MAIN_DIR_CALIB}_calibrated_on_06feb2024'

t = t_annotated
# t = [497, 958, 1223]
# t_annotated = [497, 958, 1223]

# restore calibration
multicam_geom = np.load(f'{CALIB_FOLDER}/resources/multicam_geom_{POSTFIX}.npy', allow_pickle=True)
markers = np.load(f'{CALIB_FOLDER}/resources/markers_{POSTFIX}.npy', allow_pickle=True).item()

res_path = CALIB_FOLDER / "resources"
multicam_data = annotated_data(
    PROJS_PATH,
    t_annotated,
    fname=MAIN_DIR_CALIB,
    resource_path=res_path,
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
                               ref_full=False)
    projs = prep_projs(projs)
    all_projs.append(projs)

if len(all_projs[0].shape) < 3:
    all_projs = np.array(all_projs).swapaxes(1, 2)
else:
    all_projs = np.concatenate(all_projs, axis=1).swapaxes(0, 1)

vol_id, vol_geom = astra_reco_rotation_singlecamera(
    reco,
    all_projs,
    all_geoms,
    'sirt',
    [int(1500 / 2), int(1500 / 2), int(1500 / 2)],
    0.02 * 2,
    max_constraint=1.0,
    r=int(24/2/0.04),
    iters=50
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
