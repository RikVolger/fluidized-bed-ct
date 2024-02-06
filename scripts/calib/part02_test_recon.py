import matplotlib.pyplot as plt
import pyqtgraph as pq

import cate.astra as cate_astra
from cate.util import plot_projected_markers
from fbrct.reco import AstraReconstruction
from scripts.calib.util import *
from scripts.settings import *

detector = cate_astra.Detector(DETECTOR_ROWS, DETECTOR_COLS,
                               DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)

CALIB_FOLDER = Path(__file__).parent

# directory of the calibration scan
DATA_DIR_CALIB = "U:\Xray RPT ChemE\X-ray\Xray_data\\2023-02-10 Sophia SBI"
MAIN_DIR_CALIB = "pre_proc_VROI500_1000_Cal_20degsec"

# directory of a scan to reconstruct (can be different or same to calib)
DATA_DIR = "U:\Xray RPT ChemE\X-ray\Xray_data\\2023-02-10 Sophia SBI"
MAIN_DIR = "pre_proc_VROI500_1000_Cal_20degsec"
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
    nr_projs = 1371  # a guess
else:
    raise Exception()

# postfix of stored claibration
POSTFIX = f'{MAIN_DIR_CALIB}_calibrated_on_13june2023'

t = [497, 958, 1223]
t_annotated = [497, 958, 1223]

# restore calibration
multicam_geom = np.load(f'{CALIB_FOLDER}/multicam_geom_{POSTFIX}.npy', allow_pickle=True)
markers = np.load(f'{CALIB_FOLDER}/markers_{POSTFIX}.npy', allow_pickle=True).item()

res_path = CALIB_FOLDER / "resources"
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

# for d1, d2 in zip(multicam_data[3],
#                   xray.xray_multigeom_project(multicam_geom, markers)):
#     plot_projected_markers(d1, d2, det=detector, det_padding=1.2)
#

detector_cropped = cate_astra.crop_detector(detector, 0)
reco = AstraReconstruction(PROJS_PATH, detector_cropped.todict())

all_geoms = []
all_projs = []
for cam_id in range(1, 2):
    # geoms_interp = geoms_from_interpolation(
    #     interpolation_geoms=multicam_geom[cam_id - 1],
    #     interpolation_nrs=t,
    #     interpolation_calibration_nrs=t_annotated,
    #     plot=False)
    all_geoms.extend(multicam_geom[cam_id - 1])
    projs = reco.load_sinogram(t_range=t, cameras=[cam_id],
                               ref_full=True)
    projs = prep_projs(projs)
    all_projs.append(projs)
all_projs = np.concatenate(all_projs, axis=0).swapaxes(0, 1)

vol_id, vol_geom = astra_reco_rotation_singlecamera(
    reco, all_projs, all_geoms, 'FDK', [100 * 3, 100 * 3, 200 * 3], 0.025 * 2)
x = reco.volume(vol_id)
x = np.transpose(x, (2, 1, 0))
print(x.shape)
pq.image(x)
plt.figure()
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
