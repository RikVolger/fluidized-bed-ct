import matplotlib.pyplot as plt
import pyqtgraph as pq

import cate.astra as cate_astra
from cate.util import plot_projected_markers, geoms_from_interpolation
from fbrct.reco import AstraReconstruction
from scripts.calib.util import *
from scripts.settings import *

detector = cate_astra.Detector(DETECTOR_ROWS, DETECTOR_COLS,
                               DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)

CALIB_FOLDER = Path(__file__).parent


# directory of the calibration scan
DATA_DIR_CALIB = r"U:\Xray RPT ChemE\X-ray\Xray_data\2024-05-16 Lisanne\VROI190_1320"
MAIN_DIR_CALIB = "pre_proc_VROI190_1320_needles_10degsec"

# directory of a scan to reconstruct (can be different or same to calib)
DATA_DIR = r"U:\Xray RPT ChemE\X-ray\Xray_data\2024-05-16 Lisanne\VROI190_1320"
MAIN_DIR = "pre_proc_VROI190_1320_needles_10degsec"
PROJS_PATH = f'{DATA_DIR}/{MAIN_DIR}'

# configure which projection range to take
if MAIN_DIR == "pre_proc_VROI190_1320_needles_10degsec":
    proj_start = 37
    proj_end = 1440
    #ref_path = '/home/adriaan/ownCloud3/pre_proc_Full_30degsec_03'
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
POSTFIX = f'{MAIN_DIR_CALIB}_calibrated_on_2march2025'

t = [50, 433, 816]
t_annotated = [50, 433, 816]

#t = [497, 958, 1223]
#t_annotated = [497, 958, 1223]

# restore calibration
multicam_geom = np.load(f'multicam_geom_{POSTFIX}.npy', allow_pickle=True)
#multicam_geom = np.load(f'geom_{POSTFIX}.npy', allow_pickle=True)
markers = np.load(f'markers_{POSTFIX}.npy', allow_pickle=True).item()


res_path = CALIB_FOLDER / "resources"
multicam_data = annotated_data(
    PROJS_PATH,
    t_annotated,
    fname=MAIN_DIR,
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
all_projs = np.concatenate(all_projs, axis=0)

print("Original shape of all_projs:", all_projs.shape)

# Transpose om dimensies te corrigeren
all_projs = np.transpose(all_projs, (1, 0, 2))
print("Transposed shape of all_projs:", all_projs.shape)


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