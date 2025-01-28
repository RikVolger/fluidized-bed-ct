import matplotlib.pyplot as plt
import pyqtgraph as pq

import cate.astra as cate_astra
from cate.util import plot_projected_markers
from fbrct.reco import KernelKitReconstruction
from scripts.calib.util import *
from scripts.settings import *

detector = cate_astra.Detector(DETECTOR_ROWS, DETECTOR_COLS,
                               DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)

# directory of the calibration scan
DATA_DIR_CALIB =  R"D:/XRay/2024-11-14 Rik en Sam" #Path(r"D:\XRay\2024-11-14 Rik en Sam\preprocessed_Rotation_needles_5degps_again\calibration")#"/run/media/adriaan/Elements/ownCloud_Sophia_SBI/VROI500_1000/"
MAIN_DIR_CALIB = "preprocessed_Rotation_needles_5degps_again/calibration"

# directory of a sbi_scan to reconstruct (can be different or same to calib)
DATA_DIR = R"D:/XRay/2024-11-14 Rik en Sam" #Path(r"D:\XRay\2024-11-14 Rik en Sam\preprocessed_Rotation_needles_5degps_again\calibration")
    #"/run/media/adriaan/Elements/academic/data/ownCloud_Sophia_SBI/VROI500_1000")
MAIN_DIR = "preprocessed_Rotation_needles_5degps_again/calibration"
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
elif MAIN_DIR == "preprocessed_Rotation_needles_5degps_again/calibration":
    proj_start = 20
    proj_end = 1604
    nr_projs = proj_end - proj_start
    x = 50  # safety margin for start
else:
    raise Exception()

# postfix of stored claibration
POSTFIX = f'{MAIN_DIR_CALIB}_calibrated_on_14janc2025'
POSTFIX = f'preprocessed_Rotation_needles_5degps_again_calibrated_on_14janc2025'

t = [497, 958, 1223]
t_annotated = [497, 958, 1223]

######
proj_start = 27 # needs to be changed (see part01)
proj_end = 1610 # needs to be changed (see part01)
nr_projs = proj_end - proj_start
t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]
print(f't_anootated = {t_annotated}')
#######
# t_annotated = None
# if t_annotated is None:
#     n_annotated = 6
#     t_annotated = [int(x + n * nr_projs / n_annotated) for n in range(n_annotated)]
##############

#base_path = R"D:/XRay/2024-11-14 Rik en Sam/preprocessed_Rotation_needles_5degps_again/calibration"#Path(r'D:\XRay\2024-11-14 Rik en Sam\preprocessed_Rotation_needles_5degps_again\calibration')
# restore calibration
#multicam_geom = np.load(base_path / f'multicam_geom_preprocessed_Rotation_needles_5degps_again_calibrated_on_14janc2025.npy', allow_pickle=True)#f'multicam_geom_{POSTFIX}.npy', allow_pickle=True)
multicam_geom = np.load(f'scripts/calib/multicam_geom_{POSTFIX}.npy', allow_pickle=True) 
geom = np.load( f'scripts\\\calib\\\geom_{POSTFIX}.npy', allow_pickle=True)
markers = np.load( f'scripts\\\calib\\\markers_{POSTFIX}.npy', allow_pickle=True).item()

multicam_data = annotated_data(
    PROJS_PATH,
    t_annotated,
    fname="preprocessed_Rotation_needles_5degps_again",#MAIN_DIR,
    resource_path=PROJS_PATH,#f'scripts/calib/geom_preprocessed_Rotation_needles_5degps_again_calibrated_on_14janc2025.npy',
    cameras=[1, 2, 3],
    open_annotator=False,  # set to `True` if images have not been annotated
    vmin=6.0,
    vmax=10.0,
)
print(multicam_data[1][0])
cate_astra.pixels2coords(multicam_data, detector)  # convert to physical coords

# for cam in (1, 2, 3):
#     annotations = multicam_data[cam]
#     projections = xray.xray_multigeom_project(multicam_geom[cam - 1], markers)
#     for d1, d2 in zip(annotations, projections):
#         plot_projected_markers(d1, d2, det=detector, det_padding=1.2)

for cam in (1, 2, 3):
    g = geom[0][cam]
    annotation = [multicam_data[cam][0]]
    projection = xray.xray_multigeom_project([g], markers)
    for d1, d2 in zip(annotation, projection):
        plot_projected_markers(d1, d2, det=detector, det_padding=1.2)

detector_cropped = cate_astra.crop_detector(detector, 0)
reco = KernelKitReconstruction(PROJS_PATH, detector_cropped.todict())

all_geoms = []
all_projs = []
for cam_id in (1, 2, 3,):
    # geoms_interp = geoms_from_interpolation(
    #     interpolation_geoms=multicam_geom[cam_id - 1],
    #     interpolation_nrs=t,
    #     interpolation_calibration_nrs=t_annotated,
    #     plot=False)
    all_geoms.append(multicam_geom[cam_id - 1][0])
    projs = reco.load_sinogram(t_range=t_annotated[0], cameras=[cam_id])
    # projs = prep_projs(projs)
    all_projs.append(projs)
y = np.asarray([p[0, 0] for p in all_projs])

vectors = np.array([geom2astravec(g, reco.detector) for g in all_geoms])
_, proj_geom = reco.sino_gpu_and_proj_geom(all_projs, vectors)

import kernelkit as kk

vol_geom = kk.resolve_volume_geometry(
    shape=[300, 300, 200],
    voxel_size=0.0259,
    extent_min=[None, None, 0.0259 * -100],
    extent_max=[None, None, 0.0259 * (200 - 100)],
    projection_geometry=proj_geom,
    verbose=True)

plt.subplots(1, 3)
plt.subplot(1, 3, 1)
plt.imshow(y[0])
plt.subplot(1, 3, 2)
plt.imshow(y[1])
plt.subplot(1, 3, 3)
plt.imshow(y[2])
plt.show()


x = kk.fdk(
    projections=y,
    projection_geometry=proj_geom,
    volume_geometry=vol_geom,
)

# vol_id, vol_geom = astra_reco_rotation_singlecamera(
#     reco, all_projs, all_geoms, 'FDK', [100 * 3, 100 * 3, 200 * 3], 0.025 * 2)
print(x.shape)
pq.image(x.transpose(2, 1, 0))
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
