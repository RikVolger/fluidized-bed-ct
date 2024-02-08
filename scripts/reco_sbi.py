import copy
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pq

from fbrct import loader, reco, Scan, DynamicScan, StaticScan, FluidizedBedScan
from fbrct.reco import AstraReconstruction
from fbrct.util import plot_projs
from fbrct import column_mask

"""1. Configuration of set-up and calibration"""
DETECTOR_COLS = 500  # including ROI
DETECTOR_ROWS = 1548  # including ROI
DETECTOR_COLS_SPEC = 1524  # also those outside ROI
DETECTOR_WIDTH_SPEC = 30.2  # cm, also outside ROI
DETECTOR_HEIGHT = 30.7  # cm, also outside ROI
DETECTOR_WIDTH = DETECTOR_WIDTH_SPEC / DETECTOR_COLS_SPEC * DETECTOR_COLS  # cm
DETECTOR_PIXEL_WIDTH = DETECTOR_WIDTH / DETECTOR_COLS
DETECTOR_PIXEL_HEIGHT = DETECTOR_HEIGHT / DETECTOR_ROWS
DETECTOR = {
    "rows": DETECTOR_ROWS,
    "cols": DETECTOR_COLS,
    "pixel_width": DETECTOR_PIXEL_WIDTH,
    "pixel_height": DETECTOR_PIXEL_HEIGHT,
}
DATA_DIR = Path(
    "U:/Xray RPT ChemE/X-ray/Xray_data/2023-02-10 Sophia SBI")
CALIBRATION_FILE = str(Path(__file__).parent
                       / "calib"
                       / "resources"
                       / "geom_pre_proc_VROI500_1000_Cal_20degsec_calibrated_on_06feb2024.npy")

# / "geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy")


"""2. Configuration of pre-experiment scans. Use `StaticScan` for scans where
the imaged object is not dynamic.
 - select the projections to use with `proj_start` and `proj_end`.  Sometimes
a part of the data is on a rotation table, and a different part is dynamic. Or,
some starting frames are jittered and must be skipped. 
 - set `is_full` when the column filled (this is required for preprocessing).
 - set `is_rotational` if the part between `proj_start` and `proj_end` is
   rotating. This is useful when the object is to be reconstructed from a full
   angle scan.
 - set `geometry` to a calibration file.
 - set `geometry_scaling_factor` if an additional scaling correction has
   been computed.
"""

full = StaticScan(  # example: a full scan that is not rotating
    "full",  # give the scan a name
    DETECTOR,
    str(DATA_DIR / "pre_proc_5cm_VROI500_1000_Full_01"),
    proj_start=10,  # TODO
    proj_end=110,  # TODO: set higher for less noise
    is_full=True,
    is_rotational=False,  # TODO: check, the column should not rotate!
    geometry=CALIBRATION_FILE,
    geometry_scaling_factor=1.0,
)

empty = StaticScan(
    "empty",
    DETECTOR,
    str(DATA_DIR / "pre_proc_5cm_VROI500_1000_Empty"),
    proj_start=10,  # TODO
    proj_end=20,  # TODO: set higher to reduce noise levels
    is_full=False,
    is_rotational=False,  # TODO: check, the column should not rotate!
    geometry=CALIBRATION_FILE,
    geometry_scaling_factor=1.0,
)

"""3. Configuration of fluidized bed scan.
 - set `col_inner_diameter` to have the software estimate a density factor
   of the bed material. This is required to compute gas fractions in the
   reconstruction.
"""
scan = FluidizedBedScan(
    "MF4",
    DETECTOR,
    str(DATA_DIR / "pre_proc_5cm_P18_VROI500_1000_MF4_VO50"),
    liter_per_min=None,  # liter per minute (set None to ignore)
    projs=range(1000, 2000),  # TODO: set this to a valid range
    projs_offset={1: 0, 2: 1, 3: 1},
    geometry=CALIBRATION_FILE,
    cameras=(1, 2, 3),
    col_inner_diameter=5.0,
)
timeframes = [1630, 1446]  # which images to reconstruct
# timeframes = [1658, 1650, 1630, 1446]  # which images to reconstruct

"""4. Select a background reference.
There are basically two options: 
 1. either use a `StaticScan` of a full column. In this case the full column 
    projections can be averaged to get a noiseless estimate.
 2. An alternative is to use the fluidized bed itself. This is only valid if 
    the statistical mode of a distribution of pixel values estimates the 
    background density. Computing the mode can also take a long time.
"""
# option 1:
# ref = full
# ref_reduction = 'median'
ref = copy.deepcopy(scan)
ref_reduction = 'mode'

if isinstance(ref, StaticScan):
    ref_path = ref.projs_dir
    ref_projs = [i for i in range(ref.proj_start, ref.proj_end)]
    ref_rotational = ref.is_rotational
elif isinstance(ref, FluidizedBedScan):
    ref_path = ref.projs_dir
    ref_projs = ref.projs
    if not ref_reduction in ('mode',):
        warnings.warn("For fluidized bed scans, 'mode' is a good"
                      " reduction choice. Another option that might not"
                      " introduce bubble bias is 'min', but this will cause"
                      " a density mismatch.")
    ref_rotational = ref.is_rotational
else:
    ref_projs = []
    ref_path = None
    ref_rotational = False

"""5. Preprocess the sinogram."""
assert np.all(
    [t in loader.projection_numbers(scan.projs_dir) for t in timeframes])
recon = reco.AstraReconstruction(
    scan.projs_dir,
    detector=scan.detector)
# recon = reco.KernelKitReconstruction(
#     scan.projs_dir,
#     detector=scan.detector)

sino = recon.load_sinogram(
    t_range=timeframes,
    t_offsets=scan.projs_offset,
    ref_rotational=ref_rotational,
    ref_reduction=ref_reduction,
    ref_path=ref_path,
    ref_projs=ref_projs,
    empty_path=empty.projs_dir,
    empty_rotational=empty.is_rotational,
    empty_projs=[p for p in range(empty.proj_start, empty.proj_end)],
    # darks_ran=range(10),
    # darks_path=scan.darks_dir,
    ref_full=ref.is_full,
    density_factor=scan.density_factor,
    col_inner_diameter=scan.col_inner_diameter,
    scatter_mean_full=600,
    scatter_mean_empty=500,
)

"""6. Perform a SIRT reconstruction (ASTRA Toolbox)"""
for sino_t in sino:  # go through timeframes one by one
    sino_t = np.fliplr(np.transpose(sino_t, [0, 2, 1]))
    plot_projs(sino_t, subplot_row=True)
    # plt.show()

    proj_id, proj_geom = recon.sino_gpu_and_proj_geom(sino_t, scan.geometry())
    vol_id, vol_geom = recon.backward(
        proj_id,
        proj_geom,
        algo='sirt',
        voxels=(200, 200, 800),  # (300, 300, 1500) for better resolution
        voxel_size=5.5 / 200,  # 5.5 cm / 200 voxels
        iters=200,
        min_constraint=0.0,
        max_constraint=1.0,
        col_mask=True)
    x = recon.volume(vol_id)
    recon.clear()

    pq.image(x.T)
    plt.figure()
    plt.imshow(x[:, 100, :])
    plt.show()

# """6. Perform a SIRT reconstruction (experimental - ASTRA KernelKit)"""
# import cupy as cp
# import kernelkit as kk

# for sino_t in sino:  # go through timeframes one by one
#     sino_t = np.fliplr(np.transpose(sino_t, [0, 2, 1]))
#     sino_t[:, :100, :] = 0.
#     sino_t[:, 1400:, :] = 0.
#
#     _, proj_geom = recon.sino_gpu_and_proj_geom(sino_t, scan.geometry())
#
#     # geom_t0 = kk.ProjectionGeometry(
#     #     source_position=[-10.0, 0.0, 0.0],
#     #     detector_position=[20.0, 0.0, 0.0],
#     #     detector=kk.Detector(rows=150, cols=200, pixel_height=0.02,
#     #                          pixel_width=0.01),
#     # )
#     # angles = np.linspace(0, 2 * np.pi, 3, False)
#     # proj_geom = [kk.rotate(geom_t0, yaw=a) for a in angles]
#
#     vol_geom = kk.resolve_volume_geometry(
#         shape=[300, 300, 200],
#         voxel_size=0.0259,
#         extent_min=[None, None, 0.0259*-100],
#         extent_max=[None, None, 0.0259*(200 -100)],
#         projection_geometry=proj_geom,
#         verbose=True)
#
#     vol_geom = kk.resolve_volume_geometry(
#         shape=[300, 300, 200],
#         voxel_size=0.0259,
#         extent_min=[None, None, 0.0259*-100],
#         extent_max=[None, None, 0.0259*(200 -100)],
#         projection_geometry=proj_geom,
#         verbose=True)
#
#
#     # target_is_mean = cp.ones((100, 100, 300), dtype=cp.float32)
#     # target_is_mean[25:75, 25:75, 25:75] = cp.random.random([50] * 3)
#     # projs = kk.fp(target_is_mean, proj_geom, vol_geom)
#
#     plt.figure(1, dpi=150)
#     plt.imshow(sino_t[0], vmin=0., vmax=0.45)
#     # plt.figure(2, dpi=150)
#     # plt.imshow(sino_t[1], vmin=0., vmax=0.45)
#     # plt.figure(3, dpi=150)
#     # plt.imshow(sino_t[2], vmin=0., vmax=0.45)
#     plt.show()
#
#     def callbackf(i, x, y_tmp):
#         for fig in (1, 2, 3,):
#             plt.figure(fig)
#             plt.clf()
#             plt.imshow(y_tmp[fig -1].get(), vmin=0.0, vmax=0.05)
#             plt.colorbar()
#             plt.pause(0.011)
#
#
#     col_mask = column_mask(vol_geom.shape)
#     vol_gpu = kk.sirt(
#         projections=[cp.copy(p) for p in sino_t],
#         projection_geometry=proj_geom,
#         volume_geometry=vol_geom,
#         iters=200,
#         # mask=col_mask,
#         min_constraint=0.0,
#         # max_constraint=1.0,
#         callback=callbackf,
#     )
#
#     pq.image(vol_gpu.T)
#     plt.figure()
#     plt.show()