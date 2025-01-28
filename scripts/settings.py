import copy
import os
from pathlib import Path
import warnings

from fbrct.scan import (
    StaticScan,
    Phantom,
    FluidizedBedScan,
    TraverseScan,
    MovingPhantom,
)

SOURCE_RADIUS = (71.4+72+70.8)/3+7.1+10 # average distance between source and column + tape measure + radius outer wall column # old value = 94.5
DETECTOR_RADIUS = (15+14.7+15.2)/3+7.1+10 # average distance between detector and column + tape measure + radius outer wall column # old value = 27.0
DETECTOR_COLS = 1548  # including ROI
DETECTOR_ROWS = 1524  # including ROI
DETECTOR_COLS_SPEC = 1548  # also those outside ROI
DETECTOR_ROWS_SPEC = 1524  # also those outside ROI
DETECTOR_WIDTH_SPEC = 30.7   # cm, also outside ROI
DETECTOR_HEIGHT_SPEC = 30.2  # cm, also outside ROI
DETECTOR_WIDTH = DETECTOR_WIDTH_SPEC / DETECTOR_COLS_SPEC * DETECTOR_COLS       # cm
DETECTOR_HEIGHT = DETECTOR_HEIGHT_SPEC / DETECTOR_ROWS_SPEC * DETECTOR_ROWS     # cm
DETECTOR_PIXEL_WIDTH = DETECTOR_WIDTH / DETECTOR_COLS
DETECTOR_PIXEL_HEIGHT = DETECTOR_HEIGHT / DETECTOR_ROWS
DETECTOR_PIXEL_SPEC = 0.0198        # cm
if not DETECTOR_PIXEL_SPEC * 0.99 < DETECTOR_PIXEL_HEIGHT < DETECTOR_PIXEL_SPEC * 1.01:
    warnings.warn(f"\n\nCalculated pixel height ({DETECTOR_PIXEL_HEIGHT:.3e}) has"
                  f" more than 1% deviation with spec ({DETECTOR_PIXEL_SPEC:.3e})\n")
if not DETECTOR_PIXEL_SPEC * 0.99 < DETECTOR_PIXEL_WIDTH < DETECTOR_PIXEL_SPEC * 1.01:
    warnings.warn(f"\n\nCalculated pixel width ({DETECTOR_PIXEL_WIDTH:.3e}) has"
                  f" more than 1% deviation with spec ({DETECTOR_PIXEL_SPEC:.3e})\n")
APPROX_VOXEL_WIDTH = (
    DETECTOR_PIXEL_WIDTH / (SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS)
APPROX_VOXEL_HEIGHT = (
    DETECTOR_PIXEL_HEIGHT / (SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS)
detector = {
    "rows": DETECTOR_ROWS,
    "cols": DETECTOR_COLS,
    "pixel_width": DETECTOR_PIXEL_WIDTH,
    "pixel_height": DETECTOR_PIXEL_HEIGHT,
}

data_dir = str(Path(__file__).parent.parent / "data")

data_dir_scratch2 = os.getenv("DATA_DIR", data_dir)
data_dir_scratch3 = os.getenv("DATA_DIR", data_dir)
data_dir_19 = f"{data_dir_scratch2}/2021-08-19"
data_dir_20 = f"{data_dir_scratch2}/2021-08-20"
data_dir_23 = f"{data_dir_scratch3}/2021-08-23"
data_dir_24 = f"{data_dir_scratch3}/2021-08-24"

calib_dir = str(Path(__file__).parent / "calib")

SCANS = []


def get_scans(name: str) -> list:
    selected = []
    for s in SCANS:
        if s.name == name:
            selected.append(s)

    return selected


###############################################################################
# Scans on 19 aug 2021
###############################################################################
_19_empty_static = StaticScan(
    "2021-08-19_pre_proc_Empty_30degsec",
    detector,
    f"{data_dir_19}/pre_proc_Empty_30degsec",
    proj_start=10,
    proj_end=1026,
    is_full=False,
    is_rotational=False,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)

_19_darks_static = StaticScan(
    "2021-08-19_pre_proc_Darkfield",
    detector,
    f'{data_dir_19}/pre_proc_Dark_frames',
    proj_start=10,  # never take first frames, they are doomed
    proj_end=1000,
    is_full=False,
)

# Note: many of these scans contain jittered/missing frames, reconstructions may
# be incoherent
for lmin in [18, 19, 20, 22, 25]:
    _scan = FluidizedBedScan(
        f"2021-08-19_{lmin}Lmin",
        detector,
        f"{data_dir_19}/pre_proc_{lmin}Lmin",
        liter_per_min=lmin,
        darks=_19_darks_static,
        # geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
        geometry=f"{calib_dir}/geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy",
        geometry_scaling_factor=1.0 / 1.0106333,
        cameras=(1, 2, 3),
        col_inner_diameter=5.0,
        references=[],
        empty=_19_empty_static,
        projs=[t for t in range(100, 1300)]  # TODO: per scan decide
        # density_factor=0.10167654752731324,
    )
    _scan.references.append(copy.deepcopy(_scan))  # other possible reference is own
    SCANS.append(_scan)

###############################################################################
# Scans on 20 aug 2021
###############################################################################
# # looks okay with negligable artifacts, from the 474mm calibration
# scan = StaticScan(
#     "Calibration_needle_phantom_30degsec_table534mm",
#     f'{data_dir_23}/pre_proc_Calibration_needle_phantom_30degsec_table534mm',
#     proj_start=39,
#     proj_end=813,
#     ref_dir=f'{data_dir_23}/pre_proc_Brightfield',
#     # darks_dir=f'{data_dir}/pre_proc_Dark_frames',
#     geometry=f'{calib_dir}/resources/geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
#     ref_full=True
# )
# # SCANS.append(scan)
#
# scan = StaticScan(
#     "Calibration_needle_phantom_30degsec_table474mm",
#     '/home/adriaan/ownCloud2/pre_proc_Calibration_needle_phantom_30degsec_table474mm',
#     proj_start=33,
#     proj_end=806,
#     ref_dir='/home/adriaan/ownCloud3/pre_proc_Brightfield',
#     darks_dir='/home/adriaan/ownCloud3/pre_proc_Dark_frames',
#     geometry='resources/multicam_rot0_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
#     ref_full=True
# )
# # SCANS.append(scan)
#
# scan = StaticScan(
#     "Calibration_needle_phantom_30degsec_table534mm",
#     '/home/adriaan/ownCloud2/pre_proc_Calibration_needle_phantom_30degsec_table534mm',
#     proj_start=39,
#     proj_end=813,
#     ref_dir='/home/adriaan/ownCloud3/pre_proc_Brightfield',
#     darks_dir='/home/adriaan/ownCloud3/pre_proc_Dark_frames',
#     geometry='resources/multicam_rot0_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
#     ref_full=True
# )
# # SCANS.append(scan)
#

for mmsec, ran in [(62, range(100, 200)), (125, range(40, 100))]:
    for pos in ["center", "wall_det2", "wall_source2"]:
        for scaling in ["", "_noscaling"]:
            _scan = TraverseScan(
                f"Small_marble_{pos}_{mmsec}mmsec_65Hz{scaling}",
                detector,
                f"{data_dir_20}/pre_proc_Small_marble_{pos}_{mmsec}mmsec_65Hz",
                motor_velocity=mmsec,
                # darks_dir='/home/adriaan/ownCloud3/pre_proc_Dark_frames',
                # geometry='resources/multicam_rot0_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
                # geometry=calib,
                geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
                geometry_scaling_factor=1.0 / 1.0106333 if scaling != "" else None,
                framerate=65,
                timeframes=range(ran.start - 10, ran.stop + 10),
                references=[_19_empty_static]
                # col_inner_diameter=5.0,
            )
            _scan.add_phantom(MovingPhantom(2.5, interesting_time=ran))
            SCANS.append(_scan)


###############################################################################
# Scans on 23 aug 2021
###############################################################################
_23_full_rotating = StaticScan(
    "2021-08-23_pre_proc_Full_30degsec",
    detector,
    f"{data_dir_23}/pre_proc_Full_30degsec",
    proj_start=1026,
    proj_end=1800,
    is_full=True,
    is_rotational=True,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_23_full_rotating)

_23_full_static = StaticScan(
    "2021-08-23_pre_proc_Full",
    detector,
    f"{data_dir_23}/pre_proc_Full_30degsec",
    proj_start=10,
    proj_end=1050,
    is_full=True,
    is_rotational=False,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_23_full_static)

_23_empty_static = StaticScan(
    "2021-08-23_pre_proc_Empty",
    detector,
    f"{data_dir_23}/pre_proc_Empty_30degsec",
    proj_start=10,
    proj_end=1000,
    is_full=False,
    is_rotational=False,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_23_empty_static)

for size in ["10mm", "14mm", "23mm"]:
    _scan = StaticScan(
        f"2021-08-23_3x{size}_foamballs_vertical_refempty",
        detector,
        f"{data_dir_23}/pre_proc_3x{size}_foamballs_vertical",
        proj_start=1025,
        proj_end=1801,
        # darks_dir=f'{data_dir}/pre_proc_Darkfield',
        geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
        references=[_23_full_rotating],
    )
    SCANS.append(_scan)

# # The 474mm table not-amended geometry looks better than
# #     the amended 3x10mm post-calibration, for the 19lmin scan
for lmin in [19, 20, 22, 25]:
    _scan = FluidizedBedScan(
        f"2021-08-23_{lmin}Lmin",
        detector,
        f"{data_dir_23}/pre_proc_{lmin}Lmin",
        liter_per_min=lmin,
        # darks_dir=f'{data_dir}/pre_proc_Darkfield',
        geometry=f"{calib_dir}/geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy",
        cameras=(1, 2, 3),
        col_inner_diameter=5.0,
        references=[_23_full_static],
        empty=_23_empty_static,
        # density_factor=0.10167654752731324,
    )
    _scan.references.append(copy.deepcopy(_scan))  # other possible reference is own
    SCANS.append(_scan)

###############################################################################
# Scans on 24 aug 2021
###############################################################################
_24_darks = StaticScan(
    "2021-08-24_pre_proc_Dark",
    detector,
    f"{data_dir_24}/pre_proc_Dark",
    proj_start=1,
    proj_end=10,
    is_full=False,
    is_rotational=False,
)
SCANS.append(_24_darks)

_24_empty_rotating = StaticScan(
    "2021-08-24_pre_proc_Empty_30degsec_nonrotating",
    detector,
    f"{data_dir_24}/pre_proc_Empty_30degsec",
    proj_start=1026,
    proj_end=1800,
    is_full=False,
    is_rotational=True,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_24_empty_rotating)

_24_full_rotating = StaticScan(
    "2021-08-24_pre_proc_Full_30degsec_rotating",
    detector,
    f"{data_dir_24}/pre_proc_Full_30degsec",
    proj_start=1026,
    proj_end=1800,
    is_full=True,
    is_rotational=True,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_24_full_rotating)

_24_empty_nonrotating = StaticScan(
    "2021-08-24_pre_proc_Empty_30degsec_nonrotating",
    detector,
    f"{data_dir_24}/pre_proc_Empty_30degsec",
    proj_start=1,
    proj_end=1000,
    is_full=False,
    is_rotational=False,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_24_empty_nonrotating)

# for nr, pos in [(2, "horizontal"), (3, "vertical_wall")]:
#     for size in [10, 14, 23]:
#         _scan = StaticScan(
#             f"2021-08-24_{nr}x{size}mm_foamballs_{pos}_refempty",
#             detector,
#             f"{data_dir_24}/pre_proc_{nr}x{size}mm_foamballs_{pos}",
#             proj_start=1025,
#             proj_end=1799,
#             # darks_dir='/home/adriaan/data/evert/2021-08-23/pre_proc_Darkfield',
#             geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
#             cameras=(1, 2, 3),
#             references=[_24_empty_rotating],
#         )
#         # SCANS.append(scan)

# _scan = StaticScan(
#     "2021-08-24_10mm_14mm_23mm_horizontal_refempty",
#     detector,
#     f"{data_dir_24}/pre_proc_10mm_14mm_23mm_foamballs_horizontal",
#     proj_start=1025,
#     proj_end=1799,
#     # darks_dir=f'{data_dir}/pre_proc_Dark',
#     geometry="resources/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
#     references=[_24_empty_rotating],
# )
# # SCANS.append(scan)

_scan = StaticScan(
    "2021-08-24_10mm_23mm_horizontal",
    detector,
    f"{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal",
    proj_start=1025,
    proj_end=1799,
    # darks_dir=f'{data_dir}/pre_proc_Dark',
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
    references=[_24_full_rotating],
    empty=_24_empty_rotating,
    # darks=_24_darks,
    col_inner_diameter=5.0,
)
SCANS.append(_scan)