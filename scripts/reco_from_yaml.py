import warnings
import yaml
import itertools
from pathlib import Path
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt

from fbrct import loader, reco, StaticScan, AveragedScan

"""0. Helper functions"""
def pathbuilder(root, parts):
    # return root and then the parts joined by underscores.
    # Ignores None or empty (falsey) entries.
    return Path(root, "_".join([part for part in parts if part]))


def concpathbuilder(root, concentration, flowrate, framerate):
    return pathbuilder(root, ["preprocessed", concentration, flowrate, framerate])


def fullpathbuilder(root, concentration, framerate):
    return pathbuilder(root, ["preprocessed", concentration, "0lmin", framerate])


def emptypathbuilder(root, framerate):
    return pathbuilder(root, ["preprocessed", "Empty", framerate])


def create_empty_scan(root, framerate):
    empty_dir = emptypathbuilder(root, framerate)
    empty = StaticScan(
            "empty",
            DETECTOR,
            str(empty_dir),
            proj_start=10,  # TODO
            proj_end=210,  # TODO: set higher to reduce noise levels
            is_full=False,
            is_rotational=False,  # TODO: check, the column should not rotate!
            geometry=CALIBRATION_FILE,
            geometry_scaling_factor=1.0,
        )

    return empty


def create_full_scan(root, c, framerate):
    full_dir = fullpathbuilder(root, c, framerate)
    full = StaticScan(  # example: a full scan that is not rotating
            "full",  # give the scan a name
            DETECTOR,
            str(full_dir),
            proj_start=30,  # TODO
            proj_end=210,  # TODO: set higher for less noise
            is_full=True,
            is_rotational=False,  # TODO: check, the column should not rotate!
            geometry=CALIBRATION_FILE,
            geometry_scaling_factor=1.0,
        )

    return full


def create_avg_scan(root, c, gasflow, framerate):
    t_avg_dir = concpathbuilder(root, c, gasflow, framerate)
    t_avg = AveragedScan(
            f"{c}_{gasflow}_{framerate}",
            DETECTOR,
            str(t_avg_dir),
            proj_start=10,
            proj_end=2210,
            # liter_per_min=None,  # liter per minute (set None to ignore)
            # projs=range(5, 2640),  # TODO: set this to a valid range
            projs_offset={1: 0, 2: 0, 3: 0},
            geometry=CALIBRATION_FILE,
            cameras=(1, 2, 3),
            col_inner_diameter=COLUMN_ID,
        )

    return t_avg


def reconstruct(t_avg, recon, algo, sino_t, niters, voxel_size, recon_size, mask_size):
    # Make sure mask is not larger than recon volume
    mask_size = min(recon_size['side'], mask_size)

    recon_box_side = int(np.ceil(recon_size['side'] / voxel_size))
    recon_box_height = int(np.ceil(recon_size['height'] / voxel_size))
    recon_box = (recon_box_side, recon_box_side, recon_box_height)
    proj_id, proj_geom = recon.sino_gpu_and_proj_geom(sino_t, t_avg.geometry())
    niters = NITERS
    vol_id, vol_geom, loss = recon.backward(
                proj_id,
                proj_geom,
                algo=algo,
                voxels=recon_box,
                voxel_size=voxel_size,
                iters=niters,
                min_constraint=0.0,
                max_constraint=1.0,
                r=int(mask_size/2/voxel_size),
                investigating_loss=INVESTIGATING_LOSS
                # col_mask=True,
                )
    x = recon.volume(vol_id)
    recon.clear()
    return loss, x


"""1. Configuration of set-up and calibration"""
# load scans.yaml
with open("./recon_volume_scans.yaml") as scans_yaml:
    scans = yaml.safe_load(scans_yaml)

PLOTTING = False

SOURCE_RADIUS = scans['source_radius']
DETECTOR_RADIUS = scans['detector_radius']
DETECTOR_COLS = scans['detector_cols']
DETECTOR_ROWS = scans['detector_rows']
DETECTOR_COLS_SPEC = scans['detector_cols_spec']
DETECTOR_ROWS_SPEC = scans['detector_rows_spec']
DETECTOR_WIDTH_SPEC = scans['detector_width_spec']
DETECTOR_HEIGHT_SPEC = scans['detector_height_spec']
DETECTOR_WIDTH = DETECTOR_WIDTH_SPEC / DETECTOR_COLS_SPEC * DETECTOR_COLS       # cm
DETECTOR_HEIGHT = DETECTOR_HEIGHT_SPEC / DETECTOR_ROWS_SPEC * DETECTOR_ROWS     # cm
DETECTOR_PIXEL_WIDTH = DETECTOR_WIDTH / DETECTOR_COLS
DETECTOR_PIXEL_HEIGHT = DETECTOR_HEIGHT / DETECTOR_ROWS
DETECTOR_PIXEL_SPEC = scans['detector_pixel_spec']
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
DETECTOR = {
    "rows": DETECTOR_ROWS,
    "cols": DETECTOR_COLS,
    "pixel_width": DETECTOR_PIXEL_WIDTH,
    "pixel_height": DETECTOR_PIXEL_HEIGHT,
}
CALIBRATION_FILE = scans['calibration_file']
INVESTIGATING_LOSS = scans['investigate_loss']
NITERS = scans['number_of_iterations']
COLUMN_ID = scans['column_inner_diameter']
COLUMN_OD = scans['column_outer_diameter']
RECON_VOLUMES = scans['recon_volumes']
VOXEL_SIZES = scans['voxel_sizes']
MASK_SIZES = scans['mask_sizes']

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
# Iterate over measurement folders
for series in scans['measurements']:
    # extract root
    root = series['root']
    # TODO Catch file not found errors, write to log file and continue.
    for c, gasflow, framerate in itertools.product(series['concentrations'], scans['flowrates'], scans['framerates']):
        empty = create_empty_scan(root, framerate)
        full = create_full_scan(root, c, framerate)
        t_avg = create_avg_scan(root, c, gasflow, framerate)

        timeframes = range(t_avg.proj_start, t_avg.proj_end)

        ref = full
        ref_reduction = 'median'
        ref_path = ref.projs_dir
        ref_projs = [i for i in range(ref.proj_start, ref.proj_end)]
        ref_rotational = ref.is_rotational

        # reconstruction steps
        assert np.all(
            [t in loader.projection_numbers(t_avg.projs_dir) for t in timeframes])
        recon = reco.AstraReconstruction(
            t_avg.projs_dir,
            detector=t_avg.detector)

        sino = recon.load_sinogram(
            t_range=timeframes,
            t_offsets=t_avg.projs_offset,
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
            density_factor=t_avg.density_factor,
            col_inner_diameter=t_avg.col_inner_diameter,
            # scatter_mean_full=600,
            # scatter_mean_empty=500,
            averaged=True
        )
        algo = 'sirt'
        niters = 200
        for sino_t in sino:
            for recon_size, voxel_size, mask_size in itertools.product(RECON_VOLUMES, VOXEL_SIZES, MASK_SIZES):
                loss, x = reconstruct(t_avg, recon, algo, sino_t, niters, voxel_size, recon_size, mask_size)
                # save results in cXXX folder
                dataset = {
                    'reconstruction': x,
                    'frames': timeframes,
                    'volume': recon_size,
                    'voxel_size': voxel_size,
                    'mask_size': mask_size,
                    'scan_folder': t_avg.projs_dir,
                    'empty_folder': empty.projs_dir,
                    'full_folder': full.projs_dir,
                    'iterations': niters,
                    'algorithm': algo,
                    'loss': loss,
                }
                filenameparts = ["recon"]
                if INVESTIGATING_LOSS:
                    filenameparts += ["loss"]
                if len(RECON_VOLUMES) > 1:
                    filenameparts += [f"volume-{recon_size['side']}x{recon_size['height']}"]
                if len(VOXEL_SIZES) > 1:
                    filenameparts += [f"resolution-{voxel_size}cm"]
                if len(MASK_SIZES) > 1:
                    filenameparts += [f"mask-{mask_size}cm"]

                full_path = concpathbuilder(root, c, gasflow, framerate) / "_".join(filenameparts) + ".mat"
                scio.savemat(full_path, dataset, appendmat=True, do_compression=True)

                if INVESTIGATING_LOSS and PLOTTING:
                    plt.plot(loss)
                    plt.show()
