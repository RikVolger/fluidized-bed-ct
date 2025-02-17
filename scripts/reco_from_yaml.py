import pickle
import time
import warnings
import yaml
import itertools
import h5py
from pathlib import Path
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt

from fbrct import loader, reco, StaticScan, AveragedScan
from scripts.pathbuilders import concpathbuilder, emptypathbuilder, fullpathbuilder, hdf5_filename, mat_filename, pkl_filename

"""0. Helper functions"""


def create_empty_scan(empty_dir):
    empty = StaticScan(
            "empty",
            DETECTOR,
            str(empty_dir),
            proj_start=FRAMES['empty']['start'],  # TODO
            proj_end=FRAMES['empty']['stop'],  # TODO: set higher to reduce noise levels
            is_full=False,
            is_rotational=False,  # TODO: check, the column should not rotate!
            geometry=CALIBRATION_FILE,
            geometry_scaling_factor=1.0,
        )

    return empty


def create_full_scan(full_dir):
    full = StaticScan(  # example: a full scan that is not rotating
            "full",  # give the scan a name
            DETECTOR,
            str(full_dir),
            proj_start=FRAMES['full']['start'],  # TODO
            proj_end=FRAMES['full']['stop'],  # TODO: set higher for less noise
            is_full=True,
            is_rotational=False,  # TODO: check, the column should not rotate!
            geometry=CALIBRATION_FILE,
            geometry_scaling_factor=1.0,
        )

    return full


def create_avg_scan(scan_dir):
    t_avg = AveragedScan(
            scan_dir.name,
            DETECTOR,
            str(scan_dir),
            proj_start=FRAMES['measurements']['start'],
            proj_end=FRAMES['measurements']['stop'],
            # liter_per_min=None,  # liter per minute (set None to ignore)
            # projs=range(5, 2640),  # TODO: set this to a valid range
            projs_offset={1: 0, 2: 0, 3: 0},
            geometry=CALIBRATION_FILE,
            cameras=(1, 2, 3),
            col_inner_diameter=COLUMN_ID,
        )

    return t_avg


def reconstruct(t_avg, recon, algo, sino_t, niters, voxel_size, recon_size, mask_size, init):
    # Make sure mask is not larger than recon volume
    mask_size = min(recon_size['side'], mask_size)

    recon_box_side = int(np.ceil(recon_size['side'] / voxel_size))
    recon_box_height = int(np.ceil(recon_size['height'] / voxel_size))
    recon_box = (recon_box_side, recon_box_side, recon_box_height)
    proj_id, proj_geom = recon.sino_gpu_and_proj_geom(sino_t, t_avg.geometry())
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
                investigating_loss=INVESTIGATING_LOSS,
                initialization=init,
                # col_mask=True,
                )
    x = recon.volume(vol_id)
    recon.clear()
    return loss, x


def find_path(key, exp_ref, day_ref, global_ref):
    if key in exp_ref.keys():
        path = Path(root, exp_ref[key])
    elif key in day_ref.keys():
        path = Path(root, day_ref[key])
    elif key in global_ref.keys():
        path = Path(global_ref[key])
    else:
        raise Exception(
            f"No {key} path provided for reconstruction of "
            f"{experiment['measured']}"
        )

    return path


def get_ref_paths(exp_ref, day_ref, global_ref):
    """find the most local path for 'empty', 'dark' and 'full' measurements

    Args:
        exp_ref (dict): Dictionary of experiment-specific reference paths
        day_ref (dict): Dictionary of day-specific reference paths
        global_ref (dict): Dictionary of global reference paths

    Returns:
        dict: Most local paths for 'empty', 'full' and 'dark' measurements
    """
    keys = ['empty', 'full', 'dark']
    paths = {}
    for key in keys:
        paths[key] = find_path(key, exp_ref, day_ref, global_ref)

    return paths


def find_refs(src, keys=['empty', 'dark', 'full']):
    """Find reference measurement directories for `keys` in `src`

    Args:
        src (dict): Dictionary that can contain one or more `keys`
        keys (list): Keys to search for in `src`. Defaults to ['empty', 'dark', 'full']

    Returns:
        dict: Dictionary with the `keys` found in `src`
    """
    ref_dirs = {}
    for key in keys:
        if key in src.keys():
            ref_dirs[key] = src[key]
    return ref_dirs


"""1. Configuration of set-up and calibration"""
# load scans.yaml
with open("./sam_scans.yaml") as scans_yaml:
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
FRAMES = scans['frames']
INVESTIGATING_LOSS = scans['investigate_loss']
NITERS = scans['number_of_iterations']
COLUMN_ID = scans['column_inner_diameter']      # Maybe rename, 'ID' can be confusing.
COLUMN_OD = scans['column_outer_diameter']
RECON_VOLUMES = scans['recon_volumes']
VOXEL_SIZES = scans['voxel_sizes']
MASK_SIZES = scans['mask_sizes']
CORRECT_BEAM_HARDENING = scans['beam_hardening_correction']
if CORRECT_BEAM_HARDENING:
    assert 'BHC' in scans.keys()
    BHC = {
        'a': scans['BHC']['a'],
        'b': scans['BHC']['b'],
        'c': scans['BHC']['c']
    }
else:
    BHC = {
        'a': 1.0,
        'b': 1.0,
        'c': 2.0
    }
INITIALIZATION = scans['initialize']

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

# load global 'empty', 'dark' and 'full' measurements, if provided
global_ref = find_refs(scans)

# Iterate over measurement folders
for day in scans['measurements']:
    # extract root
    root = day['root']
    # load day-reference measurements, if provided
    day_ref = find_refs(day)

    # TODO Catch file not found errors, write to log file and continue.
    for experiment in day['reconstructions']:
        # load experiment-reference measurements, if provided
        exp_ref = find_refs(experiment)

        ref_paths = get_ref_paths(exp_ref, day_ref, global_ref)
        exp_path = Path(day['root'], experiment['measured'])

        empty = create_empty_scan(ref_paths['empty'])
        full = create_full_scan(ref_paths['full'])
        # dark = create_dark_scan(ref_paths['dark'])
        t_avg = create_avg_scan(exp_path)

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
            darks_ran=range(FRAMES['dark']['start'], FRAMES['dark']['stop']),
            darks_path=ref_paths['dark'],
            ref_full=ref.is_full,
            density_factor=t_avg.density_factor,
            col_inner_diameter=t_avg.col_inner_diameter,
            # scatter_mean_full=600,
            # scatter_mean_empty=500,
            averaged=True,
            correct_beam_hardening=CORRECT_BEAM_HARDENING
        )

        algo = 'sirt'
        for sino_t in sino:
            for recon_size, voxel_size, mask_size in itertools.product(RECON_VOLUMES, VOXEL_SIZES, MASK_SIZES):
                # Investigating change in recon volume doesn't add much if the mask is kept the same.
                # Mask size is overridden in these cases.
                if len(RECON_VOLUMES) > 1 and len(MASK_SIZES) == 1:
                    mask_size = recon_size['side']
                tic = time.perf_counter()
                loss, x = reconstruct(t_avg, recon, algo, sino_t, NITERS,
                                      voxel_size, recon_size, mask_size, INITIALIZATION)
                toc = time.perf_counter()
                print(f"Reconstructing took {toc-tic:.0f} seconds")
                # save results in cXXX folder
                # dataset = {
                #     'reconstruction': x,
                dataset_attributes = {
                    'frames': timeframes,
                    'volume_side': recon_size['side'],
                    'volume_height': recon_size['height'],
                    'voxel_size': voxel_size,
                    'mask_size': mask_size,
                    'scan_folder': t_avg.projs_dir,
                    'empty_folder': empty.projs_dir,
                    'full_folder': full.projs_dir,
                    'iterations': NITERS,
                    'algorithm': algo,
                    'loss': loss,
                }
                # filename = mat_filename(
                filename = hdf5_filename(
                    loss=INVESTIGATING_LOSS,
                    volume=len(RECON_VOLUMES) > 1,
                    resolution=len(VOXEL_SIZES) > 1,
                    mask=len(MASK_SIZES) > 1,
                    init=INITIALIZATION,
                    bhc=CORRECT_BEAM_HARDENING,
                    recon_size=recon_size,
                    voxel_size=voxel_size,
                    mask_size=mask_size)

                full_path = exp_path / filename

                print(f"Saving {full_path}")
                # scio.savemat(full_path, dataset, do_compression=True)
                # with open(full_path, 'wb+') as pkl_file:
                #     pickle.dump(dataset, pkl_file)

                with h5py.File(full_path, 'w') as h5_file:
                    ds = h5_file.create_dataset('reconstruction', data=x, compression='gzip', compression_opts=9)
                    ds.attrs.update(dataset_attributes)

                if INVESTIGATING_LOSS and PLOTTING:
                    plt.plot(loss)
                    plt.show()
