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


def create_avg_scan(root, conc, gasflow, framerate):
    if not conc['special']:
        t_avg_dir = concpathbuilder(root, conc['c'], gasflow, framerate)
    elif conc['special'] == "Empty":
        t_avg_dir = emptypathbuilder(root, framerate)
    elif conc['special'] == "Full":
        t_avg_dir = fullpathbuilder(root, conc['c'], framerate)
    t_avg = AveragedScan(
            f"{conc['c']}_{gasflow}_{framerate}",
            DETECTOR,
            str(t_avg_dir),
            proj_start=conc['start'],
            proj_end=conc['end'],
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
                initialization=init
                # col_mask=True,
                )
    x = recon.volume(vol_id)
    recon.clear()
    return loss, x


"""1. Configuration of set-up and calibration"""
# load scans.yaml
with open("./scatter_scans_reco.yaml") as scans_yaml:
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
COLUMN_ID = scans['column_inner_diameter']      # Maybe rename, 'ID' can be confusing.
COLUMN_OD = scans['column_outer_diameter']
RECON_VOLUMES = scans['recon_volumes']
VOXEL_SIZES = scans['voxel_sizes']
MASK_SIZES = scans['mask_sizes']
CORRECT_BEAM_HARDENING = scans['beam_hardening_correction']
if any(CORRECT_BEAM_HARDENING):
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


# Iterate over measurement folders
for series in scans['measurements']:
    # extract root
    root = series['root']
    # TODO Catch file not found errors, write to log file and continue.
    iterable_product = itertools.product(
        series['concentrations'],
        scans['flowrates'],
        scans['framerates'],
        CORRECT_BEAM_HARDENING)
    for conc, gasflow, framerate, bhc in iterable_product:
        empty = create_empty_scan(root, framerate)
        full = create_full_scan(root, conc['c'], framerate)
        t_avg = create_avg_scan(root, conc, gasflow, framerate)

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
            darks_ran=range(10, 200),
            darks_path="D:\\XRay\\2024-06-13 Rik\\preprocessed_Dark_22Hz",
            ref_full=ref.is_full,
            density_factor=t_avg.density_factor,
            col_inner_diameter=t_avg.col_inner_diameter,
            # scatter_mean_full=600,
            # scatter_mean_empty=500,
            averaged=True,
            correct_beam_hardening=bhc
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
                    bhc=bhc,
                    recon_size=recon_size,
                    voxel_size=voxel_size,
                    mask_size=mask_size)

                if not conc['special']:
                    full_path = concpathbuilder(root, conc['c'], gasflow, framerate) / filename
                elif conc['special'] == "Empty":
                    full_path = emptypathbuilder(root, framerate) / filename
                elif conc['special'] == "Full":
                    full_path = fullpathbuilder(root, conc['c'], framerate) / filename

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
