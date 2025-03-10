import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import warnings

import scripts.settings as sett
from fbrct import TraverseScan, loader, reco, Scan, DynamicScan, StaticScan, \
    FluidizedBedScan
from fbrct.reco import Reconstruction
from fbrct.util import plot_projs


def _plot(x):
    import pyqtgraph as pq
    import matplotlib.pyplot as plt

    pq.image(x)
    plt.figure()
    plt.imshow(x[..., x.shape[2] // 2])
    plt.show()


def reconstruct(
    scan: Scan,
    recodir: str,
    ref: Scan = None,
    voxels: tuple = None,
    voxel_size: float = None,
    plot: bool = False,
    iters: int = 200,
    locking=False,
    overwrite=False,
    save=False,
    save_mat=False,
    algo="sirt",
    timeframes=None,
    detector_rows: range = None,
    ref_reduction: str = None,
    angles=None
):
    print(f"Next up is {scan}...")

    reconstructor = reco.AstraReconstruction(
        scan.projs_dir,
        detector=scan.detector)

    if isinstance(scan, DynamicScan):
        if scan.projs is None:
            scan_projs = loader.projection_numbers(scan.projs_dir)
        else:
            scan_projs = scan.projs

        if timeframes is not None:
            if not np.all([s in scan_projs for s in timeframes]):
                raise ValueError(
                    "One or more timeframes are not in the set of"
                    " the scan's defined projections."
                )
        else:
            timeframes = scan_projs

        projs = _sino_dynamic(
            scan, reconstructor, ref,
            timeframes=timeframes, detector_rows=detector_rows,
            ref_reduction=ref_reduction
        )
        # per-frame reconstruction:
        for t, sino_t in zip(timeframes, projs):
            if plot:
                from fbrct.plotting import plt, CM
                if detector_rows is not None:
                    ids = slice(detector_rows.start, detector_rows.stop)
                else:
                    ids = slice(None, None)
                plot_projs(
                    projs[0, :, ids],
                    subplot_row=True,
                    figsize=(9 * CM, 7.0 * CM))
                plt.savefig("projs.pdf")
                plt.pause(10.)
                plt.show()

            _inner_reco(
                scan,
                reconstructor,
                recodir,
                algo,
                sino_t,
                scan.geometry(),
                voxels,
                voxel_size,
                iters,
                t=t,
                save=save,
                save_mat=save_mat,
                locking=locking,
                overwrite=overwrite,
                plot=plot,
            )

        reconstructor.clear()
    elif isinstance(scan, StaticScan):
        projs, geoms = _sino_static(
            scan, reconstructor, ref,
            angles=angles,
            detector_rows=detector_rows,
        )
        if plot:
            from fbrct.plotting import plt, CM
            plot_projs(
                projs[:,
                detector_rows] if detector_rows is not None else projs,
                figsize=(6 * CM, 7.0 * CM),
                pause=1.)

        _inner_reco(
            scan,
            reconstructor,
            recodir,
            algo,
            projs,
            geoms,
            voxels,
            voxel_size,
            iters,
            locking=locking,
            overwrite=overwrite,
            save=save,
            plot=plot,
        )
    else:
        raise ValueError()

    reconstructor.clear()


def _sino_dynamic(
    scan: Scan,
    reco: Reconstruction,
    ref: Scan,
    ref_reduction: str = None,
    timeframes=None,
    **kwargs
):
    if isinstance(ref, StaticScan):
        if issubclass(type(scan), FluidizedBedScan):
            assert (
                not ref.is_rotational
            ), "Scan is fluidized bed, but reference is rotational?"
        else:
            if not issubclass(type(scan), TraverseScan):
                assert ref.is_rotational, "Scan is dynamic, but reference is static?"

        ref_path = ref.projs_dir
        ref_full = ref.is_full
        ref_projs = [i for i in range(ref.proj_start, ref.proj_stop)]
        ref_rotational = ref.is_rotational
        if ref_reduction is None:
            ref_reduction = 'mean'
        if not ref_reduction in ('mean', 'median'):
            warnings.warn("For static scans, 'mean' or 'medium' are good"
                          " reduction choices.")
    elif isinstance(ref, FluidizedBedScan):
        if issubclass(type(scan), FluidizedBedScan):
            if ref.liter_per_min != scan.liter_per_min:
                warnings.warn("Using a reference with a different l/min then "
                              "that is used in the experiment. This is not "
                              "optimal.")
        ref_path = ref.projs_dir
        ref_full = ref.is_full
        ref_projs = ref.projs
        if ref_reduction is None:
            ref_reduction = 'mode'
        if not ref_reduction in ('mode',):
            warnings.warn("For fluidized bed scans, 'mode' is a good"
                          " reduction choice. Another option that might not"
                          " introduce bubble bias is 'min', but this will cause"
                          " a density mismatch.")
        ref_rotational = ref.is_rotational
    else:
        ref_projs = []
        ref_path = None
        ref_full = None
        ref_rotational = False

    empty_path = None
    empty_projs = None
    empty_rotational = False
    if scan.density_factor is None:
        assert ref_path is not None
        if scan.empty is not None:
            empty_rotational = scan.empty.is_rotational
            assert not scan.empty.is_rotational
            assert not scan.empty.is_full
            empty_path = scan.empty.projs_dir
            empty_projs = [p for p in range(
                scan.empty.proj_start, scan.empty.proj_end
            )]

    darks_path = None
    darks_projs = None
    if scan.darks is not None:
        darks_path = scan.darks.projs_dir
        darks_projs = [p for p in range(
            scan.darks.proj_start, scan.darks.proj_end
        )]

    sino = reco.load_sinogram(
        t_range=timeframes,
        # t_range=range(t, t + 1),
        # t_range=range(1, t),  # hacky way to average projs
        ref_rotational=ref_rotational,
        ref_reduction=ref_reduction,
        ref_path=ref_path,
        ref_projs=ref_projs,
        empty_path=empty_path,
        empty_rotational=empty_rotational,
        empty_projs=empty_projs,
        darks_ran=darks_projs,
        darks_path=darks_path,
        ref_full=ref_full,
        density_factor=scan.density_factor,
        col_inner_diameter=scan.col_inner_diameter,
        **kwargs,
    )
    return sino


def _sino_static(
    scan: StaticScan,
    reco: Reconstruction,
    ref: Scan = None,
    ref_reduction: str = None,
    cameras: Tuple = None,
    angles=None,
    **kwargs
):
    if cameras is None:
        cameras = scan.cameras
    elif not np.all([c in scan.cameras for c in cameras]):
        raise ValueError("One or more unknown cameras.")

    scan_angles = range(scan.proj_start, scan.proj_stop)
    if angles is None:
        angles = scan_angles
    elif not np.all([a in scan_angles for a in angles]):
        raise ValueError(
            f"One or more unknown projection angles. {scan}"
            f" has angles defined for"
            f" {scan.proj_start}-{scan.proj_stop}."
        )

    if isinstance(ref, StaticScan):
        ref_path = ref.projs_dir
        ref_full = ref.is_full
        ref_projs = [a - scan.proj_start + ref.proj_start for a in angles]
        ref_rotational = ref.is_rotational

        if ref_reduction is None:
            ref_reduction = 'mean'
        if not ref_reduction in ('mean', 'median'):
            warnings.warn("For static scans, 'mean' or 'median' are good"
                          " reduction choices.")
    else:
        ref_projs = []
        ref_path = None
        ref_full = None
        ref_rotational = False

    empty_path = None
    empty_projs = None
    empty_rotational = False
    if scan.density_factor is None:
        assert ref_path is not None
        if scan.empty is not None:
            empty_rotational = scan.empty.is_rotational
            assert not scan.empty.is_full
            empty_path = scan.empty.projs_dir
            empty_projs = [a - scan.proj_start + scan.empty.proj_end
                           for a in angles]

    darks_ran = None
    darks_path = None
    if scan.darks is not None:
        darks_path = scan.darks.projs_dir
        darks_ran = range(scan.darks.proj_start, scan.darks.proj_end)

    sinos = []
    for cam in cameras:
        sino = reco.load_sinogram(
            t_range=angles,
            cameras=(cam,),
            ref_path=ref_path,
            ref_projs=ref_projs,
            ref_rotational=ref_rotational,
            darks_path=darks_path,
            darks_ran=darks_ran,
            ref_full=ref_full,
            empty_path=empty_path,
            empty_rotational=empty_rotational,
            empty_projs=empty_projs,
            col_inner_diameter=scan.col_inner_diameter,
            **kwargs
        )
        sino = np.squeeze(sino, axis=0)
        sinos.append(sino)

    # concat 3 cams into 1
    sino_flat = np.concatenate(sinos, axis=0)

    scan._geometry_rotation_offset = np.pi / 6
    geom_flat = np.array(
        [
            g
            for c, gs in scan.geometry().items()
            if c in cameras
            for i, g in enumerate(gs, scan.proj_start)
            if i in angles
        ]
    )
    return sino_flat, geom_flat


def _inner_reco(
    scan: Scan,
    reco: Reconstruction,
    recodir: str,
    algo: str,
    sino,
    geoms,
    voxels,
    voxel_size,
    iters: int,
    t: int = None,
    locking: bool = False,
    overwrite: bool = False,
    save: bool = False,
    save_mat: bool = False,
    min_constraint: float = 0.0,
    max_constraint: float = 1.0,
    plot=False,
):
    def _fname(ext, t=None):
        """Filename for reconstruction"""

        if t is not None:
            save_name = f"recon_t{str(t).zfill(6)}.{ext}"
            return os.path.join(
                recodir,
                scan.name,
                f"size_{voxels[0]}_algo_{algo}_iters_{iters}",
                save_name,
            )
        else:
            return os.path.join(
                recodir, scan.name,
                f"size_{voxels[0]}_algo_{algo}_iters_{iters}.{ext}"
            )

    # sino[0, ...] = np.mean(sino, axis=0)  # hacky averaging projs

    # for i, s in enumerate(sino):
    #    sino[i, ...] = savgol_filter(s, 7, 5, axis=-2, mode='mirror')
    #    sino[i, ...] = savgol_filter(s, 7, 5, axis=-1, mode='mirror')

    # sino[sino < 0.025] = 0.

    # for i, s in enumerate(sino):
    #     sino[i, ...] = savgol_filter(s, 13, 7, axis=-2, mode='mirror')

    #    sino[i, ...] = savgol_filter(s, 13, 7, axis=-1, mode='mirror')
    # sino[sino < 0.5] = 0.

    # for i, s in enumerate(sino[0]):
    #    sino[0, i] = denoise_wavelet(s, 1e-1, rescale_sigma=True)

    filename = _fname("npy", t)
    if os.path.exists(filename) and not overwrite:
        print(f"File {filename} exists, continuing.")
        return

    if locking:
        lockfile = _fname("lock", t)
        if os.path.exists(lockfile):
            print(f"Lockfile {lockfile} exists, continuing.")
            return
        else:
            os.makedirs(os.path.dirname(lockfile), exist_ok=True)
            Path(lockfile).touch()

    x = None
    try:
        print(f"Starting {filename}...")

        proj_id, proj_geom = reco.sino_gpu_and_proj_geom(sino, geoms)

        vol_id, vol_geom = reco.backward(
            proj_id,
            proj_geom,
            algo=algo,
            voxels=voxels,
            voxel_size=voxel_size,
            iters=iters,
            min_constraint=min_constraint,
            max_constraint=max_constraint,
            col_mask=True)
        x = reco.volume(vol_id)

        w, h = voxels[0] * voxel_size / 2, voxels[2] * voxel_size / 2
        infodict = {
            "timeframe": t,
            "name": scan.name,
            "volume": x,
            "geometry": geoms,
            "algorithm": algo,
            "nr_iters": iters,
            "vol_params": [  # for backwards compatibility
                voxels,
                [-w, -w, -h],
                [w, w, h],
                [voxel_size] * 3
            ]}

        print(f"Saving {filename}...")
        if save or save_mat:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        if save:
            np.save(filename, infodict, allow_pickle=True)

        if save_mat:
            from scipy.io import savemat

            filename = _fname("mat", t)
            savemat(filename, infodict)
    finally:
        if locking:
            try:
                print(f"Lockfile {_fname('lock', t)} could not be unlinked.")
                Path(_fname("lock", t)).unlink()
            except:
                pass

    if plot and x is not None:
        import pyqtgraph as pq

        pq.image(x.T)
        plt.figure()
        plt.show()


def run():
    parser = argparse.ArgumentParser(
        description="This script allows an organized way of reconstructing"
                    " fluidized beds, given the scans in settings.py.")

    parser.add_argument(
        "scan",
        type=str,
        help="Name of the Scan to run."
             " Must be available in settings.py. Runs"
             " all SCANS from settings.py if not provided."
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="Name of the reference scan."
             " Must be available in settings.py. If not "
             " given, will use `scan.references[0]`.",
        default=None,
    )
    parser.add_argument(
        "--recodir", type=str, help="directory to store reconstructions",
        default="./"
    )
    parser.add_argument(
        "--algo", type=str, help="Algorithm to use (sirt, fdk, nesterov)",
        default=None
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="Number of iterations for iterative algorithms.",
        default=None,
    )
    parser.add_argument(
        "--cam", type=int, help="Reconstructs with a single camera.",
        default=None
    )
    parser.add_argument(
        "--voxels-x",
        type=int,
        help="Number of voxels in one dimension of the horizontal plane.",
        default=300,
    )
    parser.add_argument(
        "--voxels-z",
        type=int,
        help="Number of voxels in one dimension of the horizontal plane.",
        default=850,
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        help="Isotropic voxel size (cm)",
        default=0.0183333333,
    )
    parser.add_argument(
        "--time",
        type=int,
        help="Starting timeframe from a dynamic scan.",
        default=None,
    )
    parser.add_argument(
        "--time-start",
        type=int,
        help="Starting timeframe from a dynamic scan.",
        default=None,
    )
    parser.add_argument(
        "--time-end",
        type=int,
        help="Timeframe to end, in a dynamic scan.",
        default=None,
    )
    parser.add_argument(
        "--angle",
        type=int,
        help="Reconstruct a single angle from a rotational scan.",
        default=None,
    )
    parser.add_argument(
        "--detector-rows",
        type=int,
        nargs=2,
        help="Subselect a detector row range, i.e. --detector-rows 100 500",
        default=None,
    )
    parser.add_argument(
        "--ref-max",
        type=int,
        help="Limit the number of averaging reference projections, to save loading time.",
        default=None,
    )
    parser.add_argument(
        "--ref-reduction",
        type=str,
        help="For nonrotational references, how to average the projections."
             " When the ref is a fluidized bed, choose `mode` or `min`. For "
             " e.g. a full bed with many shots, choose `mean` or `median`.",
        default=None,
    )

    parser.add_argument("--plot", "--p", action="store_true", default=False)
    parser.add_argument("--save", "--s", action="store_true", default=False)
    parser.add_argument("--save-mat", action="store_true", default=False)
    parser.add_argument("--locking", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()
    name = args.scan
    refname = args.ref
    algo = args.algo
    iters = args.iters
    recodir = args.recodir
    voxels_x = args.voxels_x
    voxels_z = args.voxels_z
    voxel_size = args.voxel_size
    time = args.time
    time_start = args.time_start
    time_end = args.time_end
    angle = args.angle
    cam = args.cam
    plot = args.plot
    save = args.save
    save_mat = args.save_mat
    locking = args.locking
    overwrite = args.overwrite
    detector_rows = args.detector_rows
    ref_max = args.ref_max
    ref_reduction = args.ref_reduction

    if plot:
        from fbrct.plotting import plt

        plt.rcParams.update({"figure.raise_window": False})

    if name != "all":
        scans = sett.get_scans(name)
        if len(scans) == 0:
            raise ValueError(f"Scan {name} not found in "
                             f"{[s.name for s in sett.SCANS]}.")
    else:
        scans = sett.SCANS

    _kwargs = {"plot": plot, "locking": locking, "save": save,
               "overwrite": overwrite,
               'save_mat': save_mat, 'ref_reduction': ref_reduction,
               'voxels': [voxels_x, voxels_x, voxels_z],
               'voxel_size': voxel_size}
    if algo is not None:
        _kwargs["algo"] = algo
    if iters is not None:
        _kwargs["iters"] = iters

    times = None
    if time is not None:
        assert time_start is None and time_end is None
        times = [t for t in range(time, time + 1)]
    if time_start is not None:
        assert time is None and time_end > time_start
        times = [t for t in range(time_start, time_end)]

    _kwargs["timeframes"] = times

    if cam is not None:
        _kwargs["cameras"] = [cam]
    if angle is not None:
        _kwargs["angles"] = [angle]
    if detector_rows is not None:
        assert 0 <= detector_rows[0] < detector_rows[1]
        _kwargs["detector_rows"] = range(detector_rows[0], detector_rows[1])

    for scan in scans:
        if refname is None:
            if len(scan.references) == 1:
                ref = scan.references[0]
            else:
                raise ValueError(
                    f"Zero, or multiple references defined for scan {scan.name}. Either"
                    f" define a reference in settings.py or provide with which reference"
                    f" to reconstruct."
                )
        else:
            found = False
            for ref in scan.references:
                if ref.name == refname:
                    found = True
                    break

            if not found:
                refs = sett.get_scans(refname)
                assert len(refs) == 1
                raise ValueError(f"Add {refs[0].name} to {scan}'s references.")

        if ref_max is not None:
            if isinstance(ref, StaticScan):
                if ref.is_rotational:
                    raise ValueError(
                        "Cannot set maximum to-use projections for averaging"
                        " reference. When the scan is rotational, there is only"
                        " one projection per angle, nothing to average."
                    )
                assert ref_max >= 1
                ref.proj_stop = min((ref.proj_start + ref_max, ref.proj_stop))
            elif isinstance(ref, FluidizedBedScan):
                if ref.projs is None:
                    ref.projs = range(ref_max)
                else:
                    ref.projs = range(ref.projs[0], ref.projs[0] + ref_max)
            else:
                raise NotImplementedError(f"Don't know how to apply `ref_max` "
                                          f"to {ref}.")

            if isinstance(scan.empty, StaticScan):
                if scan.empty.is_rotational:
                    raise ValueError(
                        "Cannot set maximum to-use projections for averaging"
                        " empty. When the scan is rotational, there is only"
                        " one projection per angle, nothing to average."
                    )
                assert ref_max >= 1
                scan.empty.proj_stop = min(
                    (scan.empty.proj_start + ref_max, scan.empty.proj_stop))
            else:
                raise NotImplementedError(f"Don't know how to apply `ref_max` "
                                          f"to {scan.empty}.")

        reconstruct(scan, recodir, ref, **_kwargs)


if __name__ == "__main__":
    run()  # avoid scope issues
