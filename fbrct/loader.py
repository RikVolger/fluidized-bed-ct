import itertools
import warnings
from typing import Sequence
from pathlib import Path

import numpy as np
from tifffile import tifffile
from tqdm import tqdm


def roughly_equal(n1, n2, play=100):
    """Determine if n1 and n2 are roughly equal - within an absolute margin (play)"""
    n2_min = n2 - play
    n2_max = n2 + play
    return n2_min < n1 < n2_max


def load(
    path: str | Path,
    time_range: range,
    time_offsets = None,
    dtype = np.float32,
    verbose: bool = True,
    cameras: Sequence = (1, 2, 3),
    img_shape: Sequence = (1524, 1548),
    average: bool = False,
):
    """Load a stack of data from disk using a pattern."""

    # Ideas for a renewed load function
    # path updated to pathlib.Path
    # check for path existence in load with path.exists(), raise sensible error 
    #   if it doesn't
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Failed to load data. {str(path)} does not exist")
    
    ims = np.zeros((len(time_range), len(cameras), *img_shape), dtype=dtype)

    n_images = None
    # change regex for glob. run for each camera.
    for cam in cameras:
        cam_path = path / f"camera {cam}"
        files = list(cam_path.glob("img_*.tif"))
        # For each glob, check that there are results in the list.
        if len(files) == 0:
            raise FileNotFoundError(f"No images found in {str(cam_path)}")
        if n_images is None:
            n_images = len(files)
        # for camera 2 & 3, check that the number of results matches the number 
        # obtained for cam 1 exact (time resolved) or is roughly equal (time average)
        elif average:
            if not roughly_equal(n_images, len(files)):
                raise FileNotFoundError(f"Number of images for camera {cam} is " \
                                        f"too different from the number of images " \
                                        f"for camera 1. Folder {str(path)}")
        else:
            if n_images != len(files):
                raise FileNotFoundError(f"Number of images for camera {cam} is " \
                                        f"too different from the number of images " \
                                        f"for camera 1. Folder {str(path)}")
        
        requested_files = [f"img_{t}.tif" for t in time_range]
        filenames = [file.name for file in files]
        matches = np.isin(requested_files, filenames)
        if average and sum(~matches) > 50:
            raise ValueError(f"Missing more than 50 requested files in " \
                             f"{str(cam_path)}")
        if not any(matches):
            raise ValueError(f"Found none of the requested timesteps for " \
                             f"{str(cam_path)}")
        elif not all(matches):
            raise ValueError(f"Did not find all requested timesteps for " \
                             f"{str(cam_path)}")
        
        # TODO The real timestamps are written to file (timestamp data.txt).
        #   These can be used for more precise timestamps, also accounting for
        #   skipped frames in the middle of a measurement.
        ordered_files = []
        for r_file in requested_files:
            for file in files:
                if file.name == r_file:
                    ordered_files.append(file)
        if verbose:
            print(f"Reading {cam_path}")
        ims[:, cam - 1, ...] = tifffile.imread(ordered_files,
                                               ioworkers=4, maxworkers=2)
        if verbose:
            print(f"Read.")
    
    return np.ascontiguousarray(ims)


# @memory.cache
def reference_via_mode(data, qu=1, deg=20, nr_bins=100, nr_linspace=1000):
    xp = np
    data = xp.asarray(data)

    if qu != 0:
        # edges are unreachable and are estimated by mean
        ref_mode = xp.mean(data, axis=0)
    else:
        ref_mode = xp.zeros(data.shape[1:])

    cams = range(data.shape[1])
    rows = range(qu, ref_mode.shape[-2] - qu)
    cols = range(ref_mode.shape[-1])
    total = len(rows) * len(cols) * len(cams)
    for cam, row, col in tqdm(itertools.product(cams, rows, cols),
                              total=total):
        pixeldata = data[:, cam, row - qu:row + qu + 1, col].flatten()
        bin_values, bin_edges = xp.histogram(
            pixeldata,
            bins=nr_bins,
            # range=(5500, 5600),
            density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:100] + bin_width / 2

        if xp != np:
            import cupy as cp
            assert xp == cp
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = xp.polyfit
                pol = fit(bin_centers, bin_values, deg=deg)
                xx = xp.linspace(xp.min(bin_centers),
                                 xp.max(bin_centers),
                                 nr_linspace)
                yy = xp.polyval(pol, xx)
        else:
            fit = np.polynomial.Polynomial.fit
            pol = fit(bin_centers, bin_values, deg=deg)
            xx, yy = pol.linspace(nr_linspace)

        mode_distr = xx[xp.argmax(yy)]
        ref_mode[cam, row, col] = mode_distr

        # import matplotlib.pyplot as plt
        # plt.figure(2)
        # plt.cla()
        # plt.hist(pixeldata, bins=nr_bins,
        #          # range=(5500, 7500),
        #          density=True)
        # plt.axvline(x=mode_distr, color='r', linewidth=1)
        # plt.pause(1.)

    return ref_mode


def _isfinite(a):
    m = a.min()
    M = a.max()
    assert np.isfinite(m)
    assert np.isfinite(M)


def compute_bed_density(empty, ref, L: float, nr_bins=1000,
                        max_bed_val=None) -> float:
    """Computes the average bed density along a ray with length L, using an
    empty column, histogram with `nr_bins` bins."""
    # ref is generally the full image.
    
    # computing log(ref/meas) / log(ref)
    # should be normalized between 0 and 1
    np.clip(empty, 1.0, None, out=empty)
    # Avoid divide by zero and have 1.0 in these locations.
    bed = np.log(empty / ref, where=ref != 0, out=np.ones_like(empty))
        
    # This is an annoying value that I need to have in here, because sometimes
    # pieces of metal appear in the bed and they have huge attenuation. In such
    # case the modal value of the bed gets disturbed.
    np.clip(bed, 0.0, max_bed_val, out=bed)
    _isfinite(bed)

    modal_values = np.zeros(bed.shape)
    sum = 0.0
    for i in range(bed.shape[0]):
        counts, bins = np.histogram(bed[i, :, 500:1000].flatten(), bins=nr_bins)
        counts[:100] = 0.0
        max_value = bins[np.argmax(counts)]  # corresponds to inner diam
        print(f"Proj {i}: mode of column statistic: ", max_value)
        print(f"Proj {i}: density approx.: ", 1 / L * max_value)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(bed[i, :, 500:1000].flatten(), bins=1000)
        # plt.show()
        sum += (1 / L) * max_value
        modal_values[i, ...] = max_value / L

    avg = sum / bed.shape[0]  # average density over nr. projs/cams
    print("Average density: ", avg)
    return modal_values
    # return avg
    # return bed


def preprocess(
    meas,
    ref = None,
    density_factor = 1.0,
    dtype = np.float32,
    ref_full = True,
    average = False,
):
    """A simple implementation of Beer-Lambert with referencing

    :type ref_full: If we reconstruct bubbles, the reference is a full
    column, and the log computation is inverted.
    """

    if ref is not None:
        if not ref_full:
            # we want to measure lower density, so need to multiply by -1
            # -(log(meas) - log(ref)) = log(ref/meas)
            np.divide(ref, meas, out=meas, where=meas != 0)
            _isfinite(ref)
        else:
            np.divide(meas, ref, out=meas, where=ref != 0)
            _isfinite(meas)

        np.clip(meas, 1.0, None, out=meas)

    np.log(meas, out=meas)
    np.divide(meas, density_factor, out=meas, where=density_factor != 0)
    _isfinite(meas)
    if average:
        meas = np.mean(meas, axis=0, dtype=dtype)
        # reintroduce the time axis with just a single input, it is expected elsewhere.
        meas = np.expand_dims(meas, axis=0)
    return meas.astype(dtype)

