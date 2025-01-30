import itertools
import os
import pathlib
import re
import warnings
from typing import Sequence
from joblib import Memory

import numpy as np
from tifffile import tifffile
import imageio
from tqdm import tqdm

PROJECTION_FILE_REGEX = "camera ([1-3])\\\img_([0-9]{1,6})\.tif$"

# we use joblibs `Memory` to cache long results
path = pathlib.Path(__file__).parent.resolve()
cachedir = str(path.parent / "cache")
memory = Memory(cachedir, verbose=0)


def _collect_fnames(
    path: str,
    regex: str = PROJECTION_FILE_REGEX,
):
    if not os.path.exists(path):
        raise IOError(f"The path to {path} does not seem to exist.")

    results, results_filenames = [], []
    regex = re.compile(regex)
    print(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            full_filename = os.path.join(root, file)
            match = regex.search(full_filename)
            if match is not None:
                groups = match.groups()

                tmp_result = [None] * len(groups)
                for i, group in enumerate(groups):
                    if not group.isdigit():
                        raise Exception(
                            "The regex captured a non-digit, cannot proceed."
                        )

                    tmp_result[i] = int(group)

                results.append(tmp_result)
                results_filenames.append(full_filename)

    return results, results_filenames


def roughly_equal(n1, n2, play=100):
    """Determine if n1 and n2 are roughly equal - within an absolute margin (play)"""
    n2_min = n2 - play
    n2_max = n2 + play
    return n2_min < n1 < n2_max


def load(
    path: str,
    time_range: range = None,
    time_offsets=None,
    regex: str = PROJECTION_FILE_REGEX,
    dtype=np.float32,
    verbose: bool = True,
    cameras: Sequence = (1, 2, 3),
    detector_rows: Sequence = None,
    average: bool = False,
    correct_beam_hardening: bool = False,
    bhc: dict = {'a': 1.0, 'b': 1.0, 'c': 3.0}
):
    """Load a stack of data from disk using a pattern."""

    results, results_filenames = _collect_fnames(path, regex)
    # Check the results for continuity in the subsequences range
    lists = list(zip(*results))

    first_cam = cameras[0]
    amount_matches = lists[0].count(first_cam)
    assert amount_matches > 0
    for cam in cameras:
        if average:
            assert roughly_equal(amount_matches, lists[0].count(cam)) is True
        else:
            assert amount_matches == lists[0].count(cam)

    if time_range is None:
        time_range = range(np.min(lists[1]), np.max(lists[1]) + 1)

    # load into results
    im_shape = list(imageio.v2.imread(results_filenames[0]).shape)
    if detector_rows is None:
        rows = slice(0, im_shape[0])
    else:
        rows = slice(detector_rows.start, detector_rows.stop)

    if average:
        ims = np.zeros((1, len(cameras), *im_shape), dtype=dtype)
    else:
        ims = np.zeros((len(time_range), len(cameras), *im_shape), dtype=dtype)

    # make a dictionary with in the first key the detector, and second key
    # the timestep
    detector_timesteps = {i: {} for i in cameras}
    for i, ((cam_id, t), filename) in enumerate(
        zip(results, results_filenames)):
        if cam_id in cameras:
            t_actual = t - time_offsets[cam_id] if time_offsets is not None else t
            detector_timesteps[cam_id][t_actual] = filename

    # check if wanted timesteps are in the dict, and load them
    count = 0
    for t_i, t in enumerate(tqdm(time_range)) if verbose else enumerate(
        time_range):
        for d_i, d in enumerate(detector_timesteps.keys()):
            if not t in detector_timesteps[d]:
                raise FileNotFoundError(
                    f"Could not find timestep {t} from "
                    f"detector {d} in directory {path}."
                )
            if verbose:
                tqdm.write(f"Reading {detector_timesteps[d][t]}")
            if average:
                ims[0, d_i, rows] += tifffile.imread(detector_timesteps[d][t],
                                                     maxworkers=1)[detector_rows][0]
                count += 1
            else:
                ims[t_i, d_i, rows] = tifffile.imread(detector_timesteps[d][t],
                                                      maxworkers=1)[detector_rows]
    if average:
        ims /= count

    if correct_beam_hardening:
        ims = bhc['a'] * ims + bhc['b'] * ims**bhc['c']
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


def _apply_darkfields(dark, meas):
    dark = np.median(dark, axis=0)
    dark = np.expand_dims(dark, 0)
    dark = np.clip(dark, 0.0, None, out=dark)

    meas[:] -= dark
    np.clip(meas, 0.0, None, out=meas)
    _isfinite(meas)


def _scatter_correct(meas, scatter_mean: float = 0.0):
    # TODO update to deal with ndarray scatter
    if scatter_mean != 0.0:
        meas -= scatter_mean
        np.clip(meas, 0.0, None, out=meas)
        _isfinite(meas)


def compute_bed_density(empty, ref, L: float, nr_bins=1000,
                        max_bed_val=1.0) -> float:
    """Computes the average bed density along a ray with length L, using an
    empty column, histogram with `nr_bins` bins."""
    # ref is generally the full image.

    # computing log(ref/meas) / log(ref)
    # should be normalized between 0 and 1
    np.clip(empty, 1.0, None, out=empty)
    bed = np.log(empty / ref, where=ref != 0)

    # This is an annoying value that I need to have in here, because sometimes
    # pieces of metal appear in the bed and they have huge attenuation. In such
    # case the modal value of the bed gets disturbed.
    # np.clip(bed, 0., max_bed_val, out=bed)
    # _isfinite(bed)

    # TODO Why is this here? Should already be covered by previous lines, right?
    np.clip(bed, 0.0, None, out=bed)
    # np.divide(bed, L, out=bed)
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
    ref=None,
    density_factor=1.0,
    dtype=np.float32,
    ref_full=True,
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
    return meas.astype(dtype)


def projection_numbers(path, cam=None) -> list:
    """Returns projection numbers of camera `cam`."""
    if cam is None:
        cam = 1

    results, results_fnames = _collect_fnames(path)
    return sorted([r[1] for r in results if r[0] == cam])


def load_first_projection_number(path) -> int:
    results, results_fnames = _collect_fnames(path)
    # Check the results for continuity in the subsequences range
    lists = list(zip(*results))
    return sorted(lists[1])[0]
