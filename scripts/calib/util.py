import copy
from typing import Sequence

import numpy as np
from cate import xray
from cate.annotate import Annotator
from cate.astra import geom2astravec
from cate.param import ScalarParameter, VectorParameter
from cate.xray import markers_from_leastsquares_intersection, \
    XrayOptimizationProblem
from cate.xray import Geometry, shift, transform
from cate import annotate

from fbrct.loader import load, preprocess
from fbrct.reco import Reconstruction


class DelftNeedleEntityLocations(annotate.EntityLocations):
    """
    A bit cryptic, but the encoding is based on the 4 vertical sequences
    of needles glued onto the column, from top to bottom:
        BBB = ball ball ball
        BEE = ball eye eye
        EB = eye ball
        B = ball
    Then with `top`, `middle` of `bottom` I encode which of the three needles
    it is, and also provide additional redundant information for convenience.
    Note however that the vertical description does not tell how far up the
    the needle is in the column, only its relative position. There does not
    seem to be good horizontal alignment, so its difficult to segment the
    column in annotated layers.
    """

    ENTITIES = (
        "BBB  top ball stake",
        "BBB  middle ball stake",
        "BBB  bottom ball stake",
        "BEE  top ball drill",
        "BEE  middle eye drill",
        "BEE  bottom eye drill",
        "EB  top eye stake",
        "EB  bottom ball drill",
        "B  ball drill",
    )

    @staticmethod
    def nr_entities():
        return len(DelftNeedleEntityLocations.ENTITIES)

    @staticmethod
    def get_iter():
        return iter(DelftNeedleEntityLocations.ENTITIES)


def triple_camera_circular_geometry(
    source_positions: Sequence,
    detector_positions: Sequence,
    angles: Sequence,
    optimize_rotation=False,
):
    """
    This function retunrns Geometry objects with free parameters, so that
    these can be optimized.

    Parameters:
     - each source and detector has a free position, and detector has RPY
      - with exception from detector 1, this is fixed.
     - one shift and global rotation w.r.t. center of rotation
     - for each angle there is a rotation of the rotation table

    :param source_position:
    :param detector_position:
    :param nr_angles:
    :param angle_start:
    :param angle_stop:
    :return:
    """
    nr_cams = len(source_positions)
    assert nr_cams == len(detector_positions)

    # The first cam needs to be fixed, to prevent arbitrary shifts in the
    # solution
    initial_geoms = [
        Geometry(
            source=VectorParameter(source_positions[0]),
            detector=detector_positions[0],
            roll=None,  # are computed automatically, from src and det
            pitch=None,
            yaw=None,
        )
    ]
    for i in range(1, nr_cams):
        initial_geoms.append(
            Geometry(
                source=VectorParameter(source_positions[i]),
                detector=VectorParameter(detector_positions[i]),
                roll=ScalarParameter(None),
                pitch=ScalarParameter(None),
                yaw=ScalarParameter(None),
            )
        )

    # the whole set-up can be arbitrarily shifted and rotated in space
    # but while keeping sources and detectors relatively fixed
    shift_param = VectorParameter(np.array([0.0, 0.0, 0.0]))
    initial_geoms = [shift(g, shift_param) for g in initial_geoms]
    roll_param, pitch_param, yaw_param = [ScalarParameter(0.0) for _ in range(3)]
    initial_geoms = [
        transform(g, roll_param, pitch_param, yaw_param) for g in initial_geoms
    ]

    if optimize_rotation:
        # per initial geom, make a list of angles
        # rotations per angle, the first rotation is not optimizable
        geoms = [[transform(g, yaw=angles[0])] for g in initial_geoms]
        # a rotation param for each angle of the rotation table.
        # each detector sees the same rotation, so the angle should be the same
        # for all initial geometries of the source-detectors
        for i in range(1, len(angles)):
            angle = ScalarParameter(angles[i] - angles[0])
            for cam_angles in geoms:
                cam_angles.append(transform(cam_angles[0], yaw=angle))
    else:
        geoms = []
        for g in initial_geoms:
            ang = [transform(g, yaw=a) for a in angles]
            geoms.append(ang)

    return geoms


def _load_projs(path, t_range: range, camera: int):
    p = load(path, time_range=t_range, cameras=[camera])
    return preprocess(p)


def annotated_data(
    projs_path: str,
    times: Sequence,
    fname: str,
    resource_path: str,
    cameras: Sequence = (1, 2, 3),
    open_annotator: bool = False,
    vmin=None,
    vmax=None,
) -> list:
    """(Re)store marker projection coordinates from annotations

    :return dict
        Dictionary of `dict`, each dict being a projection angle, and each item
        from the dictionary is a key-value pair of identifier and pixel
        location."""

    data = {c: [] for c in cameras}
    for cam in cameras:
        for t in times:
            # open a EntityLocations class for this file, in the
            points = DelftNeedleEntityLocations(
                f"{resource_path}/{fname}_cam{cam}.npy",
                t)
            if open_annotator:
                projs = _load_projs(projs_path, t_range=range(t, t + 1),
                                    camera=cam)
                projs = np.squeeze(projs)
                Annotator(points, projs, block=True, vmin=vmin, vmax=vmax)

            l = points.locations()
            data[cam].append(l)

    return data


def triangle_geom(
    src_rad, det_rad, rotation=False, shift=False, fix_first_det=True,
):
    geoms = []
    for i, src_a in enumerate([0, 2 / 3 * np.pi, 4 / 3 * np.pi]):
        det_a = src_a + np.pi  # opposing
        src = src_rad * np.array([np.cos(src_a), np.sin(src_a), 0])
        det = det_rad * np.array([np.cos(det_a), np.sin(det_a), 0])
        if i == 0 and fix_first_det:
            print("Fixing first detector.")
            geom = xray.Geometry(
                source=VectorParameter(src),
                # source=src,
                detector=det,
                roll=None,
                pitch=None,
                yaw=None,
            )
        else:
            geom = xray.Geometry(
                source=VectorParameter(src),
                detector=VectorParameter(det),
                roll=ScalarParameter(None),
                pitch=ScalarParameter(None),
                yaw=ScalarParameter(None),
            )

        geoms.append(geom)

    if rotation:
        # transform the whole geometry by a global rotation, this is the
        # same as if the phantom rotated

        rotation_roll = ScalarParameter(0.0)
        rotation_pitch = ScalarParameter(0.0)
        rotation_yaw = ScalarParameter(0.0)

        for i in range(len(geoms)):
            geoms[i] = xray.transform(
                geoms[i], rotation_roll, rotation_pitch, rotation_yaw
            )

    if shift:
        shift_param = VectorParameter(np.array([0.0, 0.0, 0.0]))

        for i in range(len(geoms)):
            geoms[i] = xray.shift(geoms[i], shift_param)

    return geoms


def astra_reco_rotation_singlecamera(
    reco: Reconstruction,
    data,
    geoms,
    algo,
    voxels,
    voxel_size,
    **kwargs):
    vectors = np.array([geom2astravec(g, reco.detector) for g in geoms])
    proj_id, proj_geom = reco.sino_gpu_and_proj_geom(data, vectors)
    vol_id, vol_geom = reco.backward(
        proj_id, proj_geom, algo=algo, voxels=voxels,
        voxel_size=voxel_size, **kwargs)
    return vol_id, vol_geom


def astra_reco(reco: Reconstruction,
               projs,
               geoms,
               algo,
               voxels,
               voxel_size,
               **kwargs):
    """
    Reconstruction without rotation table:
     - 3 sources
     - 3 detectors
     - triangular set-up
    """
    assert len(geoms) == 3
    vectors = np.array([geom2astravec(g, reco.detector) for g in geoms])
    proj_id, proj_geom = reco.sino_gpu_and_proj_geom(projs, vectors)
    vol_id, vol_geom = reco.backward(proj_id, proj_geom, algo=algo,
                                     voxels=voxels, voxel_size=voxel_size,
                                     **kwargs)
    return vol_id, vol_geom


def astra_project(reco, vol_id, vol_geom, geoms):
    vectors = np.array([geom2astravec(g, reco.detector) for g in geoms])
    # zero-sinogram
    sino_id, proj_geom = reco.sino_gpu_and_proj_geom(0.0, vectors)
    proj_id = reco.forward(
        volume_id=vol_id,
        volume_geom=vol_geom,
        projection_geom=proj_geom)
    return reco.sinogram(proj_id)


def astra_residual(reco, data, vol_id, vol_geom, geoms):
    """Projects then substracts a volume with `vol_id` and `vol_geom`
    onto projections from `projs_path` with `nrs`.

    Using `geoms` or `angles`. If geoms are perfect, then the residual will be
    zero. Otherwise it will show some geometry forward-backward mismatch.
    """
    reproj = astra_project(reco, vol_id, vol_geom, geoms)
    assert reproj.shape == data.shape, (
        f"reproj shape is {reproj.shape} while" f" data.shape is {data.shape}")
    return data - reproj


def plot_projections(res, vmin=None, vmax=None, title=None):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=1, ncols=res.shape[1])
    if title is not None:
        plt.title(title)

    if res.shape[1] == 1:
        plt.imshow(res[:, 0], vmin=vmin, vmax=vmax)
        plt.colorbar()
    else:
        for i in range(res.shape[1]):
            im = axs[i].imshow(res[:, i], vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axs[i])

    plt.pause(0.1)


def prep_projs(projs):
    projs = np.squeeze(projs)
    projs = np.swapaxes(projs, 0, 1)
    return np.ascontiguousarray(projs)


def marker_optimization(
    geoms, data, nr_iters: int = 1, max_nfev=10, plot=False, **kwargs):
    from cate.param import params2ndarray
    from scipy.optimize import least_squares

    # `geoms` and `markers` will be optimized in-place, so we make a
    # backup here for later reference.
    geoms_before = copy.deepcopy(geoms)

    # Since the prescan is not perfect, we do a simple optimization process
    # over the prescan parameters.
    assert nr_iters > 0
    for i in range(nr_iters):
        # Get Least-Square optimal points by analytically backprojection the
        # marker locations, which is an LS intersection of lines.
        markers = markers_from_leastsquares_intersection(
            geoms,
            data,
            optimizable=False,
            plot=plot)
        # Then find the optimal geoms given the `points` and `data`
        problem = XrayOptimizationProblem(
            markers=markers,
            geoms=geoms,
            data=data,
            use_multiprocessing=False)
        _ = least_squares(  # solution is in-place
            fun=problem,
            x0=params2ndarray(problem.params()),
            bounds=problem.bounds(),
            verbose=2,
            jac="3-point",
            max_nfev=max_nfev,
            **kwargs)

        np.set_printoptions(precision=4, suppress=True)
        for i, (g1, g2) in enumerate(zip(geoms_before, geoms)):
            print("")
            print(f"--- GEOM {i} old:new properties ---")
            print(f"source   : {g1.source} : {g2.source}")
            print(f"detector : {g1.detector} : {g2.detector}")
            print(f"roll     : {g1.roll} : {g2.roll}")
            print(f"pitch    : {g1.pitch} : {g2.pitch}")
            print(f"yaw      : {g1.yaw} : {g2.yaw}")
            print(f"--- GEOM {i} unshifted old:new:err properties ---")
            g1_un = g1.decorated_geometry.decorated_geometry
            g2_un = g2.decorated_geometry.decorated_geometry
            print(
                f"source   : {g1_un.source} : {g2_un.source} : {g1_un.source - g2_un.source}"
            )
            print(
                f"detector : {g1_un.detector} : {g2_un.detector} : {g1_un.detector - g2_un.detector}"
            )
            print(
                f"roll     : {g1_un.roll} : {g2_un.roll} : {g1_un.roll - g2_un.roll}")
            print(
                f"pitch    : {g1_un.pitch} : {g2_un.pitch} : {g1_un.pitch - g2_un.pitch}"
            )
            print(
                f"yaw      : {g1_un.yaw} : {g2_un.yaw} : {g1_un.yaw - g2_un.yaw}")
            print(f"--- GEOM {i} old:new parameters ---")
            for i, (p1, p2) in enumerate(
                zip(g1.parameters(), g2.parameters())):
                print(f"{id(p1)} : {p1.value} : {p2.value}")

    # noinspection PyUnboundLocalVariable
    return markers
