from abc import ABC

import numpy as np


def _manual_geometry(
    detector,
    cams=(1, 2, 3),
    nr_projs=1,
):
    """Generate a geometry from manual calibrated values."""

    from cate import xray, astra

    assert isinstance(detector, xray.Detector)

    # Log 2021-08-20.docx
    _MANUAL_SDD_1 = 120.8  # cm
    _MANUAL_SDD_2 = 121.5
    _MANUAL_SDD_3 = 123.1
    _MANUAL_COL_RADIUS = 6.0  # incl. wall
    _MANUAL_COL_DET_1 = 24.3  # from wall to det
    _MANUAL_COL_DET_2 = 23.8
    _MANUAL_COL_DET_3 = 26.4

    MANUAL_SOURE_RADIUS = (
        _MANUAL_SDD_1 - _MANUAL_COL_DET_1 - _MANUAL_COL_RADIUS / 2,
        _MANUAL_SDD_2 - _MANUAL_COL_DET_2 - _MANUAL_COL_RADIUS / 2,
        _MANUAL_SDD_3 - _MANUAL_COL_DET_3 - _MANUAL_COL_RADIUS / 2,
    )
    MANUAL_DET_RADIUS = (
        _MANUAL_COL_DET_1 + _MANUAL_COL_RADIUS / 2,
        _MANUAL_COL_DET_2 + _MANUAL_COL_RADIUS / 2,
        _MANUAL_COL_DET_3 + _MANUAL_COL_RADIUS / 2,
    )

    geoms_all_cams = []
    for c, cam in enumerate(cams):
        geom = xray.Geometry(
            source=np.array([MANUAL_SOURE_RADIUS[c], 0.0, 0.0]),
            detector=np.array([-MANUAL_DET_RADIUS[c], 0.0, 0.0]),
        )

        # cam 1,2,3 at angle 0, 120, 240 degrees
        geom = xray.transform(geom, yaw=c * 1 / 3 * 2 * np.pi / 3)

        geoms = []
        ang_increment = 2 * np.pi / nr_projs
        for i in range(nr_projs):
            g = xray.transform(geom, yaw=i * ang_increment)
            v = astra.geom2astravec(g, detector.todict())
            geoms.append(v)

        geoms_all_cams.append(geoms)

    return geoms_all_cams


def cate_to_astra(path, det, geom_scaling_factor=None, angles=None):
    """Convert `Geometry` objects from our calibration package to the
    ASTRA vector convention."""

    import pickle
    from cate import astra, xray
    from numpy.lib.format import read_magic, _check_version, _read_array_header

    class RenamingUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "StaticGeometry":
                name = "Geometry"
            return super().find_class(module, name)

    with open(path, "rb") as fp:
        version = read_magic(fp)
        _check_version(version)
        dtype = _read_array_header(fp, version)[2]
        assert dtype.hasobject
        multicam_geom = RenamingUnpickler(fp).load()[0]

    detector = astra.Detector(
        det["rows"], det["cols"], det["pixel_width"], det["pixel_height"]
    )

    def _to_astra_vec(g):
        v = astra.geom2astravec(g, detector.todict())
        if geom_scaling_factor is not None:
            v = np.array(v) * geom_scaling_factor
        return v

    if angles is None:
        geoms = []
        for _, g in sorted(multicam_geom.items()):
            geoms.append(_to_astra_vec(g))
        return geoms
    else:
        geoms_all_cams = {}
        for cam in list(multicam_geom.keys()):
            geoms = []
            for a in angles:
                g = xray.transform(multicam_geom[cam], yaw=a)
                geoms.append(_to_astra_vec(g))
            geoms_all_cams[cam] = geoms

        return geoms_all_cams


def astra_to_kernelkit(vectors, det):
    from kernelkit.geom import ProjectionGeometry, Detector

    geoms = []
    for vec in vectors:
        u = np.array(vec[6:9])
        v = np.array(vec[9:12])
        geom = ProjectionGeometry(
            source_position=vec[0:3],
            detector_position=vec[3:6],
            u=u / np.linalg.norm(u),
            v=v / np.linalg.norm(v),
            detector=Detector(
                rows=det['rows'],
                cols=det['cols'],
                pixel_width=det['pixel_width'],
                pixel_height=det['pixel_height'],
            ),
        )
        geoms.append(geom)

    return geoms


class Phantom:
    def __init__(self, diameter, position=None):
        if position is not None:
            if position not in ["center", "side", "wall"]:
                raise ValueError()

        self.position = position
        self.diameter = diameter

    @property
    def radius(self):
        return self.diameter / 2


class MovingPhantom(Phantom):
    def __init__(self, diameter, interesting_time: slice = None, **kwargs):
        if interesting_time is None:
            interesting_time = slice(None)  # :, basically

        self.interesting_time = interesting_time

        super().__init__(diameter, **kwargs)


class Scan(ABC):
    def __init__(
        self,
        name,
        detector,
        projs_dir,
        projs=None,
        projs_offset=None,
        geometry=None,
        geometry_scaling_factor=None,
        geometry_rotation_offset=0.0,
        geometry_manual=None,
        framerate=None,
        cameras=(1, 2, 3),
        references=None,
        darks=None,
        empty=None,
        is_rotational: bool = False,
        cams_are_rotated: bool = False,
        is_full: bool = False,
        col_inner_diameter: float = None,
        density_factor: float = None
    ):
        if references is None:
            references = []
        self.name = name
        self.detector = detector
        self.projs_dir = projs_dir
        self._geometry = geometry
        self._geometry_scaling_factor = geometry_scaling_factor
        self._geometry_rotation_offset = geometry_rotation_offset
        self._geometry_manual = geometry_manual
        self.framerate = framerate
        self.cameras = cameras
        self.phantoms = []
        self.references = references
        self.darks = darks
        self.empty = empty
        self.is_rotational = is_rotational
        self.is_full = is_full
        self.projs = projs
        self.projs_offset = projs_offset
        self.col_inner_diameter = col_inner_diameter
        self.density_factor = density_factor
        self.cams_are_rotated = cams_are_rotated

    def add_phantom(self, phantom: Phantom):
        self.phantoms.append(phantom)

    def geometry(self):
        raise NotImplementedError

    def __str__(self):
        return f"Scan in directory {self.projs_dir}"


class StaticScan(Scan):
    """Scan made from a static object.

    The scanned object does not change over time.
    The scan be on a rotation table though.
    If a scan features a static and dynamic part (for example, when in the first
    frames there is no movement) then two different `Scan` objects have to be
    made, with different projection ranges.
    """

    def __init__(
        self,
        *args,
        proj_start: int,
        proj_end: int,
        **kwargs,
    ):
        assert proj_end > proj_start > 0
        self.proj_start = proj_start
        self.proj_end = proj_end
        projs = list(range(proj_start, proj_end))
        super().__init__(*args, projs, **kwargs)

    @property
    def nr_projs(self):
        return self.proj_end - self.proj_start

    def geometry(self):
        if self._geometry_manual is True:
            return _manual_geometry(self.cameras, self.nr_projs)

        ang_increment = 2 * np.pi / self.nr_projs if self.is_rotational else 0.0
        angles = [self._geometry_rotation_offset + i * ang_increment
                           for i in range(self.nr_projs)]

        return cate_to_astra(self._geometry,
                             self.detector,
                             self._geometry_scaling_factor,
                             angles=angles)


class DynamicScan(Scan):
    """The scanned object changes over time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def geometry(self):
        if self._geometry_manual:
            geoms = _manual_geometry(self.cameras, nr_projs=1)
            return [g[0] for g in geoms]  # flattening

        if self._geometry:
            return cate_to_astra(
                self._geometry,
                self.detector,
                self._geometry_scaling_factor,
            )


class AveragedScan(Scan):
    """The scanned object changes over time but we're interested in time-averages."""

    def __init__(
        self,
        name,
        detector,
        projs_dir,
        proj_start: int,
        proj_end: int,
        **kwargs,
    ):
        assert proj_end > proj_start > 0
        self.proj_start = proj_start
        self.proj_end = proj_end
        projs = list(range(proj_start, proj_end))
        super().__init__(name, detector, projs_dir, projs, **kwargs)

    def geometry(self):
        if self._geometry_manual:
            geoms = _manual_geometry(self.cameras, nr_projs=1)
            return [g[0] for g in geoms]  # flattening

        if self._geometry:
            return cate_to_astra(
                self._geometry,
                self.detector,
                self._geometry_scaling_factor,
            )


class TraverseScan(DynamicScan):
    def __init__(self, *args, timeframes, motor_velocity, **kwargs):
        self.timeframes = timeframes
        self.expected_velocity = motor_velocity
        assert 'is_rotational' not in kwargs, ("TraverseScan is non"
                                               "rotational by default.")
        kwargs['is_rotational'] = False
        super().__init__(*args, **kwargs)


class FluidizedBedScan(DynamicScan):
    def __init__(self, *args, liter_per_min, **kwargs):
        self.liter_per_min = liter_per_min
        assert 'is_rotational' not in kwargs or kwargs[
            'is_rotational'] is False, ("FluidizedBedScan is not rotational.")
        assert 'is_full' not in kwargs or kwargs['is_full'] is True, (
            "FluidizedBedScan is always with full column.")
        kwargs['is_full'] = True
        kwargs['is_rotational'] = False
        super().__init__(*args, **kwargs)
