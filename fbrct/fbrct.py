import numpy as np


def column_mask_2d(shp, r=None, center=None, val=lambda x, y: 1):
    """Generate a 2D mask binary mask

    :param shp: Shape tuple
    :param r: Radius leave to None to compute automatically
    :param center: leave to None to compute automatically
    :return: np.ndarray
    """
    assert len(shp) == 2
    s1, s2 = shp

    if center is None:
        cx, cy = s1 // 2, s2 // 2
    else:
        cx, cy = center

    if r is None:
        r = np.min([cx, s1 - cx, cy, s2 - cy])

    def dist(x1, y1, x2, y2):
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    v = np.zeros(shp)
    for x in range(cx - r, cx + r):
        for y in range(cy - r, cy + r):
            if dist(cx, cy, x, y) <= r * r:
                v[x][y] = val(x, y)

    return v


def column_mask(shp, r=None, center=None, val=lambda x, y: 1):
    """Generate a 3D binary mask

    :param shp: Shape tuple
    :param r: leave to None to compute automatically
    :param center: leave to None to compute automatically
    :return:
    """
    assert len(shp) == 3
    s1, s2, s3 = shp

    col = np.empty(shp)
    col[..., 0] = column_mask_2d((s1, s2), r, center, val)
    for k in range(1, s3):
        col[..., k] = col[..., 0]

    return col
