from pathlib import Path


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


def mat_filename(
        loss: bool = False,
        volume: bool = False,
        resolution: bool = False,
        mask: bool = False,
        recon_size: dict = None,
        voxel_size: float = None,
        mask_size: float = None) -> str:
    filenameparts = ["recon"]
    if loss:
        filenameparts += ["loss"]
    if volume:
        filenameparts += [f"volume-{recon_size['side']}x{recon_size['height']}"]
    if resolution:
        filenameparts += [f"resolution-{voxel_size}cm"]
    if mask:
        filenameparts += [f"mask-{mask_size}cm"]

    filename = "_".join(filenameparts) + ".mat"

    return filename
