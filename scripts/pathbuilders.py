from pathlib import Path


def pathbuilder(root, parts):
    # return root and then the parts joined by underscores.
    # Ignores None or empty (falsey) entries.
    return Path(root, "_".join([part for part in parts if part]))


def concpathbuilder(root, substance, concentration, flowrate):
    return pathbuilder(root,
                       ["preprocessed",
                        substance,
                        concentration,
                        flowrate])


def fullpathbuilder(root, substance, concentration):
    return pathbuilder(root, ["preprocessed", substance, concentration, "0lmin"])


def emptypathbuilder(root):
    # return pathbuilder(root, ["preprocessed", "Empty", framerate])
    return pathbuilder(root, ["preprocessed", "Empty"])


def extensionless_filename(
        loss: bool = False,
        volume: bool = False,
        resolution: bool = False,
        mask: bool = False,
        bhc: bool = False,
        init: str = "flat",
        recon_size: dict = None,
        voxel_size: float = None,
        mask_size: float = None) -> str:
    """Build filename for the requested savefile."""
    filenameparts = ["recon"]
    if loss:
        filenameparts += ["loss"]
    if volume:
        filenameparts += [f"volume-{recon_size['side']}x{recon_size['height']}"]
    if resolution:
        filenameparts += [f"resolution-{voxel_size}cm"]
    if mask:
        filenameparts += [f"mask-{mask_size}cm"]
    if bhc:
        filenameparts += ["bh-corrected"]
    if init != "flat":
        filenameparts += [f"init-{init}"]

    filename = "_".join(filenameparts)

    return filename


def mat_filename(
        loss: bool = False,
        volume: bool = False,
        resolution: bool = False,
        mask: bool = False,
        bhc: bool = False,
        init: str = "flat",
        recon_size: dict = None,
        voxel_size: float = None,
        mask_size: float = None) -> str:
    """Create filename with mat extension for savemat. Relies mainly on `extensionless_filename()`"""
    filename = extensionless_filename(
        loss,
        volume,
        resolution,
        mask,
        bhc,
        recon_size,
        voxel_size,
        mask_size) + ".mat"

    return filename


def pkl_filename(
        loss: bool = False,
        volume: bool = False,
        resolution: bool = False,
        mask: bool = False,
        bhc: bool = False,
        init: str = "flat",
        recon_size: dict = None,
        voxel_size: float = None,
        mask_size: float = None) -> str:
    """Create filename with pkl extension for pickle. Relies mainly on `extensionless_filename()`"""
    filename = extensionless_filename(
        loss,
        volume,
        resolution,
        mask,
        bhc,
        recon_size,
        voxel_size,
        mask_size) + ".pkl"

    return filename


def hdf5_filename(
        loss: bool = False,
        volume: bool = False,
        resolution: bool = False,
        mask: bool = False,
        bhc: bool = False,
        init: str = "flat",
        recon_size: dict = None,
        voxel_size: float = None,
        mask_size: float = None) -> str:
    """Create filename with hdf5 extension. Relies mainly on `extensionless_filename()`"""
    filename = extensionless_filename(
        loss,
        volume,
        resolution,
        mask,
        bhc,
        init,
        recon_size,
        voxel_size,
        mask_size) + ".hdf5"

    return filename
