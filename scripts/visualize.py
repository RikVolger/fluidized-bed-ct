import pickle
import h5py
import yaml
import itertools
import numpy as np
import pyqtgraph as pq
import scipy.io as scio
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from scripts.pathbuilders import concpathbuilder, emptypathbuilder, fullpathbuilder, hdf5_filename, mat_filename, pkl_filename

blues9 = [
    '#ef3b2c',
    '#f7fbff',
    '#deebf7',
    '#c6dbef',
    '#9ecae1',
    '#6baed6',
    '#4292c6',
    '#2171b5',
    '#08519c',
    '#08306b',
]

blues9_map = ListedColormap(blues9[::-1], name='blues9')


def load_mat(framerate, root, conc, mat_kwargs, f):
    """Load reconstruction file from drive."""
    conc_path = concpathbuilder(root, conc, f, framerate)
    filename = mat_filename(**mat_kwargs)

    print(f"Loading from {conc_path / filename}")
    recon_mat = scio.loadmat(conc_path / filename)

    return recon_mat


def load_pkl(framerate, root, conc, mat_kwargs, f):
    """Load reconstruction file from drive."""
    conc_path = concpathbuilder(root, conc, f, framerate)
    filename = pkl_filename(**mat_kwargs)

    full_path = conc_path / filename
    print(f"Loading from {full_path}")
    with open(full_path, 'rb') as pkl_file:
        recon_pkl = pickle.load(pkl_file)

    return recon_pkl


def load_hdf5(framerate, root, conc, mat_kwargs, f):
    """Load reconstruction file from drive."""
    if f == "Empty":
        conc_path = emptypathbuilder(root, framerate)
    elif f == "Full":
        conc_path = fullpathbuilder(root, conc, framerate)
    else:
        conc_path = concpathbuilder(root, conc, f, framerate)
    filename = hdf5_filename(**mat_kwargs)

    full_path = conc_path / filename
    print(f"Loading from {full_path}")
    recon_hdf5 = h5py.File(full_path, 'r')

    return recon_hdf5['reconstruction']


def extract_mat_kwargs(setting, mat_kwargs):
    """Extent mat_kwargs with appropriate setting."""
    if 'volume' in mat_kwargs.keys() and mat_kwargs['volume']:
        mat_kwargs['recon_size'] = setting
        return mat_kwargs
    if 'resolution' in mat_kwargs.keys() and mat_kwargs['resolution']:
        mat_kwargs['voxel_size'] = setting
        return mat_kwargs
    if 'mask' in mat_kwargs.keys() and mat_kwargs['mask']:
        mat_kwargs['mask_size'] = setting
        return mat_kwargs
    if 'bhc' in mat_kwargs.keys() and mat_kwargs['bhc']:
        return mat_kwargs
    return mat_kwargs


def setting_text(mat_kwargs, setting):
    """Create setting text for left-most column.
    The layout is based on mat_kwargs, content on setting
    """

    if 'volume' in mat_kwargs.keys() and mat_kwargs['volume']:
        return f"{setting['side']} cm\nx\n{setting['height']} cm"
    if 'resolution' in mat_kwargs.keys() and mat_kwargs['resolution']:
        return f"{setting} cm"
    if 'mask' in mat_kwargs.keys() and mat_kwargs['mask']:
        return f"{setting} cm"
    if 'bhc' in mat_kwargs.keys() and mat_kwargs['bhc']:
        if setting:
            return "BHC"
        return "No BHC"
    return f"{setting}"


def create_setting_figure(
        day: dict,
        conc: str,
        settings: list,
        iterations: int,
        flowrates: list,
        mat_kwargs: dict,
        setting_title: str,
        out_subfolder: str,
        volume: dict = {'side': 20, 'height': 15},
        width_ratios: list = [1, 3, 3, 1],
        figsize: tuple = (8, 10)
        ):
    """Create a figure for the current investigated setting and return figure handle
    
    TODO: More detailed description.
    """
    fig = plt.figure(layout='constrained', figsize=figsize)
    n_rows = max(len(settings), 6)
    subfigs = fig.subfigures(n_rows, len(width_ratios), wspace=0.07, width_ratios=width_ratios)
    # figure counters
    row = 0
    # for volume in volumes
    for s in settings:
        # if volume is investigated, overwrite volume
        if 'volume' in mat_kwargs.keys() and mat_kwargs['volume']:
            volume = s
        mat_kwargs = extract_mat_kwargs(s, mat_kwargs)
        # data placeholder for losses
        volume_losses = np.zeros((len(flowrates), iterations))
        col = 0
        axs = subfigs[row, col].subplots(1, 1)
        axs.text(.5, .5, setting_text(mat_kwargs, s), ha="center", va="center")
        axs.axis('off')
        if row == 0:
            axs.set_title(setting_title, fontsize="medium")
        col += 1
        for i, f in enumerate(flowrates):
            # load .mat file - recon
            # recon_mat = load_mat(framerate, day['root'], conc, mat_kwargs, f)
            # print(volume)
            reconstruction = load_hdf5(framerate, day['root'], conc, mat_kwargs, f)
            voxel_size = reconstruction.attrs.get('voxel_size')
            horizontal_plane = reconstruction[:, :, int(volume['height'] / voxel_size / 2)]
            vertical_plane = np.rot90(reconstruction[:, int(volume['side'] / voxel_size / 2), :])
            # Subplots for the plane images.
            axs = subfigs[row, col].subplots(1, 2)
            axs[0].imshow(horizontal_plane, cmap=blues9_map, vmax=0.9)
            axs[0].axis('off')
            # Store axes handle for colorbar.
            im = axs[1].imshow(vertical_plane, cmap=blues9_map, vmax=0.9)
            axs[1].axis('off')
            # Create a colorbar to the right of subplots.
            cb = subfigs[row, col].colorbar(im, ax=axs, location="right", aspect=10)
            cb.ax.tick_params(labelsize='x-small')

            # Add titles to first row.
            if s == settings[0]:
                subfigs[row, col].suptitle(f)
                axs[0].set_title("Horizontal", fontsize='medium')
                axs[1].set_title("Vertical", fontsize='medium')
            # Store loss progression.
            volume_losses[i, :] = reconstruction.attrs.get('loss')
            col += 1
            del reconstruction

        # Losses subplots.
        axs = subfigs[row, col].subplots(min(2, len(flowrates)), 1, sharex=True)
        if len(flowrates) > 1:
            axs[0].plot(volume_losses[0, :])
            axs[0].set_title(f"{flowrates[0]}", fontsize='small')
            axs[0].tick_params(axis='y', labelsize='x-small')

            axs[1].plot(volume_losses[1, :])
            axs[1].set_title(f"{flowrates[1]}", fontsize='small')
            axs[1].tick_params(axis='both', labelsize='x-small')
        else:
            axs.plot(volume_losses[0, :])
            axs.set_title(f"{flowrates[0]}", fontsize='small')
            axs.tick_params(axis='both', labelsize='x-small')
        # Add title on first row.
        if row == 0:
            subfigs[row, col].suptitle("Losses", fontsize='medium')
        row += 1
    fig.suptitle(conc, fontsize="xx-large")
    print(f"Saving image {out_folder / out_subfolder / conc}")
    fig.savefig(out_folder / out_subfolder / f"{conc}.png", dpi=300)
    fig.savefig(out_folder / out_subfolder / f"{conc}.pdf", dpi=900)
    return fig
    # plt.show()


with open("./visualize_scans.yaml") as scans_yaml:
    scans = yaml.safe_load(scans_yaml)

out_folder = Path(scans['output_folder'])

if 'loss' in scans.keys() and False:
    loss_scans = scans['loss']
    concentrations = loss_scans['concentrations']
    n_iters = loss_scans['iterations']
    framerate = loss_scans['framerate']
    n_flowrates = len(loss_scans['flowrates'])

    loss_fig, axs = plt.subplots(1, n_flowrates, layout="tight", figsize=(7, 3))
    # adjust area of the subplots to use only 75% of space, leaving space for legend
    loss_fig.tight_layout(rect=[0, 0, 0.75, 1])

    flowrate_losses = [None] * n_flowrates
    for i, flowrate in enumerate(loss_scans['flowrates']):
        flowrate_losses[i] = np.zeros((n_iters, len(concentrations)))
        j = 0
        for day in loss_scans['measurements']:
            for conc in day['concentrations']:
                conc_path = concpathbuilder(day['root'], conc, flowrate, framerate)
                filename = mat_filename(loss=True)
                recon = scio.loadmat(conc_path / filename)
                flowrate_losses[i][:, j] = recon['loss']
                j += 1

        # plot on axs[i]
        axs[i].plot(range(n_iters), flowrate_losses[i])
        axs[i].set_title(flowrate)
        axs[i].set_xlabel("Iterations")
        axs[i].set_ylabel("Loss (a.u.)")

        if axs[i] == axs[-1]:
            axs[i].legend(concentrations, loc='center left', bbox_to_anchor=(1.04, 0.5))
    loss_fig.suptitle("Loss evolution", fontsize="x-large")
    # colorbar

    # suptitle of loss evolution

    # show figure
    plt.show()
    # save figure
    loss_fig.savefig(out_folder / "losses.png", dpi=300)
    loss_fig.savefig(out_folder / "losses.pdf", dpi=900)


if "volume_side" in scans.keys() and False:
    voxel_scans = scans['volume_side']
    concentrations = voxel_scans['concentrations']
    volumes = voxel_scans['recon_volumes']
    flowrates = voxel_scans['flowrates']
    framerate = voxel_scans['framerate']
    iterations = scans['iterations']
    for day in voxel_scans['measurements']:
        for conc in day['concentrations']:
            mat_kwargs = {
                'loss': True,
                'volume': True,
                'init': 'ones',
            }
            fig = create_setting_figure(
                day, conc, volumes, iterations, flowrates, mat_kwargs,
                'Volume', 'volume_side', width_ratios=[1, 3, 3, 3, 3, 1],
                figsize=(12, 10))


if "volume_height" in scans.keys() and False:
    voxel_scans = scans['volume_height']
    concentrations = voxel_scans['concentrations']
    volumes = voxel_scans['recon_volumes']
    flowrates = voxel_scans['flowrates']
    framerate = voxel_scans['framerate']
    iterations = scans['iterations']
    for day in voxel_scans['measurements']:
        for conc in day['concentrations']:
            mat_kwargs = {
                'loss': True,
                'volume': True,
            }
            fig = create_setting_figure(
                day, conc, volumes, iterations, flowrates, mat_kwargs,
                'Volume', 'volume_height')


if "mask" in scans.keys() and False:
    mask_scans = scans['mask']
    concentrations = mask_scans['concentrations']
    masks = mask_scans['mask_sizes']
    volume = mask_scans['recon_volume'][0]
    flowrates = mask_scans['flowrates']
    framerate = mask_scans['framerate']
    iterations = scans['iterations']
    for day in mask_scans['measurements']:
        for conc in day['concentrations']:
            mat_kwargs = {
                'loss': True,
                'mask': True,
                'init': 'ones',
            }
            fig = create_setting_figure(
                day, conc, masks, iterations, flowrates, mat_kwargs,
                'Mask', 'mask', volume=volume,
                width_ratios=[1, 3, 3, 3, 3, 3, 3, 1],
                figsize=(14, 10)
            )

if "voxel_size" in scans.keys() and True:
    voxel_scans = scans['voxel_size']
    concentrations = voxel_scans['concentrations']
    volume = voxel_scans['recon_volume'][0]
    flowrates = voxel_scans['flowrates']
    framerate = voxel_scans['framerate']
    iterations = voxel_scans['iterations']
    voxel_sizes = voxel_scans['voxel_sizes']
    for day in voxel_scans['measurements']:
        for conc in day['concentrations']:
            mat_kwargs = {
                'loss': True,
                'resolution': True,
                'init': 'ones',
            }
            fig = create_setting_figure(
                day, conc, voxel_sizes, iterations, flowrates, mat_kwargs,
                'Voxel\nsize', 'voxel_size', volume=volume)

if "BHC" in scans.keys() and False:
    bhc_scans = scans['BHC']
    concentrations = bhc_scans['concentrations']
    volume = bhc_scans['recon_volume'][0]
    flowrates = bhc_scans['flowrates']
    framerate = bhc_scans['framerate']
    iterations = bhc_scans['iterations']
    bhc = [True, False]
    # bhc = [True]
    out_subfolder = 'beam_hardening_correction'
    for day in bhc_scans['measurements']:
        for conc in day['concentrations']:
            mat_kwargs = {
                'loss': True,
                'bhc': True,
            }
            fig = plt.figure(layout='constrained', figsize=(10, 5))
            subfigs = fig.subfigures(2, 5, wspace=0.07, width_ratios=[1, 3, 3, 3, 3])
            # figure counters
            row = 0
            # for volume in volumes
            for setting in bhc:
                mat_kwargs['bhc'] = setting
                # data placeholder for losses
                volume_losses = np.zeros((len(flowrates), iterations))
                col = 0
                axs = subfigs[row, col].subplots(1, 1)
                axs.text(.5, .5, mat_kwargs['bhc'], ha="center", va="center")
                axs.axis('off')
                if row == 0:
                    axs.set_title("BHC", fontsize="medium")
                col += 1
                for i, f in enumerate(flowrates):
                    # load .mat file - recon
                    # recon_mat = load_mat(framerate, day['root'], conc, mat_kwargs, f)
                    reconstruction = load_hdf5(framerate, day['root'], conc, mat_kwargs, f)
                    voxel_size = reconstruction.attrs.get('voxel_size')
                    horizontal_plane = reconstruction[:, :, int(volume['height'] / voxel_size / 2)]
                    vertical_plane = np.rot90(reconstruction[:, int(volume['side'] / voxel_size / 2), :])
                    # Subplots for the plane images.
                    axs = subfigs[row, col].subplots(1, 2)
                    axs[0].imshow(
                        horizontal_plane,
                        cmap=blues9_map,
                        vmax=0.5
                    )
                    axs[0].axis('off')
                    # Store axes handle for colorbar.
                    im = axs[1].imshow(
                        vertical_plane,
                        cmap=blues9_map,
                        vmax=0.5
                    )
                    axs[1].axis('off')
                    # Create a colorbar to the right of subplots.
                    cb = subfigs[row, col].colorbar(im, ax=axs, location="right", aspect=10)
                    cb.ax.tick_params(labelsize='x-small')

                    # Add titles to first row.
                    if row == 0:
                        subfigs[row, col].suptitle(f)
                        axs[0].set_title("Horizontal", fontsize='medium')
                        axs[1].set_title("Vertical", fontsize='medium')
                    # Store loss progression.
                    volume_losses[i, :] = reconstruction.attrs.get('loss')
                    col += 1
                    # As h5py is keeping the file open, delete variable to close it.
                    del reconstruction

                # Losses subplots.
                # axs = subfigs[row, col].subplots(2, 1, sharex=True)
                # axs[0].plot(volume_losses[0, :])
                # axs[0].set_title(f"{flowrates[0]}", fontsize='small')
                # axs[0].tick_params(axis='y', labelsize='x-small')
                # axs[1].plot(volume_losses[1, :])
                # axs[1].set_title(f"{flowrates[1]}", fontsize='small')
                # axs[1].tick_params(axis='both', labelsize='x-small')
                # # Add title on first row.
                # if row == 0:
                #     subfigs[row, col].suptitle("Losses", fontsize='medium')
                row += 1
            fig.suptitle(conc, fontsize="xx-large")
            print(f"Saving image {out_folder / out_subfolder / conc}")
            fig.savefig(out_folder / out_subfolder / f"{conc}.png", dpi=300)
            fig.savefig(out_folder / out_subfolder / f"{conc}.pdf", dpi=900)

if "init" in scans.keys() and False:
    init_scans = scans['init']
    concentrations = init_scans['concentrations']
    volume = init_scans['recon_volume'][0]
    flowrates = init_scans['flowrates']
    framerate = init_scans['framerate']
    iterations = init_scans['iterations']
    init = ["flat", "parabolic"]
    # bhc = [True]
    out_subfolder = 'initialization'
    for day in init_scans['measurements']:
        for conc in day['concentrations']:
            mat_kwargs = {
                'loss': True,
                'init': "flat",
            }
            fig = plt.figure(layout='constrained', figsize=(10, 5))
            subfigs = fig.subfigures(2, 5, wspace=0.07, width_ratios=[1, 3, 3, 3, 3])
            # figure counters
            row = 0
            # for volume in volumes
            for setting in init:
                mat_kwargs['init'] = setting
                # data placeholder for losses
                volume_losses = np.zeros((len(flowrates), iterations))
                col = 0
                axs = subfigs[row, col].subplots(1, 1)
                axs.text(.5, .5, mat_kwargs['init'], ha="center", va="center")
                axs.axis('off')
                if row == 0:
                    axs.set_title("Initialization", fontsize="medium")
                col += 1
                for i, f in enumerate(flowrates):
                    # load .mat file - recon
                    # recon_mat = load_mat(framerate, day['root'], conc, mat_kwargs, f)
                    reconstruction = load_hdf5(framerate, day['root'], conc, mat_kwargs, f)
                    voxel_size = reconstruction.attrs.get('voxel_size')
                    horizontal_plane = reconstruction[:, :, int(volume['height'] / voxel_size / 2)]
                    vertical_plane = np.rot90(reconstruction[:, int(volume['side'] / voxel_size / 2), :])
                    # Subplots for the plane images.
                    axs = subfigs[row, col].subplots(1, 2)
                    axs[0].imshow(
                        horizontal_plane,
                        cmap=blues9_map,
                        vmax=0.5
                    )
                    axs[0].axis('off')
                    # Store axes handle for colorbar.
                    im = axs[1].imshow(
                        vertical_plane,
                        cmap=blues9_map,
                        vmax=0.5
                    )
                    axs[1].axis('off')
                    # Create a colorbar to the right of subplots.
                    cb = subfigs[row, col].colorbar(im, ax=axs, location="right", aspect=10)
                    cb.ax.tick_params(labelsize='x-small')

                    # Add titles to first row.
                    if row == 0:
                        subfigs[row, col].suptitle(f)
                        axs[0].set_title("Horizontal", fontsize='medium')
                        axs[1].set_title("Vertical", fontsize='medium')
                    # Store loss progression.
                    volume_losses[i, :] = reconstruction.attrs.get('loss')
                    col += 1
                    # As h5py is keeping the file open, delete variable to close it.
                    del reconstruction

                # Losses subplots.
                loss_fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(7, 3))
                axs[0].plot(volume_losses[0, :])
                axs[0].set_title(f"{flowrates[0]}", fontsize='small')
                axs[0].tick_params(axis='y', labelsize='x-small')
                axs[0].set_ylim([0, 1000])
                axs[1].plot(volume_losses[1, :])
                axs[1].set_title(f"{flowrates[1]}", fontsize='small')
                axs[1].tick_params(axis='both', labelsize='x-small')
                axs[1].set_ylim([0, 1000])
                # Add title on first row.
                loss_fig.suptitle("Losses", fontsize='medium')
                loss_fig.savefig(out_folder / out_subfolder / f"Loss_{setting}_{conc}.png", dpi=300)
                row += 1
            fig.suptitle(conc, fontsize="xx-large")
            print(f"Saving image {out_folder / out_subfolder / conc}")
            fig.savefig(out_folder / out_subfolder / f"{conc}.png", dpi=300)
            fig.savefig(out_folder / out_subfolder / f"{conc}.pdf", dpi=900)

if "scatter" in scans.keys() and False:
    scatter_scans = scans['scatter']
    out_folder = Path(scatter_scans['out_folder'])
    concentrations = scatter_scans['concentrations']
    volume = scatter_scans['recon_volume'][0]
    flowrates = scatter_scans['flowrates']
    framerate = scatter_scans['framerate']
    iterations = scatter_scans['iterations']
    scatter_cor = scatter_scans['scatter']
    for day in scatter_scans['measurements']:
        for conc in day['concentrations']:
            mat_kwargs = {
                'loss': True,
            }
            fig = create_setting_figure(
                day, conc, scatter_cor, iterations, flowrates, mat_kwargs,
                'Scatter\ncorrection', '', volume=volume,
                width_ratios=[1, 3, 3, 3, 3, 1], figsize=(16, 10))
# framerate = "22Hz"
# # Build array of paths for detailed scans
# paths_detailed = []
# for day, framerate, flowrate in itertools.product(scans['detailed'], scans['framerates'], scans['flowrates']):
#     for conc in day['concentrations']:
#         paths_detailed.append(concpathbuilder(day['root'], conc, flowrate, framerate))

# # [print(str(path)) for path in paths_detailed]
# for path in paths_detailed:
#     recon = scio.loadmat(path / "recon.mat")
#     reconstruction = recon['reconstruction']
#     pq.image(reconstruction.T)
#     plt.figure()
#     plt.imshow(reconstruction[:, :, 370])
#     plt.title(' - '.join(Path(recon['scan_folder'][0]).parts[-1].split('_')[1:]))
#     plt.show()

    # print(recon)
# Load mat into np array

# Plot horizontal cross-sections at 5 heights

# Plot collapsed horizontal cross-section (-R 0 R plot) for 5 heights

# Plot vertical cross-sections in 5 orientations

# For all folders in visualize_scans.yaml
# Compare reconstructed avg holdup with pressure sensor data (parity plot)

# Plot combined collapsed horizontal cross-section - color code data by p_GM (?)


