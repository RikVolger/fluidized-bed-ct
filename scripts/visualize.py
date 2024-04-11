import yaml
import itertools
import numpy as np
import pyqtgraph as pq
import scipy.io as scio
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from scripts.pathbuilders import concpathbuilder, mat_filename

blues9 = [
    '#f7fbff',
    '#deebf7',
    '#c6dbef',
    '#9ecae1',
    '#6baed6',
    '#4292c6',
    '#2171b5',
    '#08519c',
    '#08306b'
]

blues9_map = ListedColormap(blues9, name='blues9')


def load_planes(vol, framerate, root, conc, mat_kwargs, f):
    """Load horizontal and vertical plane from reconstruction file."""
    conc_path = concpathbuilder(root, conc, f, framerate)
    filename = mat_filename(**mat_kwargs)

    print(f"Loading from {conc_path / filename}")
    recon_mat = scio.loadmat(conc_path / filename)

    reconstruction = recon_mat['reconstruction']
    horizontal_plane = reconstruction[:, :, int(vol['height'] / recon_mat['voxel_size'] / 2)]
    vertical_plane = reconstruction[:, int(vol['side'] / recon_mat['voxel_size'] / 2), :]

    return horizontal_plane, vertical_plane


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

if "volume" in scans.keys() and True:
    vol_scans = scans['volume']
    concentrations = vol_scans['concentrations']
    volumes = vol_scans['recon_volumes']
    flowrates = vol_scans['flowrates']
    framerate = vol_scans['framerate']
    for day in vol_scans['measurements']:
        for conc in day['concentrations']:
            fig = plt.figure(layout='constrained', figsize=(8, 10))
            subfigs = fig.subfigures(6, 3, wspace=0.07, width_ratios=[3, 3, 1])
            # figure counters
            row = 0
            # for volume in volumes
            for vol in volumes:
                mat_kwargs = {
                    'loss': True,
                    'volume': True,
                    'recon_size': vol
                }
                col = 0
                for f in flowrates:
                    # load .mat file - recon
                    horizontal_plane, vertical_plane = load_planes(vol, framerate, day['root'], conc, mat_kwargs, f)
                    # in subfigs [i]
                    # plot (2, 1) subplots (horizontal & vertical plane)
                    axs = subfigs[row, col].subplots(1, 2)
                    # plot imshow of horizontal slice
                    axs[0].imshow(horizontal_plane, cmap=blues9_map, vmax=0.5)
                    axs[0].set_title("Horizontal", fontsize='medium')
                    axs[0].axis('off')
                    im = axs[1].imshow(vertical_plane, cmap=blues9_map, vmax=0.5)
                    axs[1].set_title("Vertical", fontsize='medium')
                    axs[1].axis('off')
                    cb = subfigs[row, col].colorbar(im, ax=axs, location="right", aspect=10)
                    cb.ax.tick_params(labelsize='xx-small')
                    subfigs[row, col].suptitle(f)
                    col += 1
                    # plot imshow of vertical slice
                    # i += 1
                    # legend on last subplot, outside box.
                # plot (1, 2) subplots of losses
                axs = subfigs[row, col].subplots(1, 1)
                axs.text(.5, .5, f"Sides: {vol['side']} cm\nHeight: {vol['height']} cm", ha="center", va="center")
                axs.axis('off')
                row += 1
            fig.suptitle(conc, fontsize="xx-large")
            fig.savefig(out_folder / "volume" / f"{conc}.png", dpi=300)
            fig.savefig(out_folder / "volume" / f"{conc}.pdf", dpi=900)
            # plt.show()


if "mask" in scans.keys():
    mask_scans = scans['mask']
    concentrations = mask_scans['concentrations']
    masks = mask_scans['mask_sizes']
    vol = mask_scans['recon_volume'][0]
    flowrates = mask_scans['flowrates']
    framerate = mask_scans['framerate']
    for day in mask_scans['measurements']:
        for conc in day['concentrations']:
            fig = plt.figure(layout='constrained', figsize=(8, 10))
            subfigs = fig.subfigures(6, 3, wspace=0.07, width_ratios=[3, 3, 1])
            # figure counters
            row = 0
            # for volume in volumes
            for mask in masks:
                mat_kwargs = {
                    'loss': True,
                    'mask': True,
                    'mask_size': mask
                }
                col = 0
                for f in flowrates:
                    # load .mat file - recon
                    
                    horizontal_plane, vertical_plane = load_planes(vol, framerate, day['root'], conc, mat_kwargs, f)
                    # in subfigs [i]
                    # plot (2, 1) subplots (horizontal & vertical plane)
                    axs = subfigs[row, col].subplots(1, 2)
                    # plot imshow of horizontal slice
                    axs[0].imshow(horizontal_plane, cmap=blues9_map, vmax=0.45, aspect='equal')
                    axs[0].set_title("Horizontal", fontsize='medium')
                    axs[0].axis('off')
                    im = axs[1].imshow(vertical_plane.T[::-1, :], cmap=blues9_map, vmax=0.45, aspect='equal')
                    axs[1].set_title("Vertical", fontsize='medium')
                    axs[1].axis('off')
                    cb = subfigs[row, col].colorbar(im, ax=axs, location="right", aspect=10)
                    cb.ax.tick_params(labelsize='xx-small')
                    subfigs[row, col].suptitle(f)
                    col += 1
                    # plot imshow of vertical slice
                    # i += 1
                    # legend on last subplot, outside box.
                # plot (1, 2) subplots of losses
                axs = subfigs[row, col].subplots(1, 1)
                axs.text(.5, .5, f"Mask: {mask} cm", ha="center", va="center")
                axs.axis('off')
                row += 1
            fig.suptitle(conc, fontsize="xx-large")
            print(f"Saving image {out_folder / 'mask' / conc}")
            fig.savefig(out_folder / "mask" / f"{conc}.png", dpi=300)
            fig.savefig(out_folder / "mask" / f"{conc}.pdf", dpi=900)
            # plt.show()
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


