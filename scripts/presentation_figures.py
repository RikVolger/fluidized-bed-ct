import h5py
import numpy as np
from cycler import cycler
from pathlib import Path
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

# increase all font sizes from the original 10 pt. For presentation use.
plt.rc('font', size=14)

blues9 = [
    '#3e3333',
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

tud_cycler = cycler(color=[
    "#00a6d6",
    "#c3312f",
    "#00a390",
    "#f6d37d",
    "#eb7246",
    "#017188"
])

filename = Path("D:\\XRay\\2024-06-13 Rik\\preprocessed_single-source_70lmin_22Hz\\recon_loss_init-ones.hdf5")
recon_hdf5 = h5py.File(filename, 'r')
reconstruction = recon_hdf5['reconstruction']
volume = {
    'side': reconstruction.attrs.get('volume_side'),
    'height': reconstruction.attrs.get('volume_height')
}

voxel_size = reconstruction.attrs.get('voxel_size')
hline = int(volume['height'] / voxel_size / 2)
vline = int(volume['side'] / voxel_size / 2)
horizontal_plane = reconstruction[:, :, hline]
vertical_plane = np.rot90(reconstruction[:, vline, :])

# Planes plot
fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(6, 4))
# Vertical plane
axs[0].imshow(vertical_plane, cmap=blues9_map, aspect='equal', vmin=0, vmax=0.5)
axs[0].axis('off')
axs[0].set_title("Vertical plane")
# Horizontal plane
# Store axes handle for colorbar.
im = axs[1].imshow(horizontal_plane, cmap=blues9_map, aspect='equal', vmin=0, vmax=0.5)
axs[1].axis('off')
axs[1].set_title("Horizontal plane")
# Create a colorbar to the right of subplots.
cb = fig.colorbar(im, ax=axs, location="right", aspect=10, orientation='vertical', label='Gas fraction')
fig.suptitle("Cross-sections of reconstruction")

save_location = Path("D:\XRay\output\presentation")
print(f"Saving image {save_location / 'recon-slices'}")
fig.savefig(save_location / "recon_slices.png", dpi=300)
fig.savefig(save_location / "recon_slices.svg", transparent=True)

# losses plot
fig, ax = plt.subplots(1, 1, figsize=(4, 3), layout='constrained')
ax.plot(reconstruction.attrs.get('loss'))
ax.set_title("Loss progression")
ax.set_ylabel("Loss (AU)")
ax.set_xlabel("Iterations")

# axial holdup plot
fig, ax = plt.subplots(1, 1, figsize=(4, 3), layout='constrained')
ax.plot(vertical_plane[hline, :])
ax.set_title("Axial holdup profile")
ax.set_ylabel("Gas fraction")

fig, ax = plt.subplots(1, 1, figsize=(4, 3), layout='constrained')
ax.plot(horizontal_plane[:, vline])
ax.set_title("Axial holdup profile")
ax.set_ylabel("Gas fraction")

plt.show()
