from fbrct.plotting import plt
import numpy as np


def plot_projs(
    projs: np.ndarray,
    pixel_width: float = None,
    pixel_height: float = None,
    row_start: int = None,
    row_end: int = None,
    pause=1.0,
    figsize=None,
    with_colorbar=False,
    vmin=0.0,
    vmax=3.5,
    subplot_row=False
):
    assert projs.ndim == 3

    if row_start is None:
        row_start = 0
    if row_end is None:
        row_end = projs.shape[1]

    if subplot_row:
        # TODO Add some documentation. very hard to figure out what this piece is doing.
        subplt_shape = (
            (1, len(projs),) if projs.shape[1] > projs.shape[2]
            else (len(projs), 1)
        )
    else:
        subplt_shape = (
            (len(projs),) if projs.shape[2] > projs.shape[1] else (len(projs), 1)
        )

    fig, axs = plt.subplots(*subplt_shape, figsize=figsize)

    def x_fmt(x, y):
        return "{:.0f}".format(x * pixel_width)

    def y_fmt(x, y):
        return "{:.0f}".format(x * pixel_height + row_start * pixel_height)

    for i, (ax, p) in enumerate(zip(axs, projs)):
        # ax.set_title(f"Camera {i+1}",
        #                      rotation='vertical',
        #                      x=-.15, y=.25)
        if subplot_row:
            ax.set_title(f"Detector {i+1}")
        else:
            ax.set_ylabel(f"Detector {i+1}")


        im = ax.imshow(
            p[row_start:row_end],
            vmin=vmin,
            vmax=vmax,
            cmap='gray_r'
        )

        ax.xaxis.set_major_locator(plt.MultipleLocator(100))
        ax.yaxis.set_major_locator(plt.MultipleLocator(100))
        # ax.invert_yaxis()
        ax.set_aspect("equal")

    axs.flat[0].set(ylabel="(pixels)")
    for ax in axs.flat:
        ax.set(xlabel="(pixels)")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    if with_colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.85, 0.12, 0.01, 0.80])
        fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout(w_pad=1.0, h_pad=.5)
    plt.subplots_adjust(
        left=0.119,
        bottom=0.038,
        right=.995,
        top=.962,
        wspace=0.06,
        hspace=0.20)
    # plt.savefig("plot_projs.pdf")
    if pause is not None:
        plt.pause(pause)


def plot_nicely(
    ax,
    im,
    vmin,
    vmax,
    vox_sz,
    cmap,
    geoms=None,
    with_lines=True,
    mask=None,
    mask_cmap=None,
    xlabel=None,
    ylabel="y",
    spine_color=None,
    spine_linewidth=2
):
    import matplotlib.ticker as tick

    extent = (
        -0.5 - (im.shape[0] // 2),
        (im.shape[0] // 2) - 0.5,
        -0.5 - (im.shape[1] // 2),
        (im.shape[1] // 2) - 0.5,
    )
    im = np.flipud(np.swapaxes(im, 0, 1))
    print(im.shape)
    ax.imshow(
        im,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        origin="lower",
        extent=extent,
        interpolation="none",
    )
    ax.set_aspect("equal")
    if spine_color is not None:
        plt.setp(ax.spines.values(), color=spine_color)
        plt.setp([ax.get_xticklines(),
                  ax.get_yticklines()], color=spine_color)
        [i.set_linewidth(spine_linewidth) for i in ax.spines.values()]

    ax.set_xlim(xmin=-im.shape[1] // 2, xmax=im.shape[1] // 2)
    ax.set_ylim(ymin=-im.shape[0] // 2, ymax=im.shape[0] // 2)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0 / vox_sz[1]))  # cm
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0 / vox_sz[1] / 10))  # mm
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0 / vox_sz[0]))  # cm
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.0 / vox_sz[0] / 10))  # mm

    if xlabel is not None:
        ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel(f"${ylabel}$ (cm)")

    def x_fmt(x, y):
        return "{:.1f}".format(x * vox_sz[1])  # millis

    def y_fmt(x, y):
        return "{:.1f}".format(x * vox_sz[0])  # millis

    ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

    if mask is not None:
        ax.imshow(
            mask,
            alpha=1,
            cmap=mask_cmap,
            origin="lower",
            extent=extent,
            interpolation="none",
        )

    if with_lines:
        for i, (g, ls) in enumerate(zip(geoms, ("--", "--", "--"))):
            # if i == 1:
            if True:
                try:
                    S = g.tube_position[:2]
                    D = g.detector_position[:2]
                except:
                    S = g[:2]
                    D = g[3:5]

                ax.axline(
                    S / vox_sz[:2],
                    D / vox_sz[:2],
                    color="white",
                    linewidth=1,
                    ls=ls,
                    label=f"Source {i}",
                    # alpha=0.5
                )

            if False:
                SD = D - S
                SC = s_center - S  # source-sphere vector

                # unit vector from sphere to source-det line
                CZ = np.inner(SD, SC) / np.inner(SD, SD) * SD - SC
                nCZ = CZ / np.linalg.norm(CZ)

                # outer point on the ball with radius r, being hit by SD
                P = s_center + s_rad * np.array(vox_sz[:2]) * nCZ

                ax.axline(
                    S / vox_sz[:2],
                    P / vox_sz[:2],
                    color="white",
                    linewidth=1.0,
                    ls=ls,
                    label=f"p1",
                )

                P = s_center - s_rad * np.array(vox_sz[:2]) * nCZ
                ax.axline(
                    S / vox_sz[:2],
                    P / vox_sz[:2],
                    color="white",
                    linewidth=1.0,
                    ls=ls,
                    label=f"p2",
                )
