import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import SubplotParams
import os
from typing import Optional, Union, Mapping
from typing import Sequence
from pathlib import Path
# support functions for plotting

DPI = 300


label_fontdict = {'size': 7}
title_fontdict = {'size': 10}
cbar_inset = [1.02, 0, .0125, .96]
cbar_titel_fontdict = {'size': 7}
cbar_labels_fontdict = {'size': 7}
root_fig = plt.rcParams['savefig.directory']

# customised colorbar for easy display
# 
def make_cbar(ax, title, norm, cmap, cmap_labels, im=None, prefix_labels=True):
    """Generate a colorbar for the given axes.
    Parameters
    ----------
    ax: Axes
        axes for which to plot colorbar
    title: str
        title of colorbar
    norm:
        Normalisation instance
    cmap: Colormap
        Colormap
    cmap_labels: dict
        colorbar labels
    Returns
    -------
    """
    inset = ax.inset_axes(cbar_inset)
    fig = ax.get_figure()
    if im is None:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=inset)
    else:
        cb = fig.colorbar(im, cax=inset)

    cb.ax.set_title(title, loc='left', fontdict=cbar_titel_fontdict)
    if cmap_labels:
        if prefix_labels:
            labs = [f'{key}, {val}' for key, val in cmap_labels.items()]
        else:
            labs = list(cmap_labels.values())
        cb.set_ticks(list(cmap_labels.keys()))
        cb.ax.set_yticklabels(labs, fontdict=cbar_labels_fontdict)
    else:
        cb.ax.tick_params(axis='y', labelsize=cbar_labels_fontdict['size'])

    # TODO
    def linear_mapping(cmap_labels):
        pass


def savefig(fig, fn:Union[str, Path]):
    # if only filename is given
    if fn == os.path.basename(fn):
        fn = os.path.join(plt.rcParams['savefig.directory'], fn)
    fig.savefig(fn)
    print(f'Figure saved at: {fn}')
