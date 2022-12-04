import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, NoNorm, to_rgba, to_rgb, Colormap, ListedColormap
import seaborn as sns
import pandas as pd
from typing import Optional, Union, Mapping, Sequence, Any

import numpy as np
import os
import napari

# %% figure constants
from .utils import DPI, make_cbar, savefig
from .mosap import MultiOmicsSpatial

def napari_raw_viewer(mos:MultiOmicsSpatial, spl: str, attrs: list, censor: float = .95, 
	add_masks='cellmasks', attrs_key='target', index_key:str='fullstack_index'):
    """Starts interactive Napari viewer to visualise raw images and explore samples.

    """
    attrs = list(make_iterable(attrs))
    var = mos.var[spl]
    index = var[var[attrs_key].isin(attrs)][index_key]
    names = var[var[attrs_key].isin(attrs)][attrs_key]

    img = mos.get_image(spl)[index,]
    if censor:
        for j in range(img.shape[0]):
            v = np.quantile(img[j,], censor)
            img[j, img[j] > v] = v
            img[j,] = img[j,] / img[j,].max()

    viewer = napari.Viewer()
    viewer.add_image(img, channel_axis=0, name=names)
    if add_masks:
        add_masks = make_iterable(add_masks)
        for m in add_masks:
            mask = mos.masks[spl][m]
            labels_layer = viewer.add_labels(mask, name=m)

