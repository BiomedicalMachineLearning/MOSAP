import glob
import io
from pathlib import Path
from typing import Any, Union, Optional  # Meta
from typing import Iterable, Sequence, Mapping, MutableMapping  # Generic ABCs
from typing import Tuple, List
# import tifffile
import re
import logging
from mosap.mosadata import MOSADATA
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# default

dpi=300
ax_pad = 10
label_fontdict = {'size': 7}
title_fontdict = {'size': 10}
cbar_inset = [1.02, 0, .0125, .96]
cbar_titel_fontdict = {'size': 7}
cbar_labels_fontdict = {'size': 7}

def scatter_cell_plot(mosadata:MOSADATA,sample, coordx:str, coordy:str, color_type:str, figsize=(15,15), size=1.0, title=None, color_mapper=None):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    cell_type_color_mapper = sns.color_palette("Set2", len(mosadata.obs[sample][color_type].unique()))
    if color_mapper is None: 
        color_mapper = dict()
        for index, color in enumerate(cell_type_color_mapper):
    #         print(color)
            color_mapper[sorted(list(mosadata.obs[sample][color_type].unique()))[index]] = color
    
    legend_patches = list()
    for key, value in color_mapper.items():
        tmp_patch = mpatches.Patch(color=value, label=key)
        legend_patches.append(tmp_patch)

    for index, row in mosadata.obs[sample].iterrows():
        axs.scatter(row[coordx],row[coordy],
                    color=color_mapper[row[color_type]],
                    linewidth=1.0)
#     axs.imshow(rotated_image)
    if title is None:
        title = color_type
        
    axs.legend(handles=legend_patches,
               bbox_to_anchor=(0.97, 1.0),
               loc=2, title=title, prop={'size': 8})
    return axs, fig

def extract_physical_dimension(self, ome_tiff_path):
        """ A function to load the original OME tiff to extract micron resolution and pixel conversion"""
        """ return two dictionaries: one for unit conversion and the other for channel2name"""

        import xml.etree.ElementTree
        tiff_image = tifffile.TiffFile(ome_tiff_path)
        omexml_string = tiff_image.pages[0].description
        root = xml.etree.ElementTree.parse(io.StringIO(omexml_string))
        namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        channels = root.findall('ome:Image[1]/ome:Pixels/ome:Channel', namespaces)
        channel_names = [c.attrib['Name'] for c in channels]
        resolution = root.findall('ome:Image[1]/ome:Pixels', namespaces)
        attribute = resolution[0]
        
        resolution_unit = dict()
        resolution_unit['original_X_micron'] = float(attribute.attrib['SizeX']) * float(attribute.attrib['PhysicalSizeX'])
        resolution_unit['original_Y_micron'] = float(attribute.attrib['SizeY']) * float(attribute.attrib['PhysicalSizeY'])
        resolution_unit['original_X_pixel'] = int(attribute.attrib['SizeX']) 
        resolution_unit['original_Y_pixel'] = int(attribute.attrib['SizeY']) 
        return resolution_unit, channel_names

def convert_micron2pixel(x_micron, micron_dim, scale_dim):
    """ convert the annotation box coordinate from micron unit to pixel unit
    
    ox_px_coord = convert_micron2pixel(ox_coords, 6276.93, im_demo.shape[1])
    oy_px_coord = convert_micron2pixel(oy_coords, 15235.53, im_demo.shape[0])
    """
    return (x_micron*scale_dim/micron_dim)

def convert_unit_micron2pixel(mosadata:MOSADATA, sample, original_width_micron, original_height_micron, 
                                  X_col:Optional[str]=None,Y_col:Optional[str]=None):
        """ Running the conversion of the unit from micron unit to pixel image """
        """ In case you do not have the conversion unit please use the original OME.tiff extract_physical_dimension"""
        if mosadata.original_coord_unit != mosadata.ref_image_unit:
            logging.warning(f'Converting {mosadata.original_coord_unit} to {mosadata.ref_image_unit}')
            
        if X_col:
            mosadata.obs[sample]['X_px'] = convert_micron2pixel(mosadata.obs[sample][X_col], original_width_micron, 
                                                      mosadata.obs[sample].ref_image.shape[1])
        else:
            mosadata.obs[sample]['X_px'] = convert_micron2pixel(mosadata.obs[mosadata.__centroid_X], original_width_micron, 
                                                      mosadata.obs[sample].ref_image.shape[1])
        
        if Y_col:
            mosadata.obs[sample]['Y_px'] = convert_micron2pixel(mosadata.obs[Y_col], original_height_micron, 
                                                      mosadata.obs[sample].ref_image.shape[0])
        else:
            mosadata.obs[sample]['Y_px'] = convert_micron2pixel(mosadata.obs[mosadata.__centroid_Y], original_height_micron, 
                                                      mosadata.obs[sample].ref_image.shape[0])
        logging.warning(f'X_px and Y_px are added to meta_vars')


###
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from matplotlib.colors import Normalize, NoNorm, to_rgba, to_rgb, Colormap, ListedColormap
from matplotlib.cm import ScalarMappable

def savefig(fig, save):
    # if only filename is given, add root_fig, convenient to save plots less verbose.
#     if save == os.path.basename(save):
#         save = os.path.join(plt.rcParams['savefig.directory'], save)
    fig.savefig(save)
    print(f'Figure saved at: {save}')
# the script below implemented the network visualisation using plotting script inspired by ATHENA plotting function
def custom_spatial(mosadata:MOSADATA, sample: str, attr: str, *, mode: str = 'scatter', node_size: float = 4, coordinate_keys: list = ['x', 'y'],
            mask_key: str = 'cellmasks', graph_key: str = 'knn', edges: bool = False, edge_width: float = .5,
            edge_color: str = 'black', edge_zorder: int = 2, background_color: str = 'white', ax: plt.Axes = None,
            norm=None, set_title: bool = True, cmap=None, cmap_labels: list = None, cbar: bool = True,
            cbar_title: bool = True, show: bool = True, save: str = None, tight_layout: bool = True):
    """Various functionalities to visualise samples.
   the graph representation of the sample can be overlayed by setting ``edges=True`` and specifing the ``graph_key`` as in ``mosadata.G[sample][graph_key]``.
    Args:
        mosadata: MOSADATA object
        sample: sample to visualise
        attr: feature to visualise
        mode: {scatter, mask}. In `scatter` mode, observations are represented by their centroid, in `mask` mode by their actual segmentation mask
        node_size: size of the node when plotting the graph representation
        coordinate_keys: column names in MOSADATA.obs[sample] that indicates the x and y coordinates
        mask_key: key for the segmentation masks when in `mask` mode
        graph_key: which graph representation to use
        edges: whether to plot the graph or not
        edge_width: width of edges
        edge_color: color of edges as string
        edge_zorder: z-order of edges
        background_color: background color of plot
        ax: axes object in which to plot
        norm: normalisation instance to normalise the values of `attr`
        set_title: title of plot
        cmap: colormap to use
        cmap_labels: colormap labels to use
        cbar: whether to plot a colorbar or not
        cbar_title: whether to plot the `attr` name as title of the colorbar
        show: whether to show the plot or not
        save: path to the file in which the plot is saved
        tight_layout: whether to apply tight_layout or not.
    Examples:
        .. code-block:: python
            so = sh.dataset.imc()
            sh.pl.spatial(so, 'slide_7_Cy2x2', 'meta_id', mode='mask')
            sh.pl.spatial(so, 'slide_7_Cy2x2', 'meta_id', mode='scatter', edges=True)
    """
    # get attribute information
    data = None  # pd.Series/array holding the attr for colormapping
    colors = None  # array holding the colormappig of data

    # try to fetch the attr data
    if attr:
        if attr in mosadata.obs[sample].columns:
            data = mosadata.obs[sample][attr]
        elif attr in mosadata.X[sample].columns:
            data = mosadata.X[sample][attr]
        else:
            raise KeyError(f'{attr} is not an column of X nor obs')
    else:
        colors = 'black'
        cmap = 'black'
        cbar = False

    # broadcast if necessary
    _is_categorical_flag = is_categorical_dtype(data)

    loc = mosadata.obs[sample][coordinate_keys].copy()

    # set colormap
    if cmap is None:
#         print('The function did not provide cmap')
        cmap, cmap_labels = get_cmap(mosadata, attr, data)
#         print(cmap_labels)
    elif isinstance(cmap, str) and colors is None:
#         print(cmap)
        cmap = plt.get_cmap(cmap)
#     print('cmap', cmap)
    # normalisation
    if norm is None:
        if _is_categorical_flag:
            norm = NoNorm()
        else:
            norm = Normalize()

    # generate figure
    if ax:
        fig = ax.get_figure()
        show = False  # do not automatically show plot if we provide axes
    else:
        fig, ax = plt.subplots(dpi=dpi)
        ax.set_aspect('equal')

    # compute edge lines
    if edges:
        g = mosadata.G[sample][graph_key]
        e = np.array(g.edges, dtype=type(loc.index.dtype))

        tmp1 = loc.loc[e.T[0]]
        tmp2 = loc.loc[e.T[1]]

        x = np.stack((tmp1[coordinate_keys[0]], tmp2[coordinate_keys[0]]))
        y = np.stack((tmp1[coordinate_keys[1]], tmp2[coordinate_keys[1]]))

        # we have to plot sequentially nodes and edges, this takes a bit longer but is more flexible
        im = ax.plot(x, y, linestyle='-', linewidth=edge_width, marker=None, color=edge_color, zorder=edge_zorder)

    # plot
    if mode == 'scatter':
#         print('Mode scatter')
        # convert data to numeric
        data = np.asarray(
            data) if data is not None else None  # categorical data does not work with cmap, therefore we construct an array
        _broadcast_to_numeric = not is_numeric_dtype(data)  # if data is now still categorical, map to numeric

        if _broadcast_to_numeric and data is not None:
            if attr in mosadata.uns['cmap_labels']:
                cmap_labels = mosadata.uns['cmap_labels'][attr]
#                 print(cmap_labels)
                encoder = {value: key for key, value in cmap_labels.items()}  # invert to get encoder
#                 print(encoder)
            else:
                uniq = np.unique(data)
                encoder = {i: j for i, j in zip(uniq, range(len(uniq)))}
                cmap_labels = {value: key for key, value in encoder.items()}  # invert to get cmap_labels

            data = np.asarray([encoder[i] for i in data])

        # map to colors
        if colors is None:
            no_na = data[~np.isnan(data)]
            _ = norm(no_na)  # initialise norm with no NA data.
#             print(norm(data))
            colors = cmap(norm(data))
#             print('Check_point cmap2colors',colors)

        im = ax.scatter(loc[coordinate_keys[0]], loc[coordinate_keys[1]], s=node_size, c=colors, zorder=2.5)
        ax.set_facecolor(background_color)

    else:
        raise ValueError(f'Invalide plotting mode {mode}')

    # add colorbar
    if cbar:
        title = attr
        if cbar_title is False:
            title = ''
        make_cbar(ax, title, norm, cmap, cmap_labels)

    # format plot
    ax_pad = min(loc[coordinate_keys[0]].max() * .05, loc[coordinate_keys[1]].max() * .05, 10)
    ax.set_xlim(loc[coordinate_keys[0]].min() - 0.05, loc[coordinate_keys[0]].max() + 0.05)
    ax.set_ylim(loc[coordinate_keys[1]].min() - 0.05, loc[coordinate_keys[1]].max() + 0.05)
#     ax.legend(prop=dict(size=30))
    ax.set_xticks([]);
    ax.set_yticks([])
#     ax.set_xlabel('spatial x', label_fontdict)
#     ax.set_ylabel('spatial y', label_fontdict)
    ax.set_aspect(1)
    if set_title:
        title = f'{sample}' if cbar else f'{sample}, {attr}'
        ax.set_title(title, {'size': 15})

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)
    

def get_cmap(mosadata, attr: str, data):
    '''
    Return the cmap and cmap labels for a given attribute if available, else a default
    mosadata: extract the cmaps instance from uns
    ----------
    
    '''

    # TODO: recycle cmap if more observations than colors
    cmap, cmap_labels = None, None
    if attr in mosadata.uns['cmaps'].keys():
        cmap = mosadata.uns['cmaps'][attr]
    elif is_categorical_dtype(data):
        cmap = mosadata.uns['cmaps']['category']
    else:
        cmap = mosadata.uns['cmaps']['default']
#     print(attr)
#     print(so.uns['cmap_labels'])
    if attr in mosadata.uns['cmap_labels'].keys():
        cmap_labels = mosadata.uns['cmap_labels'][attr]
#     print(cmap, cmap_labels)
    return cmap, cmap_labels
###

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
    # NOTE: The Linercolormap ticks can only be set up to the number of colors. Thus if we do not have linear, sequential
    # values [0,1,2,3] in the cmap_labels dict this will fail. Solution could be to remap.

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