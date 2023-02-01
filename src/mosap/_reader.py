"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import os
from mosap.mosap import SpatialOmics
from mosap._widget import MultiOmicRegistrationWidget, Transcript_Selection_Widget 
from mosap.utils.file_listing import get_files_in_dir_recursively 

from skimage.io import imread
import napari

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # Load metadata
    if path.endswith("_measurement.txt") or path.endswith("_measurement.csv"):
        return read_metadata_function
    # Allow parent directory or AnnData file to be opened
    # if not (path.endswith(".h5ad") or \
    #     os.path.exists(os.path.join(path, "images"))):
    #     return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function



def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    if isinstance(path, list):
        # Only want RunSummary directory if list of paths is present
        path = path[0]
    # print("Read function with Mosap",path)
    image = get_files_in_dir_recursively(path,'*tif')
    # print(image)
    # 1/0
    image = image[0]
    viewer = napari.current_viewer()
    # 1/0
    # gem = Gemini(path, viewer=viewer)
    # viewer.window.add_dock_widget(MultiOmicRegistrationWidget(viewer),
    #     area='right')
    if os.path.exists(image):
        data = imread(image)
    
    
    # labels layer added in Gemini instance initialization
    print('Loaded folder and created Mosap instance')
    # return [(None,)]
    return [
        (data, {"name": image.split('/')[-1]}, "image"),
    ]

def read_metadata_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    if isinstance(path, list):
        path = path[0]
    
    viewer = napari.current_viewer()
    for w in viewer.window._dock_widgets.values():
        if isinstance(w.widget(), GeminiQWidget):
            w.widget().update_metadata(path)
            break
    return [(None,)]
