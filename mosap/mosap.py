import pandas as pd
import networkx as nx
from skimage import io
import seaborn as sns
from scipy.sparse import issparse
import numpy as np
from typing import Optional, Union, Mapping, Sequence, Any
import os
import copy

import h5py
## abstract class for modification and storing network for heterogeneity analysis
from spatialOmics import SpatialOmics

#light weight object

class MultiOmicsSpatial:
	# to create a blank object, inherit from SpatialOmics data
    cmaps = {'default': sns.color_palette('Reds', as_cmap=True), 'category': sns.color_palette('tab10', as_cmap=True)}

    def __init__(self):
        self.h5py_file = 'MultiSpatialOmics.h5py'  # backend store

        self.spatial_transcriptomic = SpatialOmics()  # container for observation level features
        self.spatial_proteomic = SpatialOmics()
        self.obs = {}  # container for meta data at patient level
        self.obsm = {}  # container for multidimensional observation level features

        self.registration_models = {}
        #--------------------------------------------------------------

    def __init__(self, spatial_transcriptomic:SpatialOmics, spatial_proteomic:SpatialOmics, *args: Any, **kwargs: Any):
        self.spatial_transcriptomic = spatial_transcriptomic  # container for observation level features
        self.spatial_proteomic = spatial_proteomic
        self.obs = {}  # container for meta data at patient level
        self.obsm = {}  # container for multidimensional observation level features

        self.registration_models = {}

   	# def __init__(self, path, viewer=None):
   	# 	"""Initialize new instance and launch viewer
       #
       #  Args:
       #      path (str): path to adata, transcripts, and/or images or .h5ad path
       #      viewer (napari.viewer.Viewer): If None, napari will be launched.
       #  """


    def __repr__(self):
        """Function to return representation of the object"""
        l_transcript = [len(self.spatial_transcriptomic.obs[i]) for i in self.spatial_transcriptomic.obs]
        l_protein = [len(self.spatial_proteomic.obs[i]) for i in self.spatial_proteomic.obs]

        s = f""" MultiSpatialOmics object with samples from {len(l_transcript)} transcriptomic and {len(l_protein)} from proteomic data """
        return s

    def to_h5py(self, file: str = None) -> None:
        """
        Args:
            file: file to write to, defaults to self.h5py_file
        Returns:
        """
        if file is None:
            file = self.h5py_file

        with h5py.File(file, mode='w') as f:
            f.create_dataset('h5py_file', data=self.h5py_file)

            # obsm
            for spl in self.obsm:
                f.create_dataset(f'obsm/{spl}', data=self.obsm[spl])

            # images
            if self.spatial_transcriptomic:
                for spl in self.spatial_transcriptomic.images:
                    img = self.spatial_transcriptomic.images[spl]
                    f.create_dataset(f'transcriptomic_images/{spl}', data=img)

                # masks
                for spl in self.spatial_transcriptomic.masks:
                    for key in self.spatial_transcriptomic.masks[spl].keys():
                        msk = self.spatial_transcriptomic.masks[spl][key]
                        f.create_dataset(f'transcriptomic_masks/{spl}/{key}', data=msk)

            if self.spatial_proteomic:
                for spl in self.spatial_proteomic.images:
                    img = self.spatial_proteomic.images[spl]
                    f.create_dataset(f'proteomic_images/{spl}', data=img)

                # masks
                for spl in self.spatial_proteomic.masks:
                    for key in self.spatial_proteomic.masks[spl].keys():
                        msk = self.spatial_proteomic.masks[spl][key]
                        f.create_dataset(f'proteomic_masks/{spl}/{key}', data=msk)
            # uns
            # TODO: currently we do not support storing uns to h5py due to datatype restrictions
            if self.uns:
                print(
                    'warning: in the current implementation, the `uns` attribute is not stored to h5py file. Use `to_pickle` instead')

        self.spl.to_hdf(file, 'spl', format="table")

        # var
        for spl in self.var:
            # use pandas function
            self.var[spl].to_hdf(file, f'var/{spl}', format="table")

        # X
        for spl in self.X:
            # use pandas function
            self.X[spl].to_hdf(file, f'X/{spl}', format="table")

        # G
        for spl in self.spatial_proteomic.G:
            for key in self.spatial_proteomic.G[spl]:
                g = self.spatial_proteomic.G[spl][key]
                df = nx.to_pandas_edgelist(g)

                # use pandas function
                df.to_hdf(file, f'G/{spl}/{key}', format="table")
        for spl in self.spatial_transcriptomic.G:
            for key in self.spatial_transcriptomic.G[spl]:
                g = self.spatial_transcriptomic.G[spl][key]
                df = nx.to_pandas_edgelist(g)

                # use pandas function
                df.to_hdf(file, f'G/{spl}/{key}', format="table")
        # obs
        for spl in self.obs:
            # use pandas function
            self.obs[spl].to_hdf(file, f'obs/{spl}', format="table")

        print(f'File `{os.path.basename(file)}` saved to {os.path.abspath(file)}')
        print(f'File size: {os.path.getsize(file) / (1024 * 1024):.2f} MB')