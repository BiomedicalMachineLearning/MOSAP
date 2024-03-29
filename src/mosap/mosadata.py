import pandas as pd
import networkx as nx
from skimage import io
import seaborn as sns
from scipy.sparse import issparse
from scipy import ndimage
from pandas.api.types import is_numeric_dtype
import numpy as np
import re
import os
import copy
from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes
import h5py
# cosmx data support libraries
import napari
import zarr
import vaex
import dask.array as da
from napari_cosmx.pairing import pair
from sklearn import preprocessing
from napari.utils.colormaps import AVAILABLE_COLORMAPS, label_colormap
from napari.utils.colormaps.standardize_color import transform_color
# import constants for CosMx image data
from mosap import  CosMX_MM_PER_PX, CosMX_PX_PER_MM, CosMX_ALPHA_MM_PER_PX


class MOSADATA:
    cmaps = {
        'default': sns.color_palette('Reds', as_cmap=True),
        'category': sns.color_palette('Set3', as_cmap=True)
    }

    def __init__(self):

        self.graph_engine = 'networkx'
        self.h5py_file = 'MOSADATA.h5py'  # backend store

        self.obs = {}  # container for observation level features
        self.obsm = {}  # container for multidimensional observation level features
        self.spl = pd.DataFrame()  # container for sample level features
        self.splm = {}  # container for multidimensional sample level features
        self.var = {}  # container with variable descriptions of X
        self.X = {}  # container for cell level expression data of each spl
        self.G = {}  # graphs
        self.uns = {}  # unstructured container
        self.uns.update({'cmaps':self.cmaps,
                         'cmap_labels': {}})

        # self.obs_keys = None  # list of observations in self.obs.index
        self.spl_keys = None  # list of samples in self.spl.index

        self.images = {}
    def __init__(self, path:str, viewer:Optional[napari.viewer.Viewer]=None, show_widget=False):
        """
        Initialise napari instance and display viewr
        # args: 
        path (str): path to adata
        viewer (napari.viewer.Viewer): If None, napari will be launched
        """
        assert os.path.exists(path), f"Could not find {path}"
        self.graph_engine = 'networkx'
        self.random_seed = 42  # for reproducibility
        self.pickle_file = ''  # backend store
        self.h5py_file = 'spatialOmics.h5py'  # backend store
        self.folder = path
        self.obs = {}  # container for observation level features
        self.obsm = {}  # container for multidimensional observation level features
        self.spl = pd.DataFrame()  # container for sample level features
        self.splm = {}  # container for multidimensional sample level features
        self.var = {}  # container with variable descriptions of X
        self.X = {}  # container for cell level expression data of each spl
        self.G = {}  # graphs
        self.uns = {}  # unstructured container
        self.uns.update({'cmaps':self.cmaps,
                         'cmap_labels': {}})

        # self.obs_keys = None  # list of observations in self.obs.index
        self.spl_keys = None  # list of samples in self.spl.index

        self.images = {}
        self.masks = {}
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = napari.Viewer()
    
    
    def add_image(self, spl, file, in_memory=True, to_store=False):
        # def add_segmentation(self):
        """Add the cell segmentation image layer
        """        
        assert 'labels' in self.grp.group_keys(), f"labels not found in zarr keys: {self.grp.group_keys()}"
        datasets = self.grp['labels'].attrs["multiscales"][0]["datasets"]
        kernel = np.ones((3,3))
        kernel[1, 1] = -8
        labels = [da.from_zarr(os.path.join(self.folder, "images", "labels"), component=d["path"]).map_blocks(
            # show edges
            lambda x: ndimage.convolve(x, kernel, output=np.uint16),
        ) for d in datasets]
        layer = self.viewer.add_image(labels, contrast_limits=(0, 1), colormap="cyan",
            scale=(self.mm_per_px, self.mm_per_px), translate=self._top_left_mm(), blending="additive",
            rotate=self.rotate)
        layer.opacity = 0.5
        self.segmentation_layer = layer
        layer.name = 'Segmentation'

    def get_image(self, spl):
        """Get the image of a given sample"""
        if spl in self.images:
            return self.images[spl]
        else:
            with h5py.File(self.h5py_file, 'r') as f:
                path = f'images/{spl}'
                if path in f:
                    return f[path][:]
                else:
                    raise KeyError(f'no images exists for {spl}.')

    def __str__(self):
        l = [len(self.obs[i]) for i in self.obs]

        cols_spl = self.spl.columns.to_list()

        cols_obs = set()
        [cols_obs.update((self.obs[i].columns)) for i in self.obs]
        cols_obs = [*cols_obs]

        cols_var = set()
        [cols_var.update((self.var[i].columns)) for i in self.var]
        cols_var = [*cols_var]

        graph_names = set()
        [graph_names.update((self.G[i].keys())) for i in self.G]
        graph_names = [*graph_names]

        s = f"""MOSADATA object with {sum(l)} observations across {len(l)} samples.
                    X: {len(self.X)} samples,
                    spl: {len(self.spl)} samples,
                        columns: {cols_spl}
                    obs: {len(self.spl)} samples,
                        columns: {cols_obs}
                    var: {len(self.var)} samples,
                        columns: {cols_var}
                    G: {len(self.G)} samples,
                        keys: {graph_names}
                    images: {len(self.images)} samples"""
        return s

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.spl)

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
            for spl in self.images:
                img = self.images[spl]
                f.create_dataset(f'images/{spl}', data=img)

            # uns
            # TODO: currently we do not support storing uns to h5py due to datatype restrictions
            if self.uns:
                print(
                    'warning: in the current implementation, the `uns` attribute is not stored to h5py file. Use `to_pickle` instead')

        # we need to write the dataframes outside the context manager because the file is locked
        # spl
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
        for spl in self.G:
            for key in self.G[spl]:
                g = self.G[spl][key]
                df = nx.to_pandas_edgelist(g)

                # use pandas function
                df.to_hdf(file, f'G/{spl}/{key}', format="table")

        # obs
        for spl in self.obs:
            # use pandas function
            self.obs[spl].to_hdf(file, f'obs/{spl}', format="table")

        print(f'File `{os.path.basename(file)}` saved to {os.path.abspath(file)}')
        print(f'File size: {os.path.getsize(file) / (1024 * 1024):.2f} MB')

    @classmethod
    def from_h5py(cls, file=None):
        """

        Args:
            file: h5py file from which to reconstruct SpatialOmics instance
            include_images: Whether to load images into memory
            include_mask: Whether to load masks into memory

        Returns:
            SpatialOmics instance

        """

        mosadt = MOSADATA()

        with h5py.File(file, 'r') as f:
            mosadt.h5py_file = str(f['h5py_file'][...])

            # obs
            if 'obs' in f:
                for spl in f['obs'].keys():
                    mosadt.obs[spl] = pd.read_hdf(file, f'obs/{spl}')

            # obsm
            if 'obsm' in f:
                for spl in f['obsm'].keys():
                    mosadt.obsm[spl] = f[f'obsm/{spl}'][...]

            # spl
            mosadt.spl = pd.read_hdf(file, 'spl')

            # var
            if 'var' in f:
                for spl in f['var'].keys():
                    mosadt.var[spl] = pd.read_hdf(file, f'var/{spl}')

            # X
            if 'X' in f:
                for spl in f['X'].keys():
                    mosadt.X[spl] = pd.read_hdf(file, f'X/{spl}')

            # G
            if 'G' in f:
                for spl in f['G'].keys():
                    if spl not in smosadto.G:
                        mosadt.G[spl] = {}
                    for key in f[f'G/{spl}'].keys():
                        mosadt.G[spl][key] = nx.from_pandas_edgelist(pd.read_hdf(file, f'G/{spl}/{key}'))

            # images
            if 'images' in f:
                for spl in f['images'].keys():
                    if spl not in mosadt.images:
                        mosadt.images[spl] = {}
                    mosadt.images[spl] = f[f'images/{spl}'][...]

        return mosadt

    def to_pickle(self, file: str = None) -> None:
        """Save spatialOmics instance to pickle.

        Args:
            file: file to which instance is saved.

        Returns:

        """
        raise NotImplementedError('This version does not yet support saving pickled instances')

    @classmethod
    def from_pickle(cls, file: str = None) -> None:
        """Load spatialOmics instance from pickled file.

        Args:
            file: file to un-pickle

        Returns:
            spatialOmics instance

        """
        raise NotImplementedError('This version does not yet support reading pickled instances')

    def copy(self):
        """copy IMCData without copying graphs, masks and tiffstacks"""
        c = copy.copy(self)
        c.obs = copy.deepcopy(self.obs)
        c.spl = copy.deepcopy(self.spl)
        c.var = copy.deepcopy(self.var)
        c.X = copy.deepcopy(self.X)
        c.uns = copy.deepcopy(self.uns)
        return c

    def deepcopy(self):
        return copy.deepcopy(self)

    @staticmethod
    def from_annData(ad, img_container = None,
          sample_id: str = 'sample_id',
          img_layer: str = 'image',
          segmentation_layers: list = ['segmented_watershed']):
        """Converts a AnnData instance to a SpatialOmics instance.

        Args:
            ad: AnnData object
            img_container: Squidpy Image Container
            sample_id: column name that identifies different libraries in ad.obs

        Returns:
            SpatialOmics
        """

        if sample_id not in ad.obs:
            sample_name = 'spl_0'
            ad.obs[sample_id] = sample_name
            ad.obs[sample_id] = ad.obs[sample_id].astype('category')

        if len(ad.obs[sample_id].unique()) > 1:
            raise ValueError("""more than 1 sample_id present in ad.obs[sample_id].
            Please process each each sample individually.""")
        else:
            sample_name = ad.obs[sample_id][0]

        md = MOSADATA()
        x = pd.DataFrame(ad.X.A if issparse(ad.X) else ad.X, columns=ad.var.index)

        md.X = {sample_name: x}
        md.obs = {sample_name: ad.obs}
        md.var = {sample_name: ad.var}
        md.spl = pd.DataFrame(index=[sample_name])

        if 'spatial' in ad.obsm:
            coord = ad.obsm['spatial']
            coord = pd.DataFrame(coord, index=md.obs[sample_name].index, columns=['x','y'])
            md.obs[sample_name] = pd.concat((md.obs[sample_name], coord), 1)

        if img_container is not None:
            img = img_container[img_layer]
            md.images = {sample_name: img}

            # segmentations = {}
            # for i in segmentation_layers:
            #     if i in img_container:
            #         segmentations.update({i:img_container[i]})
            # md.masks.update({sample_name:segmentations})

        return md

    def to_annData(self,
        one_adata=True,
        spatial_keys_mosadt=['x', 'y'],
        spatial_key_ad='spatial'):
        """Converts the current SpatialOmics instance into a AnnData instance.
        Does only takes .X, .obs and .var attributes into account.

        Args:
            one_adata: bool whether for each sample a individual AnnData should be created
            spatial_keys_mosadt: tuple column names of spatial coordinates of observations in mosadt.obs[spl]
            spatial_key_ad: str key added to ad.obsm to store the spatial coordinates

        Returns:

        """

        try:
            from anndata import AnnData
        except ImportError as e:
            raise ImportError('Please install AnnData with `pip install anndata')

        mosadt = self

        if one_adata:
            keys = list(mosadt.obs.keys())
            # we iterate through the keys to ensure that we have the order of the different dicts aligned
            # we could apply pd.concat directly on the dicts

            X = pd.concat([mosadt.X[i] for i in keys])
            obs = pd.concat([mosadt.obs[i] for i in keys])
            obs.index = range(len(obs))
            var = mosadt.var[keys[0]]

            # create AnnData
            ad = AnnData(X=X.values, obs=obs, var=var)

            # import spatial coordinates
            if all([i in obs for i in spatial_keys_mosadt]):
                spatial_coord = ad.obs[spatial_keys_mosadt]
                ad.obs = ad.obs.drop(columns=spatial_keys_mosadt)
                ad.obsm.update({spatial_key_ad: spatial_coord.values})

            return ad

        else:
            ads = []
            for spl in mosadt.obs.keys():

                # create AnnData
                ad = AnnData(X=mosadt.X[spl].values, obs=mosadt.obs[spl], var=mosadt.var[spl])

                # import spatial coordinates
                if all([i in ad.obs for i in spatial_keys_mosadt]):
                    spatial_coord = ad.obs[spatial_keys_mosadt]
                    ad.obs = ad.obs.drop(columns=spatial_keys_mosadt)
                    ad.obsm.update({spatial_key_ad: spatial_coord.values})

                ads.append(ad)
            return ads
def add2uns(mosadata, res, spl: str, parent_key, key_added):
    if spl in mosadata.uns:
        if parent_key in mosadata.uns[spl]:
            mosadata.uns[spl][parent_key][key_added] = res
        else:
            mosadata.uns[spl].update({parent_key: {key_added: res}})
    else:
        mosadata.uns.update({spl: {parent_key: {key_added: res}}})

