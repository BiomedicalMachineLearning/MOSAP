import pandas as pd
import networkx as nx
from skimage import io
import seaborn as sns
from scipy.sparse import issparse
import numpy as np

import os
import copy

import h5py
## abstract class for modification and storing network for heterogeneity analysis
from spatialOmics import SpatialOmics

#light weight object
class MultiOmicsSpatial(SpatialOmics):
	# to create a blank object, inherit from SpatialOmics data 
	    cmaps = {
        'default': sns.color_palette('Reds', as_cmap=True),
        'category': sns.color_palette('tab10', as_cmap=True)
    }
    def __init__(self):
    	super().__init__(self)
    	self.graph_engine = 'networkx'
        self.random_seed = 42  # for reproducibility
        self.h5py_file = 'MultiSpatialOmics.h5py'  # backend store

        self.obs = {}  # container for observation level features        
        # self.meta_group = pd.DataFrame() # storing sample metadata
        self.spl = pd.DataFrame()  # container for sample level features
        self.var = {}  # container with variable descriptions of X
        self.X = {}  # cell expression dataframe of each spl
        self.G = {}  # graphs
        self.uns = {}  # unstructured container
        self.uns.update({'cmaps':self.cmaps,
                         'cmap_labels': {}})

        # self.obs_keys = None  # list of observations in self.obs.index
        self.spl_keys = None  # list of samples in self.spl.index

        self.images = {}
        self.registration_model = {}
        self.masks = {}
        #--------------------------------------------------------------
        

    def __repr__(self):
        """Function to return representation of the object"""
        l = [len(self.obs[i]) for i in self.obs]

        cols_spl = self.spl.columns.to_list()

        cols_obs = set()
        [cols_obs.update((self.obs[i].columns)) for i in self.obs]
        cols_obs = [*cols_obs]

        cols_var = set()
        [cols_var.update((self.var[i].columns)) for i in self.var]
        cols_var = [*cols_var]

        mask_names = set()
        [mask_names.update((self.masks[i].keys())) for i in self.masks]
        mask_names = [mask_names]

        graph_names = set()
        [graph_names.update((self.G[i].keys())) for i in self.G]
        graph_names = [*graph_names]

        s = f""" MultiSpatialOmics object with {sum(l)} observations across {len(l)} samples.
    X: {len(self.X)} samples,
    spl: {len(self.spl)} samples,
        columns: {cols_spl}
    obs: {len(self.spl)} samples,
        columns: {cols_obs}
    var: {len(self.var)} samples,
        columns: {cols_var}
    G: {len(self.G)} samples,
        keys: {graph_names}
    masks: {len(self.masks)} samples
        keys: {mask_names}
    images: {len(self.images)} samples"""
        return s
    
    def __len__(self):
        return len(self.spl)
    