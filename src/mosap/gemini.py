import napari
# from napari.experimental import link_layers
from mosap import CosMX_MM_PER_PX, CosMX_PX_PER_MM, CosMX_ALPHA_MM_PER_PX, CosMX_ALPHA_PX_PER_MM
from skimage import io
from scipy import ndimage
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os
import warnings
import re
import vispy
import glob
import anndata as ad
import zarr
import vaex
import dask.array as da
from napari_cosmx.pairing import pair
from sklearn import preprocessing
from napari.utils.colormaps import AVAILABLE_COLORMAPS, label_colormap
from napari.utils.colormaps.standardize_color import transform_color

class Gemini:
    def _top_left_mm(self):
        """Return (y, x) tuple of mm translation for image layers
        """
        if self.alpha:
            return (min(self.fov_offsets['Y_mm']), -max(self.fov_offsets['X_mm']))
        else:
            return (-max(self.fov_offsets['X_mm']), min(self.fov_offsets['Y_mm']))

    def available_channels(self):
        """Return available morphology channels to display.
        """
        assert self.grp is not None, "No zarr images directory found"
        return sorted(set(self.grp.group_keys()) - set(['labels', 'protein']))

    def add_channel(self, name, colormap='gray'):
        """Add the requested morphology channel layer

        Args:
            name (str): Name of channel
        """        
        assert name in self.available_channels(), f"{name} not one of available channels: {self.available_channels()}"
        datasets = self.grp[f"{name}"].attrs["multiscales"][0]["datasets"]
        im = [da.from_zarr(os.path.join(self.folder, "images", f"{name}"), component=d["path"]) for d in datasets]
        self.viewer.add_image(im, colormap=colormap, blending="additive", name=name,
            contrast_limits = (0,2**16 - 1), scale = (self.mm_per_px, self.mm_per_px),
            translate=self._top_left_mm(),
            rotate=self.rotate)
        self.reset_fov_labels()

    def add_protein(self, name, colormap='cyan', visible=True):
        """Add the requested protein expression image

        Args:
            name (str): Name of protein
        """        
        assert self.proteins is not None, "No proteins found"
        assert name in self.proteins, f"{name} not found in proteins"
        datasets = self.grp[f"protein/{name}"].attrs["multiscales"][0]["datasets"]
        im = [da.from_zarr(os.path.join(self.folder, "images", "protein", name), component=d["path"]) for d in datasets]
        self.viewer.add_image(im, colormap=colormap, blending="additive", name=name,
            contrast_limits = (0,1000), scale = (self.mm_per_px*(1/self.expr_scale), self.mm_per_px*(1/self.expr_scale)),
            translate=self._top_left_mm(), visible=visible,
            rotate=self.rotate)
        self.reset_fov_labels()

    def add_segmentation(self):
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
        
    def add_cell_labels(self):
        """Add the cell labels layer
        """        
        assert 'labels' in self.grp.group_keys(), f"labels not found in zarr keys: {self.grp.group_keys()}"
        datasets = self.grp['labels'].attrs["multiscales"][0]["datasets"]
        labels = [da.from_zarr(os.path.join(self.folder, "images", "labels"), component=d["path"]) for d in datasets]
        # TODO: need to scale for protein expression images
        layer = self.viewer.add_labels(labels, scale=(self.mm_per_px, self.mm_per_px), translate=self._top_left_mm(),
            rotate=self.rotate)
        self.cells_layer = layer
        self.cells_layer.opacity = 0.5
        if self.metadata is not None:
            df = self.metadata.copy()
            df['index'] = self.metadata.index
            layer.features = pd.concat([pd.DataFrame.from_dict({'index': [0]}), df])
        layer.name = 'Cells'
        layer.editable = False

    def get_offsets(self, fov):
        """Get offsets for given FOV

        Args:
            fov (int): FOV number

        Returns:
            tuple: x and y offsets in mm
        """
        offset = self.fov_offsets[self.fov_offsets['FOV'] == fov]
        if self.alpha:
            x_offset = offset.iloc[0, ]["Y_mm"]
            y_offset = -offset.iloc[0, ]["X_mm"]
        else:
            x_offset = -offset.iloc[0, ]["X_mm"]
            y_offset = offset.iloc[0, ]["Y_mm"]
        return (x_offset, y_offset)

    def __init__(self, path, viewer=None):
        """Initialize new instance and launch viewer

        Args:
            path (str): path to adata, transcripts, and/or images or .h5ad path
            viewer (napari.viewer.Viewer): If None, napari will be launched.
        """
        assert os.path.exists(path), f"Could not find {path}"
        self._rotate = 0
        self.folder = path
        self.adata = None
        self.fov_labels = None
        self.segmentation_layer = None
        self.cells_layer = None
        if path.endswith(".h5ad"):
            self.adata = ad.read(path)
            self.folder = os.path.dirname(path)
        files = next(os.walk(self.folder))[2]
        if self.adata is None:
            res = [f for f in files if f.endswith('.h5ad')]
            if len(res) != 1:
                print("Could not find AnnData .h5ad file")
            else:
                self.adata = ad.read(os.path.join(self.folder, res[0]))
        assert os.path.exists(os.path.join(self.folder, "images")), f"No images directory found at {self.folder}"
        self.grp = zarr.open(os.path.join(self.folder, "images"), mode = 'r',)
        self.is_protein = 'protein' in self.grp.group_keys()
        self.fov_height = self.grp.attrs['CosMx']['fov_height']
        self.fov_width = self.grp.attrs['CosMx']['fov_width']
        self.alpha = (self.fov_height/self.fov_width) == 1
        if self.alpha:
            self.mm_per_px = ALPHA_MM_PER_PX
        else:
            self.mm_per_px = MM_PER_PX
        self.fov_offsets = pd.DataFrame.from_dict(self.grp.attrs['CosMx']['fov_offsets'])

        self.targets = None
        self.proteins = None
        self.expr_scale = 1
        if not self.is_protein:
            if not os.path.exists(os.path.join(self.folder, "targets.hdf5")):
                print(f"No targets.hdf5 file found at {self.folder}")    
            else:
                df = vaex.open(os.path.join(self.folder, "targets.hdf5"))
                offsets = vaex.from_pandas(self.fov_offsets.loc[:, ['X_mm', 'Y_mm', 'FOV']])
                self.targets = df.join(offsets, left_on='fov', right_on='FOV')
                if self.alpha:
                    self.targets['global_x'] = self.targets.x - self.targets.X_mm*(1/self.mm_per_px) - self._top_left_mm()[1]*(1/self.mm_per_px)
                    self.targets['global_y'] = self.targets.y + self.targets.Y_mm*(1/self.mm_per_px) - self._top_left_mm()[0]*(1/self.mm_per_px)
                else:
                    self.targets['global_x'] = self.targets.x + self.targets.Y_mm*(1/self.mm_per_px) - self._top_left_mm()[1]*(1/self.mm_per_px)
                    self.targets['global_y'] = self.targets.y - self.targets.X_mm*(1/self.mm_per_px) - self._top_left_mm()[0]*(1/self.mm_per_px)
        else:
            self.proteins = list(self.grp['protein'].group_keys())
            if 'CosMx' in self.grp['protein'].attrs:
                self.expr_scale = self.grp['protein'].attrs['CosMx']['scale']
        self.name = self.adata.uns['name'] if (self.adata is not None) and ('name' in self.adata.uns) else \
            os.path.basename(self.folder)

        if self.adata is None:
            # search for *_metadata.csv
            res = glob.glob(os.path.join(self.folder, "*_metadata.csv"))
            if len(res) == 1:
                self.read_metadata(res[0])
            else:
                self.metadata = None
                print(f"No AnnData .h5ad or _metadata.csv file found at {self.folder}")
        else:
            if self.adata.obs.index.name == "cell_ID":
                self.adata.obs['cell_ID'] = self.adata.obs.index
                res = [re.search("c_.*_(.*)_(.*)", i) for i in self.adata.obs.index]
                self.adata.obs['UID'] = [pair(int(i.group(1)), int(i.group(2))) for i in res]
                self.adata.obs.set_index('UID', inplace=True)
                self.metadata = self.adata.obs
            elif self.adata.obs.index.name != "UID":
                print(f"Expected index cell_ID or UID in AnnData obs, not {self.adata.obs.index}, \
                    unable to read metadata")
            else:
                self.metadata = self.adata.obs

        # launch viewer
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = napari.Viewer()
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "mm"
        if 'labels' in self.grp.group_keys():
            self.add_cell_labels()
            self.add_segmentation()
        self.add_fov_labels()
    
    @property
    def rotate(self):
        return self._rotate

    @rotate.setter
    def rotate(self, angle):
        for i in self.viewer.layers:
            i.rotate = angle
        self._rotate = angle


    def target_labels(self):
        """Return list of target names.

        If transcripts are loaded, return target names.
        """
        if self.targets:
            return sorted([i for i in self.targets.category_labels('target')
                if not i.startswith('FalseCode') and
                not i.startswith('NegPrb') and
                not i.startswith('SystemControl')])
        return []

    def rect_for_fov(self, fov):
        fov_height = self.fov_height*self.mm_per_px
        fov_width = self.fov_width*self.mm_per_px
        rect = np.array([
            list(self.get_offsets(fov)),
            list(map(sum, zip(self.get_offsets(fov), (fov_height, 0)))),
            list(map(sum, zip(self.get_offsets(fov), (fov_height, fov_width)))),
            list(map(sum, zip(self.get_offsets(fov), (0, fov_width))))
        ])
        y_offset = self._top_left_mm()[0]
        x_offset = self._top_left_mm()[1]
        return [[i[0] - y_offset, i[1] - x_offset] for i in rect]

    # keep reference to layer and keep on top
    # keep selected FOVs in object
    def add_fov_labels(self):
        rects = [self.rect_for_fov(i) for i in self.fov_offsets['FOV']]
        shape_properties = {
            'label': self.fov_offsets['FOV'].to_numpy()
        }
        text_parameters = {
            'text': 'label',
            'size': 12,
            'color': 'white'
        }
        shapes_layer = self.viewer.add_shapes(rects,
            face_color='#90ee90',
            edge_color='white',
            edge_width=0.02,
            properties=shape_properties,
            text = text_parameters,
            name = 'FOV labels',
            translate=self._top_left_mm(),
            rotate=self.rotate)
        shapes_layer.opacity = 0.5
        shapes_layer.editable = False
        self.fov_labels = shapes_layer
        
    def move_to_top(self, layer):
        """Move layer to top of layer list. Layer must be in list.

        Args:
            layer (Layer): Layer to move
        """        
        self.viewer.layers.append(self.viewer.layers.pop(self.viewer.layers.index(layer)))

    def reset_fov_labels(self):
        if self.segmentation_layer is not None:
            self.move_to_top(self.segmentation_layer)
        if self.fov_labels is not None:
            self.move_to_top(self.fov_labels)

    def read_metadata(self, path):
        df = pd.read_csv(path)
        if not 'UID' in df:
            assert 'cell_ID' in df.columns, "Need UID or cell_ID column in metadata"
            res = [re.search("c_.*_(.*)_(.*)", i) for i in df['cell_ID']]
            df['UID'] = [pair(int(i.group(1)), int(i.group(2))) for i in res]
        df.set_index('UID', inplace=True)   
        self.metadata = df
        if self.cells_layer is not None:
            # update features
            df = self.metadata.copy()
            df['index'] = self.metadata.index
            self.cells_layer.features = pd.concat([pd.DataFrame.from_dict({'index': [0]}), df])
    
    def is_categorical_metadata(self, col_name):
        return not is_numeric_dtype(self.metadata[col_name]) or len(pd.unique(self.metadata[col_name])) < 30

    def color_cells(self, col_name, color=None, contour=0, subset=None):
        """Change cell labels layer based on metadata

        Args:
            col_name (str): Column name in metadata, "all" colors cells the same color.
            color (str|dict): (1) Color name if col_name is "all", or
                (2) dictionary with keys being metadata values and value being color name, or
                (3) colormap for continuous metadata, https://matplotlib.org/stable/tutorials/colors/colormaps.html
            contour (int): Labels layer contour, 0 is filled, otherwise thickness of lines.
            subset (list of int): List of cell UIDs, if given only color these cells.
        """
        assert self.cells_layer is not None, "No cells layer found"
        self.cells_layer.contour = contour
        if self.metadata is None:        
            assert col_name == "all", "No metadata loaded"
            self.cells_layer.color = {1: color, None: color}
            return

        cells = subset if (subset is not None) else self.metadata.index
        if col_name == "all":
            self.cells_layer.color = {k:color for k in cells}
            self.cells_layer.color[None] = 'transparent'
        else:
            assert col_name in self.metadata, f"{col_name} not in metadata"
            if self.is_categorical_metadata(col_name):
                if color is None:
                    vals = np.unique(self.metadata[col_name])
                    cm = label_colormap(len(vals)+1)
                    # TODO: change others to this construction for dict
                    color = dict(zip(vals, cm.colors[1:]))
                self.cells_layer.color = {k:(color[v] if v in color else 'transparent')
                    for k,v in zip(cells, self.metadata.loc[cells][col_name])}
            else:
                if color is None:
                    color = 'gray'
                assert color in AVAILABLE_COLORMAPS, f"{color} not in {AVAILABLE_COLORMAPS.keys()}"
                cm = AVAILABLE_COLORMAPS[color]
                min_max_scaler = preprocessing.MinMaxScaler()
                # normalize to 0-1 with full range present in metadata
                x = pd.Series(min_max_scaler.fit_transform(self.metadata[col_name].values.reshape(-1, 1))[:, 0],
                    self.metadata.index)
                self.cells_layer.color = {k:v for k,v in zip(cells, cm.map(x.loc[cells]))}
        self.cells_layer.visible = True

    def plot_transcripts(self, gene, color, point_size=15):
        """Plot targets as dots

        Args:
            gene (str): Target to plot
            color (str): Color for points.
            point_size (int, optional): Point size. Defaults to 5.

        Returns:
            napari.layers.Points: A points layer for transcripts.
        """
        assert self.targets is not None, "No targets found, use read_targets.py first to create targets.hdf5 file."
        self.targets.select(self.targets.target == gene)
        y = self.targets.evaluate(self.targets.global_y, selection=True)
        x = self.targets.evaluate(self.targets.global_x, selection=True)
        points = np.array(list(zip(y, x)))
        points_layer = self.viewer.add_points(points,
            size = point_size,
            edge_width=0,
            face_color=color,
            scale = (self.mm_per_px, self.mm_per_px),
            translate=self._top_left_mm(),
            rotate=self.rotate)
        points_layer.name = gene
        points_layer.opacity = 1.0
        self.reset_fov_labels()
        return points_layer

    def add_points(self, fov=None, color=None, gray_no_cell=None, point_size=1):
        """Add all targets

        Args:
            fov (int or list, optional): Only add points for specified fov(s).
            color (str, optional): Color for points. If None will color by gene.
            gray_no_cell (bool, optional): Whether to color targets with no CellID gray. Defaults to True if color specified.
            point_size (int, optional): Point size. Defaults to 1.

        Returns:
            napari.layers.Points: A points layer for transcripts.
        """
        assert self.targets is not None, "No targets found, use read_targets.py first to create targets.hdf5 file."
        if gray_no_cell is None:
            gray_no_cell = color is not None
        targets = self.targets[~self.targets.target.str.startswith("NegPrb") &
            ~self.targets.target.str.startswith("FalseCode") &
            ~self.targets.target.str.startswith("SystemControl")]
        if fov is not None:
            targets = targets[targets.fov.isin(fov if isinstance(fov, list) else [fov])]
        y = targets.global_y.evaluate()
        x = targets.global_x.evaluate()
        id = targets.CellId.evaluate()
        genes = targets.target.evaluate()
        u, inv = np.unique(genes, return_inverse=True)
        cm = label_colormap(len(self.targets.category_labels('target'))+1)
        face_color = cm.colors[1:][inv] if color is None else np.full(shape=(len(genes), 4), fill_value=transform_color(color))
        if gray_no_cell:
            face_color[id == 0] = transform_color('gray')
        points = np.array(list(zip(y,x)))
        print(f"Plotting {len(genes):,} transcripts")
        points_layer = self.viewer.add_points(points,
            size = point_size,
            edge_width=0,
            properties={
                'CellId': id,
                'Gene': genes
            },
            scale=(self.mm_per_px, self.mm_per_px),
            face_color=face_color,
            translate=self._top_left_mm(),
            rotate=self.rotate)
        points_layer.opacity = 0.7
        points_layer.name = "Targets"
        return points_layer
