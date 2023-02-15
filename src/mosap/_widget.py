"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from typing import Any, Union, Optional
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QGroupBox, QFormLayout
from qtpy.QtWidgets import QLabel, QComboBox, QGridLayout, QFileDialog, QProgressBar
from mosap.mosap  import MultiSpatialOmics
import os
# if TYPE_CHECKING:
import napari
from napari.utils.notifications import show_info


# import SimpleITK as sitk
from mosap.registration import save_transformation_model, read_transformation_model, get_itk_from_pil
from mosap.registration import affine_registration_slides, bspline_registration_slides, sitk_transform_rgb
class MultiOmicRegistrationWidget(QWidget):
    TRANSFORMATIONS = {
        "translation": "Translation",
        "scaled_rotation": "Scaled Rotation",
        "affine": "Affine",
        "bilinear": "Bilinear",
        "bspline":'BSpline'
    }
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        # self.so = so
        vbox_layout = QVBoxLayout(self)
        vbox_layout.setContentsMargins(9, 9, 9, 9)

        self.setLayout(vbox_layout)
        
        self.createImageOptionWidget()
    # def __init__(self, napari_viewer, mosap: MultiSpatialOmics, show_widget=False):
    #     super().__init__()
    #     self.viewer = napari_viewer
    #     self.mosap = mosap
    #     vbox_layout = QVBoxLayout(self)
    #     vbox_layout.setContentsMargins(9, 9, 9, 9)

    #     self.setLayout(vbox_layout)
    #     if show_widget:
    #         self.createImageOptionWidget()

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
    
    def _on_layer_change(self, e):
        self.image.clear()
        for x in self.viewer.layers:
            if (
                isinstance(x, napari.layers.image.image.Image)
                and len(x.data.shape) > 2
            ):
                self.image.addItem(x.name, x.data)

        if self.image.count() < 1:
            self.btn_register.setEnabled(False)
            self.btn_transform.setEnabled(False)
            self.btn_register_transform.setEnabled(False)
        else:
            self.btn_register.setEnabled(True)
            self.btn_transform.setEnabled(self.tmats is not None)
            self.btn_register_transform.setEnabled(True)
    
    

    def _save_tmat(self, filename:str):
        save_transformation_model(self.tmats, filename)

    def _load_tmat(self, filename:str):
        transformer = read_transformation_model(filename)
        self.tmats = transformer

    
    def _btn_save_tmat_onclick(self, value: bool):
        fname = QFileDialog.getSaveFileName(
            self, "Save transformationer to file", filter="*.tfm"
        )[0]
        self._save_tmat(fname)
        show_info("Saved transformation matrices to file")

    def _btn_load_tmat_onclick(self, value: bool):
        fname = QFileDialog.getOpenFileName(
            self, "Open transformation matrix file", filter="*.tfm"
        )[0]
        # display transformer
        try:
            self._load_tmat(fname)
        except Exception:
            show_info(f"Could not load transformer from {fname}")
            return
        self.curr_tmat_label.setText('Loaded')
        self.status.setText(f"{os.path.basename(fname)}")
        self.btn_tmat_save.setEnabled(True)

        # if image is open we can now transform it
        if self.moving_image.currentText() is not None:
            self.btn_transform.setEnabled(True)
        show_info("Loaded transformer from file")

    def _btn_register_onclick(self, value: bool):
        self._change_button_accessibility(False)
        try:
            self._run("register")
        finally:
            show_info("Registered image")
            self._change_button_accessibility(True)

    def _btn_transform_onclick(self, value: bool):
        self._change_button_accessibility(False)
        # run transform
        try:
            self._run("transform")
        finally:
            show_info("Transformed image")
            self._change_button_accessibility(True)

    def _change_button_accessibility(self, value: bool):
        self.btn_register.setEnabled(value)
        self.btn_transform.setEnabled(value)
        

    def createImageOptionWidget(self):
        groupBox = QGroupBox(self, title="Images and Transformers")
        vbox = QVBoxLayout(groupBox)
        # Raw data layer
        tooltip_message_moving_image = "Image to be registered/transformed."
        tooltip_message_reference_image = "Reference image for registration"
        tooltip_transformation = "Type of applied transformation."

        available_images = []
        if len(self.viewer.layers):
            count_img_layer = 0
            for layer in self.viewer.layers:
                if isinstance(layer, napari.layers.image.image.Image):
                    count_img_layer += 1
                    available_images.append(layer)
            if count_img_layer == 0:
                show_info('No image loaded')
                raise FileNotFoundError('MOSAP: No image loaded')
        else:
            show_info('No image found')
            raise FileNotFoundError('No image loaded')

        self.moving_image = QComboBox(groupBox)
        self.moving_image.setToolTip(tooltip_message_moving_image)
        moving_options = [i.name for i in available_images]
        self.moving_image.addItems(moving_options)
        self.moving_image.setCurrentText(moving_options[0])
        self.moving_image.currentIndexChanged.connect(
            self._move_image_onchange
        )
        self.ref_image = QComboBox(groupBox)
        self.ref_image.setToolTip(tooltip_message_reference_image)
        ref_options = [i.name for i in available_images]
        self.ref_image.addItems(ref_options)
        self.ref_image.setCurrentText(ref_options[1])
        self.ref_image.currentIndexChanged.connect(
            self._ref_image_onchange
        )
        # self.image.currentIndexChanged.connect(self._image_onchange)
        reg_img_label = QLabel("Register Image")
        reg_img_label.setToolTip(tooltip_message_moving_image)
        ref_img_label = QLabel("Reference Image")
        ref_img_label.setToolTip(tooltip_message_reference_image)

        # set transformationer for registration
        registration_model_lbl = QLabel("Registration Transformer")
        registration_model_lbl.setToolTip(tooltip_transformation)
        self.registration_model = QComboBox(groupBox)
        self.registration_model.setToolTip(tooltip_transformation)
        for k, v in self.TRANSFORMATIONS.items():
            self.registration_model.addItem(v, k)
            
        self.registration_model.setCurrentText("Affine")
        self.registration_model.currentIndexChanged.connect(
            self._transformation_onchange
        )
        
        
        tmat_label = QLabel("Transformer file")
        
        tmat_label.setToolTip('Registration transformers can be loaded and saved from files')
        
        self.btn_tmat_load = QPushButton("Load")
        self.btn_tmat_save = QPushButton("Save")
        self.curr_tmat_label = QLabel("Current transformer")
        self.status = QLabel("None")
        self.btn_tmat_save.setEnabled(False)

        self.btn_tmat_save.clicked.connect(self._btn_save_tmat_onclick)
        self.btn_tmat_load.clicked.connect(self._btn_load_tmat_onclick)

        # Register and transform functional button
        self.btn_register = QPushButton("Register")
        self.btn_transform = QPushButton("Transform")
        self.btn_register.clicked.connect(self._btn_register_onclick)
        self.btn_transform.clicked.connect(self._btn_transform_onclick)

        # self.pbar_label = QLabel()
        # self.pbar = QProgressBar()
        # self.pbar.setVisible(False)
        

        grid = QGridLayout()
        grid.addWidget(reg_img_label, 1,0)
        grid.addWidget(self.moving_image, 1,1)
        grid.addWidget(ref_img_label, 2,0)
        grid.addWidget(self.ref_image, 2,1)
        grid.addWidget(registration_model_lbl, 3,0)
        grid.addWidget(self.registration_model, 3,1)
        grid.addWidget(tmat_label, 4,0)
        grid.addWidget(self.btn_tmat_load, 4,1)
        grid.addWidget(self.btn_tmat_save, 4,2)
        grid.addWidget(self.curr_tmat_label, 5,0)
        grid.addWidget(self.status, 5,1)
        grid.addWidget(self.btn_register, 6,0)
        grid.addWidget(self.btn_transform, 6,1)
        # grid.addWidget(self.pbar_label, 7,0)
        # grid.addWidget(self.pbar, 7,1)
        # vbox.addWidget(btn)
        vbox.addLayout(grid)
        groupBox.setLayout(vbox)
        self.layout().addWidget(groupBox)
        
        # self.layout().addWidget(groupBox)

    def _transformation_onchange(self, value: str):
        def without(d, key):
            new_d = d.copy()
            new_d.pop(key)
            return new_d
        print(self.registration_model.currentData() == "bilinear")
        print(self.registration_model.currentData())

    def _move_image_onchange(self, value: str):
        print("napari has", len(self.viewer.layers), "layers")
        print('moving image selected ', self.moving_image.currentText())

    def _ref_image_onchange(self,  value: str):
        print("napari has", len(self.viewer.layers), "layers")
        print('reference image selected ', self.ref_image.currentText())

    def _update_image_options(self, value: str):
        # available_images = []
        # for layer in  self.viewer.layers:
        #     if isinstance(layer, napari.layers.image.image.Image):
        #         available_images.append(layer)
        self.moving_image.addItem(value)
        self.ref_image.addItem(value)
        print('updated new image option', value)

    # def _ref_image_onchange(self,  value: str):
    #     print("napari has", len(self.viewer.layers), "layers")
    #     print('reference image selected ', self.ref_image.currentText())

        # if self.registration_model.currentData() == "bilinear":
        #     refs = without(self.REFERENCES, "previous")
        # else:
        #     refs = self.REFERENCES

        # self.reference.clear()
        # for k, v in refs.items():
        #     self.reference.addItem(v, k)
    
    def _run(self, action):
        import numpy as np
        from PIL import Image
        fixed_img, moving_img = None, None
        for layer in self.viewer.layers:
            if layer.name == self.moving_image.currentText():
                moving_img = layer.data
            elif layer.name == self.ref_image.currentText():
                fixed_img = layer.data
            else:
                Exception('Images not found')
        if action == 'transform':
            transformer = self.tmats
            transformation = self.registration_model.currentData()
            print('Perform transformation', transformation)
            # convert to pil type image
            pil_moving_img = Image.fromarray(moving_img)
            pil_fixed_img = Image.fromarray(fixed_img)
            transformed_image  =  sitk_transform_rgb(pil_moving_img, pil_fixed_img, transformer)
            np_transformed_res = np.array(transformed_image)
            name = transformation + "_transformed_" + self.moving_image.currentText()
            self.viewer.add_image(np_transformed_res, name=name)
            self._update_image_options(name)
        elif action == 'register':
            moving_img = Image.fromarray(moving_img)
            fixed_img = Image.fromarray(fixed_img)
            # convert color images (3 channels) to gray scale images
            moving_img_gray = moving_img.convert('L')
            fixed_img_gray = fixed_img.convert('L')
            moving_img_gray = get_itk_from_pil(moving_img_gray)
            fixed_img_gray = get_itk_from_pil(fixed_img_gray)
            if self.registration_model.currentData() == 'affine':
                
                self.tmats = affine_registration_slides(fixed_img_gray,moving_img_gray,  plot_registration_progress=False)
                show_info('Registered image: ',self.moving_image.currentText())
                
            elif self.registration_model.currentData() == 'bspline':
                self.tmats = bspline_registration_slides(fixed_img_gray, moving_img_gray,  plot_registration_progress=False)
                show_info('Registered image: ',self.moving_image.currentText())



# @magic_factory
class Transcript_Selection_Widget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        # self.so = so
        # vbox_layout = QVBoxLayout(self)
        # vbox_layout.setContentsMargins(9, 9, 9, 9)

        # self.setLayout(vbox_layout)
        layout = QFormLayout()

        self.setLayout(layout)
        self.layout = layout
        # btn = QPushButton("Click me!")
        # btn.clicked.connect(self._on_click)

        # self.setLayout(QHBoxLayout())
        # self.layout().addWidget(btn)
        self.createTranscriptsWidget()
    def createTranscriptsWidget(self):
        # groupBox = QGroupBox(self, title="RNA Transcripts")
        buttons_layout = QHBoxLayout()
        btn = QPushButton("Select")
        btn.clicked.connect(self._on_select_rna_click)
        buttons_layout.addWidget(btn)
        boxp = QComboBox()
        boxp.addItems(['1','2','3','4','5', '6'])

        self.targetsComboBox = boxp
        self.targetsComboBox.currentIndexChanged.connect(
            self._selected_number_target
        )
        # vbox = QVBoxLayout()
        # grid = QGridLayout()
        layout = QFormLayout()
        lbl_numb_target = QLabel('Number of targets:')
        lbl_numb_target.setToolTip('Select the number of transcript to display')
        layout.addRow(lbl_numb_target, self.targetsComboBox)
        layout.addRow(buttons_layout)
        self.layout.addRow(layout)
        # self.groupbox = groupBox

    def _on_select_rna_click(self):
        print('Plot RNA expression function')
        # groupBox = QGroupBox(self, title="Select Markers")
        flayout = QFormLayout()
        # grid = QGridLayout()
        self.list_transcripts = list()
        self.num_transcript= int(self.targetsComboBox.currentText())
        list_markers = ['CSF1R','IL34','CD44','AXL','MKI67', 'COL1A1']
        for i in range(self.num_transcript):

            boxp1 = QComboBox()
            boxp1.addItems(list_markers)
            self.list_transcripts.append(boxp1)
            self.layout.addRow(QLabel('Transcript {0}:'.format(str(i+1))), boxp1)
            # grid.addWidget(self.list_transcripts[i], i+1, 1)
            # vbox.addLayout(grid)
        # flayout.setLayout(vbox)
        buttons_layout = QHBoxLayout()
        btn = QPushButton("Analyse")
        btn.clicked.connect(self._on_analyse_colocalisation_click)
        buttons_layout.addWidget(btn)
        # vbox.addWidget(btn)
        self.layout.addRow(buttons_layout)
        # self.layout().addWidget(groupBox)
        
            # boxp.addItems(self.gem.target_labels())
    def _on_analyse_colocalisation_click(self):
        print('Selected', self.num_transcript)
        self.transcript_names = list()
        for i in range(self.num_transcript):
            print(self.list_transcripts[i].currentText())
            self.transcript_names.append(self.list_transcripts[i].currentText())
        show_info("Display layers: "+ ' '.join(self.transcript_names))
        
    
    def _selected_number_target(self):
        print('Selected:', self.targetsComboBox.currentText())



# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
class Heterogeneity_Vis_widget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        layout = QFormLayout()

        self.setLayout(layout)
        self.layout = layout

        self.createSpatialNetworkWidget()
    def createSpatialNetworkWidget(self):
        
        boxp = QComboBox()
        boxp.addItems(['KNN','Radius','Delaunay triangulation', 'Contact'])
        self.networkbox= boxp
        self.networkbox.currentIndexChanged.connect(
            self._selected_graph_model
        )
        buttons_layout = QHBoxLayout()
        btn = QPushButton("Select")
        btn.clicked.connect(self._on_select_method)
        buttons_layout.addWidget(btn)
        layout = QFormLayout()
        lbl_spatial_nw = QLabel('Spatial cellular network:')
        lbl_spatial_nw.setToolTip('Select the spatial cellular network')
        layout.addRow(lbl_spatial_nw, self.networkbox)
        layout.addRow(buttons_layout)
        self.layout.addRow(layout)

    def _on_select_method(self):
        print('selected network models', self.networkbox.currentText())

    def _selected_graph_model(self):
        print('current network models',self.networkbox.currentText())
# def Heterogeneity_Vis_widget(img_layer: "napari.layers.Image"):
    # print(f"you have selected Heterogeneity widget {img_layer}")
