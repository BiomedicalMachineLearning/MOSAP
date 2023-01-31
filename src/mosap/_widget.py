"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QGroupBox
from qtpy.QtWidgets import QLabel, QComboBox, QGridLayout, QFileDialog
from mosap.mosap  import SpatialOmics
if TYPE_CHECKING:
    import napari
from napari.utils.notifications import show_info

class MultiOmicRegistrationWidget(QWidget):
    TRANSFORMATIONS = {
        "translation": "Translation",
        "scaled_rotation": "Scaled Rotation",
        "affine": "Affine",
        "bilinear": "Bilinear",
        "bspline":'BSpline',
        "affine_bilinear":'Affine and Bilinear',
        "affine_bspline":'Affine and BSpline',
        "bspline_bilinear":'BSpline and Bilinear'
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
        # btn = QPushButton("Click me!")
        # btn.clicked.connect(self._on_click)

        # self.setLayout(QHBoxLayout())
        # self.layout().addWidget(btn)
        self.createImageOptionWidget()


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

        try:
            self._load_tmat(fname)
        except Exception:
            show_info(f"Could not load transformation matrix from {fname}")
            return

        self.status.setText(f'Loaded from "{os.path.basename(fname)}"')
        self.btn_tmat_save.setEnabled(True)

        # if image is open we can now transform it
        if self.image.currentData() is not None:
            self.btn_transform.setEnabled(True)
        show_info("Loaded transformation matrices from file")

    def createImageOptionWidget(self):
        groupBox = QGroupBox(self, title="Images and Transformers")
        vbox = QVBoxLayout(groupBox)
        # Raw data layer
        tooltip_message_moving_image = "Image to be registered/transformed."
        tooltip_message_reference_image = "Reference image for registration"
        tooltip_transformation = "Type of applied transformation."
        self.moving_image = QComboBox(groupBox)
        self.moving_image.setToolTip(tooltip_message_moving_image)
        self.moving_image.addItems(['mov1','mov2','mov3','mov4','mov5'])
        self.ref_image = QComboBox(groupBox)
        self.ref_image.setToolTip(tooltip_message_reference_image)
        self.ref_image.addItems(['ref1', 'ref2', 'ref3', 'ref4'])
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
            
        self.registration_model.setCurrentText("Affine and BSpline")
        
        tmat_label = QLabel("Transformer file")
        
        tmat_label.setToolTip('Registration transformers can be loaded and saved from files')
        
        self.btn_tmat_load = QPushButton("Load")
        self.btn_tmat_save = QPushButton("Save")
        curr_tmat_label = QLabel("Current transformation matrix")
        self.status = QLabel("None")
        self.btn_tmat_save.setEnabled(False)

        self.btn_tmat_save.clicked.connect(self._btn_save_tmat_onclick)
        self.btn_tmat_load.clicked.connect(self._btn_load_tmat_onclick)

        # Register and transform functional button
        self.btn_register = QPushButton("Register")
        self.btn_transform = QPushButton("Transform")

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
        grid.addWidget(curr_tmat_label, 5,0)
        grid.addWidget(self.status, 5,1)
        # vbox.addWidget(btn)
        vbox.addLayout(grid)
        groupBox.setLayout(vbox)
        self.layout().addWidget(groupBox)
        
        self.layout().addWidget(groupBox)
@magic_factory
def Transcript_Selection_Widget(img_layer: "napari.layers.Image"):
    print(f"you have selected Transcript widget {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def Heterogeneity_Vis_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected Heterogeneity widget {img_layer}")
