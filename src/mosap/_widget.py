"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from typing import Any, Union, Optional
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QGroupBox
from qtpy.QtWidgets import QLabel, QComboBox, QGridLayout, QFileDialog,QProgressBar
from mosap.mosap  import SpatialOmics
import os
# if TYPE_CHECKING:
import napari
from napari.utils.notifications import show_info


# import SimpleITK as sitk
from mosap.registration import save_transformation_model, read_transformation_model, get_itk_from_pil
from mosap.registration import affine_registration_slides, bspline_registration_slides
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

        self.status.setText(f'Loaded from "{os.path.basename(fname)}"')
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
        for layer in  self.viewer.layers:
            if isinstance(layer, napari.layers.image.image.Image):
                available_images.append(layer)

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
            
        self.registration_model.setCurrentText("Affine and BSpline")
        self.registration_model.currentIndexChanged.connect(
            self._transformation_onchange
        )
        
        
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
        grid.addWidget(curr_tmat_label, 5,0)
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
        if action == 'transform':
            transformer = self.tmats
            transformation = self.registration_model.currentData()
            print('Perform transform', transformation)
            if transformation == 'affine':
                pass
            elif transformation == 'bspline':
                pass
        elif action == 'register':
            if self.registration_model.currentData() == 'affine':
                
                fixed_img, moving_img = None, None
                for layer in self.viewer.layers:
                    if layer.name == self.moving_image.currentText():
                        moving_img = layer.data
                    elif layer.name == self.ref_image.currentText():
                        fixed_img = layer.data
                    else:
                        Exception('Images not found')
                moving_img = Image.fromarray(moving_img)
                fixed_img = Image.fromarray(fixed_img)
                # convert color images (3 channels) to gray scale images
                moving_img_gray = moving_img.convert('L')
                fixed_img_gray = fixed_img.convert('L')
                moving_img_gray = get_itk_from_pil(moving_img_gray)
                fixed_img_gray = get_itk_from_pil(fixed_img_gray)
                self.tmats = affine_registration_slides(moving_img_gray, fixed_img_gray, plot_registration_progress=False)
                show_info('Registered image: ',self.moving_image.currentText())
                
        # moving_average = (
        #     self.moving_average.value()
        #     if self.perform_moving_average.isChecked()
        #     else 1
        # )

        # n_frames = self.n_frames.value()
        # reference = self.reference.currentData()

        # if reference not in self.REFERENCES:
        #     raise ValueError(f'Unknown reference "{reference}"')

        # image = self.image.currentData()

        # self.pbar.setMaximum(image.shape[0] - 1)

        # def hide_pbar():
        #     self.pbar_label.setVisible(False)
        #     self.pbar.setVisible(False)

        # def show_pbar():
        #     self.pbar_label.setVisible(True)
        #     self.pbar.setVisible(True)

        # @thread_worker(connect={"returned": hide_pbar}, start_thread=False)
        # def _register_stack(image) -> ImageData:

        #     sr = StackReg(transformations[transformation])

        #     axis = 0

        #     if action in ["register", "register_transform"]:
        #         image_reg = image
        #         idx_start = 1

        #         if moving_average > 1:
        #             idx_start = 0
        #             size = [0] * len(image_reg.shape)
        #             size[axis] = moving_average
        #             image_reg = running_mean(
        #                 image_reg, moving_average, axis=axis
        #             )

        #         tmatdim = 4 if transformation == "bilinear" else 3

        #         tmats = np.repeat(
        #             np.identity(tmatdim).reshape((1, tmatdim, tmatdim)),
        #             image_reg.shape[axis],
        #             axis=0,
        #         ).astype(np.double)

        #         if reference == "first":
        #             ref = np.mean(
        #                 image_reg.take(range(n_frames), axis=axis), axis=axis
        #             )
        #         elif reference == "mean":
        #             ref = image_reg.mean(axis=0)
        #             idx_start = 0
        #         elif reference == "previous":
        #             pass
        #         else:  # pragma: no cover - can't be reached due to check above
        #             raise ValueError(f'Unknown reference "{reference}"')

        #         self.pbar_label.setText("Registering...")

        #         iterable = range(idx_start, image_reg.shape[axis])

        #         for i in iterable:
        #             slc = [slice(None)] * len(image_reg.shape)
        #             slc[axis] = i

        #             if reference == "previous":
        #                 ref = image_reg.take(i - 1, axis=axis)

        #             tmats[i, :, :] = sr.register(
        #                 ref, simple_slice(image_reg, i, axis)
        #             )

        #             if reference == "previous" and i > 0:
        #                 tmats[i, :, :] = np.matmul(
        #                     tmats[i, :, :], tmats[i - 1, :, :]
        #                 )

        #             yield i - idx_start + 1

        #         self.tmats = tmats
        #         image_name = self.image.itemText(self.image.currentIndex())
        #         transformation_name = self.TRANSFORMATIONS[transformation]
        #         self.status.setText(
        #             f'Registered "{image_name}" [{transformation_name}]'
        #         )
        #         self.btn_tmat_save.setEnabled(True)

        #     if action in ["transform", "register_transform"]:
        #         tmats = self.tmats

        #         # transform

        #         out = image.copy().astype(np.float)

        #         self.pbar_label.setText("Transforming...")
        #         yield 0  # reset pbar

        #         for i in range(image.shape[axis]):
        #             slc = [slice(None)] * len(out.shape)
        #             slc[axis] = i
        #             out[tuple(slc)] = sr.transform(
        #                 simple_slice(image, i, axis), tmats[i, :, :]
        #             )
        #             yield i

        #         # convert to original dtype
        #         if np.issubdtype(image.dtype, np.integer):
        #             out = to_int_dtype(out, image.dtype)

        #         return out

        # def on_yield(x):
        #     self.pbar.setValue(x)

        # def on_return(img):
        #     if img is None:
        #         return
        #     image_name = self.image.itemText(self.image.currentIndex())

        #     transformation_name = self.transformation.itemText(
        #         self.transformation.currentIndex()
        #     )

        #     layer_name = f"Registered {image_name} ({transformation_name})"
        #     self.viewer.add_image(data=img, name=layer_name)

        # self.worker = _register_stack(image)

        # if running_coverage:
            
        #     patch_worker_for_coverage(self.worker)
        # self.worker.yielded.connect(on_yield)
        # self.worker.returned.connect(on_return)
        # self.worker.start()

        # show_pbar()


# @magic_factory
class Transcript_Selection_Widget(QWidget):
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
        self.createTranscriptsWidget()
    def createTranscriptsWidget(self):
        groupBox = QGroupBox(self, title="RNA Transcripts")

        btn = QPushButton("Add layer")
        btn.clicked.connect(self._on_select_rna_click)

        boxp = QComboBox(groupBox)
        # boxp.addItems(self.gem.target_labels())
        boxp.addItems(['1','2','3','4','5', '6'])
        colors = ['white', 'red', 'green', 'blue',
            'magenta', 'yellow', 'cyan']
        boxc = QComboBox(groupBox)
        boxc.addItems(colors)

        self.targetsComboBox = boxp
        self.targetsComboBox.currentIndexChanged.connect(
            self._selected_number_target
        )
        self.colorsComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()
        lbl_numb_target = QLabel('Number of targets:')
        lbl_numb_target.setToolTip('Select the number of transcript to display')

        grid.addWidget(lbl_numb_target, 1, 0)
        grid.addWidget(self.targetsComboBox, 1, 1)
        grid.addWidget(QLabel('color:'), 2, 0)
        grid.addWidget(self.colorsComboBox, 2, 1)
        vbox.addLayout(grid)
        vbox.addWidget(btn)
        groupBox.setLayout(vbox)

        self.layout().addWidget(groupBox)

    def _on_select_rna_click(self):
        print('Plot RNA expression function')
        # self.gem.plot_transcripts(gene=self.targetsComboBox.currentText(),
        # color=self.colorsComboBox.currentText())
    
    def _selected_number_target(self):
        print('Selected:', self.targetsComboBox.currentData())



# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def Heterogeneity_Vis_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected Heterogeneity widget {img_layer}")
