"""
Definition of custom QWidget to provide interface for Gemini module.

"""
from qtpy.QtCore import Qt, QSize, QItemSelectionModel
from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
    QGridLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QGroupBox,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)

from qtpy.QtGui import (
    QIcon,
    QPixmap,
    QColor,
    QImage
)

from napari.utils.notifications import (
    notification_manager,
    show_info,
)
from napari.experimental import link_layers
from napari.layers import Labels, Image
from napari.utils.colormaps import AVAILABLE_COLORMAPS, label_colormap
import numpy as np
import pandas as pd

from napari_cosmx.gemini import Gemini


class GeminiQWidget(QWidget):
    def __init__(self, napari_viewer, gem: Gemini):
        super().__init__()
        self.viewer = napari_viewer
        self.gem = gem

        vbox_layout = QVBoxLayout(self)
        vbox_layout.setContentsMargins(9, 9, 9, 9)

        self.setLayout(vbox_layout)

        self.createMorphologyImageWidget()
        if self.gem.is_protein:
            self.createProteinExpressionWidget()
        else:
            self.createTranscriptsWidget()
        self.createMetadataWidget()

    def _on_morph_click(self):
        # for debugging
        self.viewer.update_console({'gem': self.gem})
        self.gem.add_channel(self.channelsComboBox.currentText(),
                             self.channelsColormapComboBox.currentText())
        self.gem.reset_fov_labels()

    def _on_expr_click(self):
        self.gem.add_protein(self.proteinComboBox.currentText(), self.colormapComboBox.currentText())

    def _on_rna_click(self):
        self.gem.plot_transcripts(gene=self.targetsComboBox.currentText(),
                                  color=self.colorsComboBox.currentText())

    def _get_label_colors(self):
        if not self.showSelectedCheckbox.isChecked():
            colors = [i.icon().pixmap(QSize(1, 1)).toImage().pixelColor(0, 0).name() for i in
                      self.labelListWidget.items()]
            items = [i.text() for i in self.labelListWidget.items()]
            items = pd.Series(items).astype(self.gem.cells_layer.features[self.metaComboBox.currentText()].dtype)
            return dict(zip(items, colors))
        else:
            colors = [i.icon().pixmap(QSize(1, 1)).toImage().pixelColor(0, 0).name() for i in
                      self.labelListWidget.selectedItems()]
            items = [i.text() for i in self.labelListWidget.selectedItems()]
            items = pd.Series(items).astype(self.gem.cells_layer.features[self.metaComboBox.currentText()].dtype)
            return dict(zip(items, colors))

    def _labels_selected(self):
        if self.showSelectedCheckbox.isChecked():
            meta_col = self.metaComboBox.currentText()
            colors = self._get_label_colors()
            selected_items = [i.text() for i in self.labelListWidget.selectedItems()]
            selected_items = pd.Series(selected_items).astype(self.gem.cells_layer.features[meta_col].dtype)
            cells = self.gem.cells_layer.features[self.gem.cells_layer.features[meta_col].isin(selected_items)]['index']
            self.gem.color_cells(meta_col, colors, subset=cells)

    def _show_selected_changed(self, state):
        if self.showSelectedCheckbox.isChecked():
            self._labels_selected()
        else:
            self._meta_changed(self.metaComboBox.currentText())

    def _meta_changed(self, text):
        self.showSelectedCheckbox.setChecked(False)
        if text == "" or text is None or not self.gem.is_categorical_metadata(text):
            self.labelListWidget.setHidden(True)
            self.showSelectedCheckbox.setHidden(True)
        else:
            self.updateLabelsWidget(text)
            self.labelListWidget.setHidden(False)
            self.showSelectedCheckbox.setHidden(False)
        if text != "" and text is not None:
            self.gem.color_cells(text)

    def update_metadata(self, path):
        # TODO: would be nice to merge metadata, but replace for now
        self.gem.read_metadata(path)
        if self.gem.metadata is not None:
            self.metaComboBox.clear()
            self.metaComboBox.addItems([i for i in self.gem.metadata.columns if i not in ['cell_ID', 'fov', 'CellId']])

    def createMorphologyImageWidget(self):
        groupBox = QGroupBox(self, title="Morphology Images")

        btn = QPushButton("Add layer")
        btn.clicked.connect(self._on_morph_click)

        boxp = QComboBox(groupBox)
        boxp.addItems(self.gem.available_channels())

        colormaps = [i for i in list(AVAILABLE_COLORMAPS.keys()) if i not in ['label_colormap', 'custom']]
        boxc = QComboBox(groupBox)
        boxc.addItems(colormaps)

        self.channelsComboBox = boxp
        self.channelsColormapComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()

        grid.addWidget(QLabel('channel:'), 1, 0)
        grid.addWidget(self.channelsComboBox, 1, 1)
        grid.addWidget(QLabel('colormap:'), 2, 0)
        grid.addWidget(self.channelsColormapComboBox, 2, 1)
        vbox.addLayout(grid)
        vbox.addWidget(btn)
        groupBox.setLayout(vbox)

        self.layout().addWidget(groupBox)

    def createProteinExpressionWidget(self):
        groupBox = QGroupBox(self, title="Protein Expression")

        btn = QPushButton("Add layer")
        btn.clicked.connect(self._on_expr_click)

        boxp = QComboBox(groupBox)
        boxp.addItems(self.gem.proteins)

        colormaps = [i for i in list(AVAILABLE_COLORMAPS.keys()) if i not in ['label_colormap', 'custom']]
        boxc = QComboBox(groupBox)
        boxc.addItems(colormaps)

        self.proteinComboBox = boxp
        self.colormapComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()

        grid.addWidget(QLabel('protein:'), 1, 0)
        grid.addWidget(self.proteinComboBox, 1, 1)
        grid.addWidget(QLabel('colormap:'), 2, 0)
        grid.addWidget(self.colormapComboBox, 2, 1)
        vbox.addLayout(grid)
        vbox.addWidget(btn)
        groupBox.setLayout(vbox)

        self.layout().addWidget(groupBox)

    def createTranscriptsWidget(self):
        groupBox = QGroupBox(self, title="RNA Transcripts")

        btn = QPushButton("Add layer")
        btn.clicked.connect(self._on_rna_click)

        boxp = QComboBox(groupBox)
        boxp.addItems(self.gem.target_labels())

        colors = ['white', 'red', 'green', 'blue',
                  'magenta', 'yellow', 'cyan']
        boxc = QComboBox(groupBox)
        boxc.addItems(colors)

        self.targetsComboBox = boxp
        self.colorsComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()

        grid.addWidget(QLabel('target:'), 1, 0)
        grid.addWidget(self.targetsComboBox, 1, 1)
        grid.addWidget(QLabel('color:'), 2, 0)
        grid.addWidget(self.colorsComboBox, 2, 1)
        vbox.addLayout(grid)
        vbox.addWidget(btn)
        groupBox.setLayout(vbox)

        self.layout().addWidget(groupBox)

    def createMetadataWidget(self):
        groupBox = QGroupBox(self, title="Color Cells")

        boxc = QComboBox(groupBox)
        boxc.toolTip = "Open a _metadata.csv file to populate dropdown"
        if self.gem.metadata is not None:
            boxc.addItems([i for i in self.gem.metadata.columns if i not in ['cell_ID', 'fov', 'CellId']])
        else:
            boxc.addItems([])

        self.metaComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()

        grid.addWidget(QLabel('column:'), 1, 0)
        grid.addWidget(self.metaComboBox, 1, 1)
        vbox.addLayout(grid)

        listl = QListWidget(groupBox)
        listl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.labelListWidget = listl
        self.showSelectedCheckbox = QCheckBox('Only show selected labels')
        self.showSelectedCheckbox.stateChanged.connect(self._show_selected_changed)

        self.labelListWidget.setHidden(True)
        self.showSelectedCheckbox.setHidden(True)
        self._meta_changed(self.metaComboBox.currentText())
        if self.gem.cells_layer is not None:
            self.gem.cells_layer.visible = False
        self.metaComboBox.currentTextChanged.connect(self._meta_changed)

        listl.itemSelectionChanged.connect(self._labels_selected)
        vbox.addWidget(self.labelListWidget)
        vbox.addWidget(self.showSelectedCheckbox)
        groupBox.setLayout(vbox)
        self.layout().addWidget(groupBox)

    def updateLabelsWidget(self, meta_col):
        self.labelListWidget.clear()
        vals = sorted(np.unique(self.gem.metadata[meta_col]))
        cols = label_colormap(len(vals) + 1).colors
        for i, n in enumerate(vals):
            pmap = QPixmap(24, 24)
            rgba = cols[i + 1]
            color = np.round(255 * rgba).astype(int)
            pmap.fill(QColor(*list(color)))
            icon = QIcon(pmap)
            qitem = QListWidgetItem(icon, str(n), self.labelListWidget)