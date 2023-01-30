__version__ = "0.0.1"

PROTEIN_EXP_DIR = "DecodedExpMasks"
PROTEIN_EXP_ALT_DIR = "DecodedProteinExp"
PROTEIN_EXP_ALT_ALT_DIR = "LOD_Overlay"

ALPHA_UM_PER_PX = 0.168
ALPHA_MM_PER_PX = ALPHA_UM_PER_PX/1000
ALPHA_PX_PER_MM = 1/ALPHA_MM_PER_PX

UM_PER_PX = 0.18
MM_PER_PX = UM_PER_PX/1000
PX_PER_MM = 1/MM_PER_PX


from ._reader import napari_get_reader
from ._widget import ExampleQWidget, example_magic_widget

__all__ = (
    "napari_get_reader",
    "ExampleQWidget",
    "example_magic_widget",
)
