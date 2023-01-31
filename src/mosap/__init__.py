__version__ = "0.0.2"

CosMX_PROTEIN_EXP_DIR = "DecodedExpMasks"
CosMX_PROTEIN_EXP_ALT_DIR = "DecodedProteinExp"
CosMX_PROTEIN_EXP_ALT_ALT_DIR = "LOD_Overlay"

CosMX_ALPHA_UM_PER_PX = 0.168
CosMX_ALPHA_MM_PER_PX = CosMX_ALPHA_UM_PER_PX/1000
CosMX_ALPHA_PX_PER_MM = 1/CosMX_ALPHA_MM_PER_PX

CosMX_UM_PER_PX = 0.18
CosMX_MM_PER_PX = CosMX_UM_PER_PX/1000
CosMX_PX_PER_MM = 1/CosMX_MM_PER_PX


from ._reader import napari_get_reader
from ._widget import MultiOmicRegistrationWidget , Transcript_Selection_Widget, Heterogeneity_Vis_widget

__all__ = (
    "napari_get_reader",
    "MultiOmicRegistrationWidget",
    "Transcript_Selection_Widget",
    "Heterogeneity_Vis_widget"
)
