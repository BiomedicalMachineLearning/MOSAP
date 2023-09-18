import glob
import io
from pathlib import Path
from typing import Any, Union, Optional  # Meta
from typing import Iterable, Sequence, Mapping, MutableMapping  # Generic ABCs
from typing import Tuple, List
import tifffile
import re
import logging
from mosap.mosadata import MOSADATA


def extract_physical_dimension(self, ome_tiff_path):
        """ A function to load the original OME tiff to extract micron resolution and pixel conversion"""
        """ return two dictionaries: one for unit conversion and the other for channel2name"""

        import xml.etree.ElementTree
        tiff_image = tifffile.TiffFile(ome_tiff_path)
        omexml_string = tiff_image.pages[0].description
        root = xml.etree.ElementTree.parse(io.StringIO(omexml_string))
        namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        channels = root.findall('ome:Image[1]/ome:Pixels/ome:Channel', namespaces)
        channel_names = [c.attrib['Name'] for c in channels]
        resolution = root.findall('ome:Image[1]/ome:Pixels', namespaces)
        attribute = resolution[0]
        
        resolution_unit = dict()
        resolution_unit['original_X_micron'] = float(attribute.attrib['SizeX']) * float(attribute.attrib['PhysicalSizeX'])
        resolution_unit['original_Y_micron'] = float(attribute.attrib['SizeY']) * float(attribute.attrib['PhysicalSizeY'])
        resolution_unit['original_X_pixel'] = int(attribute.attrib['SizeX']) 
        resolution_unit['original_Y_pixel'] = int(attribute.attrib['SizeY']) 
        return resolution_unit, channel_names

def convert_micron2pixel(x_micron, micron_dim, scale_dim):
    """ convert the annotation box coordinate from micron unit to pixel unit
    
    ox_px_coord = convert_micron2pixel(ox_coords, 6276.93, im_demo.shape[1])
    oy_px_coord = convert_micron2pixel(oy_coords, 15235.53, im_demo.shape[0])
    """
    return (x_micron*scale_dim/micron_dim)

def convert_unit_micron2pixel(spomic:MOSADATA, original_width_micron, original_height_micron, 
                                  X_col:Optional[str]=None,Y_col:Optional[str]=None):
        """ Running the conversion of the unit from micron unit to pixel image """
        """ In case you do not have the conversion unit please use the original OME.tiff extract_physical_dimension"""
        if spomic.original_coord_unit != spomic.ref_image_unit:
            logging.warning(f'Converting {spomic.original_coord_unit} to {spomic.ref_image_unit}')
            
        if X_col:
            spomic.obs['X_px'] = convert_micron2pixel(spomic.obs[X_col], original_width_micron, 
                                                      spomic.ref_image.shape[1])
        else:
            spomic.obs['X_px'] = convert_micron2pixel(spomic.obs[spomic.__centroid_X], original_width_micron, 
                                                      spomic.ref_image.shape[1])
        
        if Y_col:
            spomic.obs['Y_px'] = convert_micron2pixel(spomic.obs[Y_col], original_height_micron, 
                                                      spomic.ref_image.shape[0])
        else:
            spomic.obs['Y_px'] = convert_micron2pixel(spomic.obs[spomic.__centroid_Y], original_height_micron, 
                                                      spomic.ref_image.shape[0])
        logging.warning(f'X_px and Y_px are added to meta_vars')