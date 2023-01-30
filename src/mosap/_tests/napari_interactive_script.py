import pandas as pd
import numpy as np

wiewer.open('/Volumes/BiomedML/Projects/MOSAP/demo_data/HE/CR-014 (1, 35531, 80237, 2611, 2482).png', plugin="napari-aicsimageio")


point_ox = pd.read_csv('/Volumes/BiomedML/Projects/MOSAP/demo_data/Anndata/point_ox_coord.csv', index_col = 0)
point_oy = pd.read_csv('/Volumes/BiomedML/Projects/MOSAP/demo_data/Anndata/point_oy_coord.csv', index_col = 0)
point_ox = list(point_ox['0'])
point_oy = list(point_oy['0'])
points = np.array(list(zip(point_ox,point_oy)))

points_layer = viewer.add_points(points, size=10, name='IMC_cells')
points_layer = viewer.add_points(points, size=20, face_color='green')

