import mrimodel
import confocalmodel
import mesh
import numpy as np
import warnings
from cardiachelpers import displayhelper
import matplotlib.pyplot as mplt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors

mesh_filename = 'C:/Users/cdw2be/Downloads/Code4Chris/data/LVGEOM_8x4_noshift.mat'
feb_filename = 'C:/Users/cdw2be/Downloads/Code4Chris/data/temp_feb.feb'

num_rings = 24
elem_per_ring = 48
elem_in_wall = 5
mesh_type = '4x8'
time_point = 0

mri_mesh = mesh.Mesh(num_rings, elem_per_ring, elem_in_wall)
mri_mesh.importPremadeMesh(mesh_filename)
mri_mesh.generateFEFile(feb_filename)

sa_filename = 'C:/Users/cdw2be/Downloads/Code4Chris/data/SA_LGE_Scar_Pnpts.mat'
la_pinpt_filename = 'C:/Users/cdw2be/Downloads/Code4Chris/data/LA_LGE_2CH_Pnpts.mat'
la_lge_filenames = ['C:/Users/cdw2be/Downloads/Code4Chris/data/LA_LGE_2CH_Scar.mat', 'C:/Users/cdw2be/Downloads/Code4Chris/data/LA_LGE_3CH_Scar.mat', 'C:/Users/cdw2be/Downloads/Code4Chris/data/LA_LGE_4CH_Scar.mat']

mri_model = mrimodel.MRIModel(sa_filename, la_pinpt_filename, sa_scar_file=sa_filename, la_scar_files=la_lge_filenames)
mri_model.importCine()
mri_model.importLGE()
mri_model.importScarLA()

mri_model.convertDataProlate(mri_mesh.focus)
mri_mesh.rotateNodesProlate()

mri_model.alignScar()
depth_smooth_list = [0.05*i for i in range(101)]
scar_elem_list = [None]*len(depth_smooth_list)
for depth_i, depth_smooth in enumerate(depth_smooth_list):
	scar_elem_list[depth_i] = mri_mesh.interpScarData(mri_model.interp_data, depth_smooth=depth_smooth)