# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:56:22 2017

@author: cdw2be
"""

# Import Classes
import mrimodel
import mesh
import confocalmodel
import time

# Define import variables
scar = False
dense = False
time_point = 12
dense_time_point = 6

# Define plot variables
plot_cine_seg = False
plot_scar_seg = False
plot_dense_pts = False
plot_cine_mesh = False
plot_scar_mesh = False
disp_scar_nodes = False
disp_nodes = False
gen_fe_file = False
open_postview = False

# Create a new instance of the MRI Model classe
heartModel = mrimodel.MRIModel(scar, dense)

# Create a new instance of the Confocal Model class
confocalModel = confocalmodel.ConfocalModel()
#image_channels = confocalModel.splitImageChannels()
image_positions = confocalModel.stitchImages()

# Import the cine data and do not plot the stack
heartModel.importCine(timepoint=time_point)

# Import the LGE data and do not plot the stack
if scar: heartModel.importLGE()

# Import the DENSE data and get it back
if dense: heartModel.importDense(align_timepoint=time_point)

# Align modalities
if scar: heartModel.alignScarCine(time_point)
if dense: heartModel.alignDense(cine_timepoint = time_point)

# Create 2 instance of the Mesh class
cineMesh = mesh.Mesh()

# Fit contours to form the mesh matrices
cine_endo_mat, cine_epi_mat = cineMesh.fitContours(heartModel.cine_endo[time_point], heartModel.cine_epi[time_point], heartModel.cine_apex_pt, heartModel.cine_basal_pt, heartModel.cine_septal_pts, '4x2')

# Implement segment rendering of the data
if plot_cine_seg:
	cineSegAxes = cineMesh.segmentRender(heartModel.cine_endo[time_point], heartModel.cine_epi[time_point], heartModel.cine_apex_pt, heartModel.cine_basal_pt, heartModel.cine_septal_pts)

if plot_scar_seg:
	cineSegAxes = cineMesh.displayScarTrace(heartModel.aligned_scar[time_point], ax=cineSegAxes)

if plot_dense_pts:
	cineSegAxes = cineMesh.displayDensePts(heartModel.dense_aligned_pts, heartModel.dense_slices, heartModel.dense_aligned_displacement, dense_plot_quiver=1, timepoint=dense_time_point, ax=cineSegAxes)

#Surface Render Code
# Display the mesh surface
if plot_cine_mesh:
	cineMeshAxes = cineMesh.surfaceRender(cine_endo_mat)
	cineMeshAxes = cineMesh.surfaceRender(cine_epi_mat, cineMeshAxes)

# Get the mesh as cartesian and prolate
cineMeshCart, cineMeshProl = cineMesh.feMeshRender()

# Perform node ordering
nodes = cineMesh.nodeNum(cineMeshCart[0], cineMeshCart[1], cineMeshCart[2])

# Get the HEX and PENT element connectivity matrices from the cine mesh
lv_hex, lv_pent = cineMesh.getElemConMatrix()

# Assign scar parameter to elements
if scar:
	scar_nodes, scar_elems = cineMesh.assignScarElems(heartModel.aligned_scar[time_point], conn_mat = 'hex')
	if disp_scar_nodes: cineMeshAxes = cineMesh.nodeRender(nodes[scar_nodes, :], ax=cineSegAxes)
	
# Assign DENSE values to elements
if dense:
	elem_displacements_x, elem_displacements_y = cineMesh.assignDenseElems(heartModel.dense_aligned_pts, heartModel.dense_slices, heartModel.dense_aligned_displacement)

# Plot the element edges to determine scar location
if disp_nodes: cineMeshAxes = cineMesh.nodeRender(nodes, cineMeshAxes) if plot_cine_mesh else cineMesh.nodeRender(nodes)

if gen_fe_file: fe_file = cineMesh.generateFEFile(heartModel.cine_file)

if open_postview: cineMesh.displayMeshPostview(fe_file)