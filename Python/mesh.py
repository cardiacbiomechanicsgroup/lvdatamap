# -*- coding: utf-8 -*-
"""
Creates the class and functions for implementing mesh formation and visualization.
Based on MRI Processing pipeline by Thien-Khoi Phung.

Created on Wed Oct 4 2:54:15 2017

@author: cdw2be
"""

# Imports
import numpy as np
import scipy as sp
import math
from cardiachelpers import stackhelper
from cardiachelpers import mathhelper
from cardiachelpers import meshhelper
from cardiachelpers import importhelper
import matplotlib.pyplot as plt
import warnings

class Mesh():

	"""Defines meshes based on endocardial and epicardial contours from MRIModel objects.
	
	This class can fit contours to a prolate mesh, and visualize the meshes and contours
		for simplified interpretation and inspection. Also can be used to generate the 
		element connectivity matrices to be passed in to FEBio.
		
	Attributes:
		num_rings (int): Number of elements in the longitudinal axis of the mesh.
		elem_per_ring (int): Number of elements circumferentially around the wall.
		elem_in_wall (int): Number of elements radially through the wall.
		endo_node_matrix (array): The nodal mesh matrix from bicubic fitting for endocardial contour.
		epi_node_matrix (array): The nodal mesh matrix from bicubic fitting for epicardial contour.
		focus (float): The focus point to be used for the prolate spheroid calculations.
		origin (array): The shift values for x, y, z to center at 0.
		transform (array): The second element of the shifting process to center cartesian at the origin.
		hex (array): The element connectivity matrix for hexahedron-shaped elements
		pent (array): The element connectivity matrix for pentahedron-shaped elements
		
	TODO:
		Align multiple meshes (define as a class function returning a transformation matrix).
	"""

	def __init__(self, num_rings=14, elem_per_ring=24, elem_in_wall=5):
		"""Initialization method for new Mesh class instances.
		
		args:
			num_rings (int): The number of rings vertically along the heart.
			elem_per_ring (int): The number of elements circumferentially per ring.
			elem_in_wall (int): The number of layers through the wall.
		"""
		# Set class variables based on passed values and initialize object fields
		self.num_rings = num_rings
		self.elem_per_ring = elem_per_ring
		self.elem_in_wall = elem_in_wall
		self.endo_node_matrix = []
		self.epi_node_matrix = []
		self.focus = []
		self.transform = []
		self.origin = []
		self.nodes = []
		self.hex = []
		self.pent = []
		self.scar_elems = []
		self.dense_x_displacements = []
		self.dense_y_displacements = []
		self.dense_radial_strain = []
		self.dense_circumferential_strain = []
		self.nodes_in_scar = np.array([])
		self.elems_in_scar = np.array([])
		self.elems_out_scar = np.array([])
	
	def importPremadeMesh(self, mesh_file):
		"""A modified version of the mesh to allow importing premade mesh structures.
		"""
		try:
			# Pull data from the .mat premade mesh file
			lv_geom = importhelper.loadmat(mesh_file)['LVGEOM']
			# Correct for indexing differences between MATLAB and Python
			self.hex = np.subtract(np.array(lv_geom['eHEX']), 1)
			self.pent = np.subtract(np.array(lv_geom['ePENT']), 1)
			# Pull mesh-centric information
			self.nodes = np.array(lv_geom['nXYZ'])
			self.focus = lv_geom['focus']
			self.num_rings, self.elem_per_ring, self.elem_in_wall = lv_geom['LVLCR']
			self.epi_node_list = np.subtract(lv_geom['eEPI'], 1)
			self.epi_nodes = self.hex[self.epi_node_list, :][:, [2, 3, 6, 7]]
			epi_node_fitting = self.nodes[np.unique(self.epi_nodes.flatten()), :]
			# Interpret the epicardial and endocardial node matrix surface
			num_nodes_per_layer = np.unique(self.epi_nodes.flatten()).shape[0]
			start_nodes = [layer_num * (num_nodes_per_layer + 1) for layer_num in range(self.elem_in_wall+2)]
			nodes_by_layer = [list(range(start_nodes[i], start_nodes[i+1])) for i in range(len(start_nodes)-1)]
			endo_node_fitting = self.nodes[nodes_by_layer[2], :]
			# Recreate node matrices using bicubic fitting method
			self.epi_node_matrix, _ = meshhelper.fitBicubicData(epi_node_fitting, self.focus, mesh_density='4x8')
			self.endo_node_matrix, _ = meshhelper.fitBicubicData(endo_node_fitting, self.focus, mesh_density='4x8')
			return(True)
		except FileNotFoundError:
			print('Premade mesh file not found.')
			return(False)
	
	def assignInsertionPts(self, apex_pt, basal_pt, septal_pts):
		"""Set the internal transform array from user-defined points to align geometry.
		"""
		# Define a local norm calculation function.
		calcNorm = lambda arr_in: np.sqrt(np.sum(np.square(arr_in)))
		
		# Calculate the e1 basis from the apical and basal points.
		c = apex_pt - basal_pt
		c_norm = calcNorm(c)
		self.origin = basal_pt + c/3
		e1_basis = [c1 / c_norm for c1 in c]
		
		# Define the e2 basis from the septal points.
		d1 = septal_pts[0, :] - self.origin
		d2 = d1 - [np.dot(d1, e1_basis)*e1_elem for e1_elem in e1_basis]
		e2_basis = d2 / calcNorm(d2)
		
		# Calculate the e3 basis from the other 2 bases.
		e3 = np.cross(e1_basis, e2_basis)
		e3_basis = e3 / calcNorm(e3)
		
		# Set up the transform.
		self.transform = np.array([e1_basis, e2_basis, e3_basis])
		return(True)
	
	def rotateNodesProlate(self):
		"""Rotates the nodes into prolate spheroidal coordinates.
		"""
		nodes_prol_list = mathhelper.cart2prolate(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], self.focus)
		self.nodes_prol = np.column_stack(tuple(nodes_prol_list))
		return(self.nodes_prol)
	
	def fitContours(self, all_data_endo, all_data_epi, apex_pt, basal_pt, septal_pts, mesh_type):
		"""Function to Perform the Contour Fitting from the Passed Endo and Epi Contour data.
		
		args:
			all_data_endo (array): Full stack of endocardial contour data contained in an MRIModel object.
			all_data_epi (array): Full stack of epicardial contour data contained in an MRIModel object.
			apex_pt (array): Apical point by long-axis of an MRIModel object.
			basal_pt (array): Basal point by long-axis of an MRIModel object.
			septal_pts (array): Septal points by short-axis of an MRIModel object.
			mesh_type (string): String indicating the mesh type to be used in contour fitting.
		returns:
			endo_node_matrix (array): An array of the endocardial nodes defining the mesh.
			epi_node_matrix (array): An array of the epicardial nodes defining the mesh.
		"""
		
		# Set up variables
		data_endo, self.focus, self.transform, self.origin = stackhelper.rotateDataCoordinates(all_data_endo, apex_pt, basal_pt, septal_pts)
		data_epi = stackhelper.rotateDataCoordinates(all_data_epi, apex_pt, basal_pt, septal_pts)[0]
		
		# Fit using a bicubic interpolation.
		self.endo_node_matrix, _ = meshhelper.fitBicubicData(data_endo, self.focus, mesh_density=mesh_type)
		self.epi_node_matrix, _ = meshhelper.fitBicubicData(data_epi, self.focus, mesh_density=mesh_type)
		
		return([self.endo_node_matrix, self.epi_node_matrix])
	
	def feMeshRender(self):
		"""Interpolate element density mesh based on endo and epi surfaces.
		
		Does not call any values, operates on the fields of the object.
		returns:
			meshCart (array): The interpolated mesh in cartesian coordinates.
			meshProl (array): The interpolated mesh in prolate coordinates.
		"""
		# Set up initial variables
		endo_fit = self.endo_node_matrix.copy()
		epi_fit = self.epi_node_matrix.copy()
		rads = math.pi/180
		
		# Sort Rows and Do Unit Conversion to Radians
		endo_fit = endo_fit[endo_fit[:, 0].argsort()]
		endo_fit = endo_fit[endo_fit[:, 1].argsort(kind='mergesort')]
		endo_fit = endo_fit[endo_fit[:, 2].argsort(kind='mergesort')]
		
		epi_fit = epi_fit[epi_fit[:, 0].argsort()]
		epi_fit = epi_fit[epi_fit[:, 1].argsort(kind='mergesort')]
		epi_fit = epi_fit[epi_fit[:, 2].argsort(kind='mergesort')]
		
		endo_fit[:, 1:3] *= rads
		epi_fit[:, 1:3] *= rads
		
		# Determine the size of the bicubic mesh
		m, n = endo_fit.shape
		mu_num = np.where(endo_fit[:, 2] == 0)[0].size
		th_num = int(m / mu_num)
		
		# Pad Nodal Mesh
		endo_fit = np.append(endo_fit, endo_fit[:mu_num, :], axis=0)
		endo_fit[m:, 2] = 2*math.pi
		epi_fit = np.append(epi_fit, epi_fit[:mu_num, :], axis=0)
		epi_fit[m:, 2] = 2*math.pi
		th_num += 1
		
		# Reshape mesh for interp function
		endo_data = np.reshape(endo_fit, (mu_num, th_num, n), order='F')
		epi_data = np.reshape(epi_fit, (mu_num, th_num, n), order='F')
		
		# Set up surface mesh for interpolation
		mu_vec = endo_fit[:mu_num, 1]
		
		# Pull unique theta values:
		sort_by_mu = endo_fit[endo_fit[:, 2].argsort()]
		sort_by_mu = sort_by_mu[sort_by_mu[:, 1].argsort(kind='mergesort')]
		th_vec = sort_by_mu[:th_num, 2]
		
		# Mesh Grid of Unique Mu and Theta for OG Nodes
		mu_grid_og, th_grid_og = np.meshgrid(mu_vec, th_vec)
		
		# New mesh to interpolate based on mesh density
		mu_vec_fe = np.linspace(0, mu_grid_og.flatten()[-1], self.num_rings+2)
		th_vec_fe = np.linspace(0, th_grid_og.flatten()[-1], self.elem_per_ring+1)
		th_vec_fe = th_vec_fe[1:]
		mu_grid_fe, th_grid_fe = np.meshgrid(mu_vec_fe, th_vec_fe)
		
		# Interpolate Along New Surface Grid
		endo_interp_surf = meshhelper.biCubicInterp(mu_grid_fe, th_grid_fe, endo_data)
		epi_interp_surf = meshhelper.biCubicInterp(mu_grid_fe, th_grid_fe, epi_data)
		
		# Convert Endo and Epi Interp to Cartesian
		endo_mu = endo_interp_surf[:, :, 0]
		endo_nu = endo_interp_surf[:, :, 1]
		endo_phi = endo_interp_surf[:, :, 2]
		
		endo_x, endo_y, endo_z = mathhelper.prolate2cart(endo_mu, endo_nu, endo_phi, self.focus)
		
		epi_mu = epi_interp_surf[:, :, 0]
		epi_nu = epi_interp_surf[:, :, 1]
		epi_phi = epi_interp_surf[:, :, 2]
		epi_x, epi_y, epi_z = mathhelper.prolate2cart(epi_mu, epi_nu, epi_phi, self.focus)
		
		# Interpolate nodes between endo and epi
		step_x = np.reshape((epi_x - endo_x)/self.elem_in_wall, [epi_x.shape[0], epi_x.shape[1], 1])
		step_y = np.reshape((epi_y - endo_y)/self.elem_in_wall, [epi_x.shape[0], epi_x.shape[1], 1])
		step_z = np.reshape((epi_z - endo_z)/self.elem_in_wall, [epi_x.shape[0], epi_x.shape[1], 1])
		
		endo_x = np.reshape(endo_x, [endo_x.shape[0], endo_x.shape[1], 1])
		endo_y = np.reshape(endo_y, [endo_y.shape[0], endo_y.shape[1], 1])
		endo_z = np.reshape(endo_z, [endo_z.shape[0], endo_z.shape[1], 1])
		
		epi_x = np.reshape(epi_x, [epi_x.shape[0], epi_x.shape[1], 1])
		epi_y = np.reshape(epi_y, [epi_y.shape[0], epi_y.shape[1], 1])
		epi_z = np.reshape(epi_z, [epi_z.shape[0], epi_z.shape[1], 1])
		
		# Create stepnumber matrix
		steps = np.zeros([endo_x.shape[0], endo_x.shape[1], self.elem_in_wall+1])
		for jz in range(self.elem_in_wall+1):
			steps[:, :, jz] = jz
		
		# Reformat x, y, z arrays
		x = np.tile(endo_x, (1, 1, self.elem_in_wall + 1)) + np.tile(step_x, (1, 1, self.elem_in_wall+1))*steps
		y = np.tile(endo_y, (1, 1, self.elem_in_wall + 1)) + np.tile(step_y, (1, 1, self.elem_in_wall+1))*steps
		z = np.tile(endo_z, (1, 1, self.elem_in_wall + 1)) + np.tile(step_z, (1, 1, self.elem_in_wall+1))*steps
		
		# Convert cartesian back to prolate
		m, n, p = mathhelper.cart2prolate(x, y, z, self.focus)
		
		# Rotate each matrix by 90 degrees
		x = np.rot90(x)
		y = np.rot90(y)
		z = np.rot90(z)
		m = np.rot90(m)
		n = np.rot90(n)
		p = np.rot90(p)
		
		# Define the return matrices
		self.meshCart = [x, y, z]
		self.meshProl = [m, n, p]
		
		return([self.meshCart, self.meshProl])
	
	def nodeNum(self, x, y, z):
		"""Generate node corners as x, y, z points
		
		Numbering order:
			Theta, to 2*pi
			Endocardial to epicardial
			Basal to apical
		returns:
			nodes (array): x, y, z coordinates for each node, ordered by row
				Formatted as (Basal to Apex, y, x)
		"""
		# Get the shape of the x array to determine number of elements (circ, long, rad)
		c, l, r = x.shape
		
		nodes = None
		for lyr in range(r):
			# Pull coordinates from each layer and flip so the first node is the top row
			# Reshape to a column array going from 0 to 360 degrees and basal to apical
			x_lyr = np.reshape(np.fliplr(x[:, :, lyr]), [c*l, 1], order='F')
			y_lyr = np.reshape(np.fliplr(y[:, :, lyr]), [c*l, 1], order='F')
			z_lyr = np.reshape(np.fliplr(z[:, :, lyr]), [c*l, 1], order='F')
			# Store in nodes and remove all apical nodes except one
			if nodes is None:
				nodes = np.column_stack([x_lyr[:-(c-1)], y_lyr[:-(c-1)], z_lyr[:-(c-1)]])
			else:
				nodes = np.append(nodes, np.column_stack([x_lyr[:-(c-1)], y_lyr[:-(c-1)], z_lyr[:-(c-1)]]), axis=0)
		self.nodes = nodes
		return(nodes)
		
	def getElemConMatrix(self):
		"""Get the connectivity matrix for the nodes to define the elements.
		
		Operates on the instance variables defined at creations. The matrix defines
		an element per row, nodes are defined by index in the nodes array.
		returns:
			hex (array): Array for a 6-sided element per node.
			pent (array): Array for a 5-sided element per node.
		"""
		# Calculate the number of nodes and element in each peel layer
		nodes_per_layer = (self.num_rings + 1)*self.elem_per_ring + 1
		elem_per_layer = self.num_rings*self.elem_per_ring
		
		# Lay out the two arrays
		hex = np.zeros([elem_per_layer*self.elem_in_wall, 8])
		pent = np.zeros([self.elem_per_ring*self.elem_in_wall, 6])
		for l in range(self.elem_in_wall):
			# Calculate number of elements
			enum = range(self.elem_per_ring*l + 1, self.elem_per_ring*(l+1) + 1)
			for n in range(elem_per_layer):
				# Calculate which nodes define each element in hex
				nn = range(elem_per_layer*l + 1, elem_per_layer*(l+1) + 1)
				sn = nodes_per_layer*l + n + 1
				hex[nn[n]-1, :] = [sn, sn+1, sn+nodes_per_layer+1, sn+nodes_per_layer, sn+self.elem_per_ring, sn+self.elem_per_ring+1, sn+nodes_per_layer+self.elem_per_ring+1, sn+nodes_per_layer+self.elem_per_ring] if math.fmod((n+1), self.elem_per_ring) != 0 else [sn, sn+1-self.elem_per_ring, sn+nodes_per_layer+1-self.elem_per_ring, sn+nodes_per_layer, sn+self.elem_per_ring, sn+1, sn+nodes_per_layer+1, sn+nodes_per_layer+self.elem_per_ring]
			for n in range(self.elem_per_ring):
				# Calculate which nodes define each element in pent
				pent[enum[n]-1, :] = [nodes_per_layer*(l+2), nodes_per_layer*(l+2)-self.elem_per_ring+n, nodes_per_layer*(l+2)-self.elem_per_ring+n+1, nodes_per_layer*(l+1), nodes_per_layer*(l+1)-self.elem_per_ring+n, nodes_per_layer*(l+1)-self.elem_per_ring+n+1] if math.fmod((n+1), self.elem_per_ring) != 0 else [nodes_per_layer*(l+2), nodes_per_layer*(l+2)-self.elem_per_ring+n, nodes_per_layer*(l+2)-self.elem_per_ring, nodes_per_layer*(l+1), nodes_per_layer*(l+1)-self.elem_per_ring+n, nodes_per_layer*(l+1)-self.elem_per_ring]
		
		# Subtract 1 from each element because of zero-indexing in python
		hex -= 1
		pent -= 1
		
		hex = hex.astype(int)
		pent = pent.astype(int)
		
		# Assign to instance variables
		self.hex = hex
		self.pent = pent
		
		self.epi_node_list = np.array(range(int(hex.shape[0] - (hex.shape[0] / self.elem_in_wall)), int(hex.shape[0])))
		self.epi_nodes = self.hex[self.epi_node_list, :][:, [2, 3, 6, 7]]
		
		return([hex, pent])
	
	def assignScarElems(self, scar_contour, conn_mat='hex'):
		"""Get a matrix indicating which elements are contained within the scar contour
		
		args:
			scar_contour (array): The x, y, z array for the scar contour, from the MRIModel object
			conn_mat (string): Indicates which connectivity matrix to use (hex or pent)
		returns:
			scar_node_inds (array): The indices of the element connectivity matrix that are scar
			center_matrix (array): The center of the elements defined by conn_mat that are in the scar
		"""
		# Set number of nodes needed for element to be in scar
		num_nodes = 1
		
		# Convert the scar contour to prolate coordinates
		scar_contour = [scar_contour_i for scar_contour_i in scar_contour if scar_contour_i.size > 0]
		num_slices = len(scar_contour)
		scar_edge_spacing = int(scar_contour[0].shape[0]/2)
		scar_vstack = np.vstack(tuple(scar_contour))
		scar_vstack = np.dot((scar_vstack - np.array([self.origin for i in range(scar_vstack.shape[0])])), np.transpose(self.transform))
		
		self.nodes_in_scar = meshhelper.assignRegionNodes(self.nodes, scar_vstack, scar_edge_spacing, num_slices, self.focus)
		
		# Get connectivity matrix
		elem_con = self.pent if conn_mat == 'pent' else self.hex
		
		# Determine number of nodes in scar region for each element
		elem_scar_mask = np.reshape(np.in1d(elem_con, self.nodes_in_scar), elem_con.shape)
		elem_node_ct = np.sum(elem_scar_mask, axis=1)
		self.elems_in_scar = np.where(elem_node_ct >= num_nodes)[0]
		self.elems_out_scar = np.where(elem_node_ct < num_nodes)[0]
		
		return([self.nodes_in_scar, self.elems_in_scar])
	
	def assignDenseElems(self, dense_pts, dense_slices, dense_displacement_all, radial_strain, circumferential_strain, conn_mat='hex'):
		"""Assign DENSE dx and dy values to elements within the field.
		
		Assignment of DENSE is done based on DENSE value at center of element. Interpolation is
		performed via tricubic interpolation in prolate spheroid coordinates.
		"""
		# Establish overhead lists:
		dense_pts_z = [[dense_slices[i] - self.origin[2]]*dense_pts[i].shape[0] for i in range(len(dense_slices))]
		
		# Convert nodes to prolate, calculate centers, and return to cartesian
		nodes_prol_mu, nodes_prol_nu, nodes_prol_phi = mathhelper.cart2prolate(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], self.focus)
		
		elem_con = self.pent if conn_mat == 'pent' else self.hex
		
		elem_cart_centers = [False]*elem_con.shape[0]
		elem_prol_centers = [False]*elem_con.shape[0]
		
		for i in range(elem_con.shape[0]):
			nodes_in_elem = elem_con[i, :]
			elem_prol = np.column_stack((nodes_prol_mu[nodes_in_elem], nodes_prol_nu[nodes_in_elem], nodes_prol_phi[nodes_in_elem]))
			elem_center_prol = np.mean(elem_prol, axis=0)
			elem_prol_centers[i] = [elem_center_prol[0], elem_center_prol[1], elem_center_prol[2]]
			elem_center_cart = mathhelper.prolate2cart(elem_center_prol[0], elem_center_prol[1], elem_center_prol[2], self.focus)
			elem_cart_centers[i] = elem_center_cart
		
		dense_pts_prol = np.array([])
		dense_pts_cart = np.array([])
		
		for slice_num in range(len(dense_slices)):
			cur_slice_pts = np.column_stack((dense_pts[slice_num], dense_pts_z[slice_num]))
			cur_slice_pts = np.dot(cur_slice_pts, np.transpose(self.transform))
			csd_mu, csd_nu, csd_phi = mathhelper.cart2prolate(cur_slice_pts[:, 0], cur_slice_pts[:, 1], cur_slice_pts[:, 2], self.focus)
			if dense_pts_prol.size:
				dense_pts_cart = np.append(dense_pts_cart, cur_slice_pts, axis=0)
				dense_pts_prol = np.append(dense_pts_prol, np.column_stack((csd_mu, csd_nu, csd_phi)), axis=0)
			else:
				dense_pts_prol = np.column_stack((csd_mu, csd_nu, csd_phi))
				dense_pts_cart = cur_slice_pts

		# Find nodes contained within the DENSE measurement volume
		elems_in_dense = np.where((np.min(dense_pts_cart[:, 0]) <= np.array(elem_cart_centers)[:, 0]) & (np.array(elem_cart_centers)[:, 0] <= np.max(dense_pts_cart[:, 0])))[0]
		elem_pts_interp = np.array(elem_prol_centers)[elems_in_dense, :]
		
		# Iterate through each timepoint to calculate an interpolation of DENSE values
		elem_displacements_x = [[np.nan for i in range(len(dense_displacement_all))] for j in range(elem_con.shape[0])]
		elem_displacements_y = [[np.nan for i in range(len(dense_displacement_all))] for j in range(elem_con.shape[0])]
		elem_radial_strain = [[np.nan for i in range(len(radial_strain))] for j in range(elem_con.shape[0])]
		elem_circumferential_strain = [[np.nan for i in range(len(circumferential_strain))] for j in range(elem_con.shape[0])]
		for time_point in range(len(dense_displacement_all)):
			dense_disp_timepoint = dense_displacement_all[time_point]
			rad_strain_timepoint = radial_strain[time_point]
			circ_strain_timepoint = circumferential_strain[time_point]
			dense_cur_disp = np.array([])
			dense_cur_rad = np.array([])
			dense_cur_circ = np.array([])
			for slice_num in range(len(dense_disp_timepoint)):
				if dense_cur_disp.size:
					dense_cur_disp = np.append(dense_cur_disp, dense_disp_timepoint[slice_num], axis=0)
					dense_cur_rad = np.append(dense_cur_rad, rad_strain_timepoint[slice_num], axis=0)
					dense_cur_circ = np.append(dense_cur_circ, circ_strain_timepoint[slice_num], axis=0)
				else:
					dense_cur_disp = dense_disp_timepoint[slice_num]
					dense_cur_rad = rad_strain_timepoint[slice_num]
					dense_cur_circ = circ_strain_timepoint[slice_num]
			# Interpolate element displacements
			elem_interp_dx = sp.interpolate.griddata(dense_pts_prol, dense_cur_disp[:, 0], elem_pts_interp, method='linear')
			elem_interp_dy = sp.interpolate.griddata(dense_pts_prol, dense_cur_disp[:, 1], elem_pts_interp, method='linear')
			elem_interp_nans = np.where(np.isnan(elem_interp_dx))[0]
			elem_interp_dx[elem_interp_nans] = sp.interpolate.griddata(dense_pts_prol, dense_cur_disp[:, 0], elem_pts_interp[elem_interp_nans, :], method='nearest')
			elem_interp_dy[elem_interp_nans] = sp.interpolate.griddata(dense_pts_prol, dense_cur_disp[:, 1], elem_pts_interp[elem_interp_nans, :], method='nearest')
			for elem_num in range(len(elems_in_dense)):
				elem_displacements_x[elems_in_dense[elem_num]][time_point] = elem_interp_dx[elem_num]
				elem_displacements_y[elems_in_dense[elem_num]][time_point] = elem_interp_dy[elem_num]
			# Interpolate element strains
			elem_interp_radial = sp.interpolate.griddata(dense_pts_prol, dense_cur_rad, elem_pts_interp, method='linear')
			elem_interp_circumferential = sp.interpolate.griddata(dense_pts_prol, dense_cur_circ, elem_pts_interp, method='linear')
			elem_interp_radial[elem_interp_nans] = sp.interpolate.griddata(dense_pts_prol, dense_cur_rad, elem_pts_interp[elem_interp_nans, :], method='nearest')
			elem_interp_circumferential[elem_interp_nans] = sp.interpolate.griddata(dense_pts_prol, dense_cur_circ, elem_pts_interp[elem_interp_nans, :], method='nearest')
			for elem_num in range(len(elems_in_dense)):
				elem_radial_strain[elems_in_dense[elem_num]][time_point] = elem_interp_radial[elem_num]
				elem_circumferential_strain[elems_in_dense[elem_num]][time_point] = elem_interp_circumferential[elem_num]
			
		# Do this as a list, set up list on creation with NaNs and fill here. Allows more open adjustments.
		self.dense_x_displacements = np.array(elem_displacements_x)
		self.dense_y_displacements = np.array(elem_displacements_y)
		self.dense_radial_strain = np.array(elem_radial_strain)
		self.dense_circumferential_strain = np.array(elem_circumferential_strain)
			
		return(elem_displacements_x, elem_displacements_y)
	
	def interpScarData(self, interp_data, trans_smooth=0.05, depth_smooth=0):
		non_nan_inds = ~np.isnan(interp_data[:, 3])
		epi_nodes = self.nodes[self.epi_nodes, :]
		epi_nodes_list = [epi_nodes[:, i, :] for i in range(epi_nodes.shape[1])]
		epi_nodes_formatted = np.vstack(tuple(epi_nodes_list))
		_, m_epi, t_epi = mathhelper.cart2prolate(epi_nodes_formatted[:, 0], epi_nodes_formatted[:, 1], epi_nodes_formatted[:, 2], self.focus)
		
		norm_func = lambda x1, x2 : 1
		
		transmurality_rbf = sp.interpolate.Rbf(interp_data[:, 0], interp_data[:, 1], interp_data[:, 2], function='multiquadric', smooth=trans_smooth)
		depth_rbf = sp.interpolate.Rbf(interp_data[non_nan_inds, 0], interp_data[non_nan_inds, 1], interp_data[non_nan_inds, 3], function='multiquadric', smooth=depth_smooth)
		
		scar_tr = transmurality_rbf(t_epi, m_epi)
		scar_dp = depth_rbf(t_epi, m_epi)
		
		scar_tr_reshape = np.transpose(np.reshape(scar_tr, (4, int(scar_tr.size/4))))
		scar_dp_reshape = np.transpose(np.reshape(scar_dp, (4, int(scar_dp.size/4))))
		scar_dp_reshape[np.where(scar_tr_reshape <= 0.0025)] = np.nan
		
		e_epi_tr = np.mean(scar_tr_reshape, axis=1)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			e_epi_wd = np.nanmean(scar_dp_reshape, axis=1)
		
		data_all = np.zeros((self.hex.shape[0]))
		data_all[self.epi_node_list] = e_epi_tr
		
		scar_layers = np.around(e_epi_tr*self.elem_in_wall)
		scar_layers[scar_layers <= 0] = 0
		
		epi_start = np.around(e_epi_wd*self.elem_in_wall)
		epi_start[scar_layers == 0] = np.nan
		
		for cur_ind in range(scar_layers.shape[0]):
			scar_span = scar_layers[cur_ind] + epi_start[cur_ind]
			if scar_span > self.elem_in_wall:
				scar_layers[cur_ind] = scar_layers[cur_ind] - 1
		
		elem_per_layer = int(self.num_rings) * int(self.elem_per_ring)
		
		scar_elems = []
		for start_val in range(max(epi_start.astype(np.int32))):
			outer_scar = self.epi_node_list[epi_start == start_val] - (elem_per_layer*start_val)
			through_wall_scar = scar_layers[epi_start == start_val]
			for wh in range(outer_scar.size):
				for by in range(int(through_wall_scar[wh])):
					scar_elems = np.append(scar_elems, outer_scar[wh] - elem_per_layer*by)
		scar_elems = scar_elems.astype(int)
		self.elems_in_scar = scar_elems
		self.nodes_in_scar = np.unique(self.hex[scar_elems, :].flatten())
		return(scar_elems)
	
	def getElemData(self, elem_list, data_out, average=True, timepoint=0):
		"""Get specified data about the listed elements, returned as an array based on the type of data requested.
		
		args:
			elem_list: A list of element indices to be references for the desired data.
			data_out: A string representing the kind of information requested (DENSE, ie).
			average: A boolean about whether or not to average the data across the elements or return data from every element (default True).
			timepoint: Only used for certain data, but selects the timepoint desired (default 0).
		"""
		# A list of currently available data types
		data_types = ['dense']
		# Compare available data types to requested type to determine availability
		if data_out not in data_types:
			raise('Data type not present in the current model. Please select a different inquiry.')
		# Pre-allocate a list for each element
		elem_data_list = [None]*len(elem_list)
		# If the requested data type is DENSE
		if data_out == 'dense':
			# Determine if timepoint is valid and convert to appropriate data type
			if timepoint != int(timepoint):
				raise('Timepoint must be an integer value for dense interpretation.')
			else:
				timepoint = int(timepoint)
			# Iterate through the elements and collect the DENSE-based strain information
			for elem_ind, elem in enumerate(elem_list):
				elem_dense_val = [self.dense_radial_strain[elem][timepoint], self.dense_circumferential_strain[elem][timepoint]]
				elem_data_list[elem_ind] = elem_dense_val
			# Convert the element data to an array for manipulation
			elem_data_arr = np.asarray(elem_data_list)
			# Average or don't average the data, then return it
			if average:
				return(np.nanmean(elem_data_arr, axis=0))
			else:
				return(elem_data_arr)
	
	def generateFEFile(self, file_name, conn_mat='hex'):
		"""Generate FEBio output file for use in FEBio and PostView.
		"""
		# XML Version String
		xml_ver_str = '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
		
		# Set up febio xml tags
		fe_start_str = '<febio_spec version="2.0">\n'
		fe_end_str = '</febio_spec>'
		
		# Set up Module section strings
		module_str = '\t<Module type="solid"/>\n'
		
		# Set up Material section strings
		material_start_str = '\t<Material>\n'
		material_end_str = '\t</Material>\n'
		material_child_str = '\t\t<material id="1" name="Material 1" type="trans iso Mooney-Rivlin">\n'
		material_child_end_str = '\t\t</material>\n'

		# Set up Geometry strings
		geometry_start_str = '\t<Geometry>\n'
		geometry_end_str = '\t</Geometry>\n'
		node_start_str = '\t\t<Nodes>\n'
		node_end_str = '\t\t</Nodes>\n'
		type_str = 'pent6' if conn_mat == 'pent' else 'hex8'
		element_start_str = '\t\t<Elements type="{}" mat="1" elset="Part14">\n'.format(type_str)
		element_end_str = '\t\t</Elements>\n'
		
		# Set up node base string
		node_str = '\t\t\t<node id="{}">{},{},{}</node>\n'
		elem_str = '\t\t\t<elem id="{}">{},{},{},{},{},{}</elem>\n' if conn_mat == 'pent' else '\t\t\t<elem id="{}">{},{},{},{},{},{},{},{}</elem>\n'
		
		# Select connectivity matrix based on selection
		conn_mat_nodes = self.pent if conn_mat == 'pent' else self.hex
		
		# Set up formatted string lists
		node_strings_formatted = [node_str.format(i+1, self.nodes[i, 0], self.nodes[i, 1], self.nodes[i, 2]) for i in range(self.nodes.shape[0])]
		elem_strings_formatted = [elem_str.format(*tuple(np.insert(conn_mat_nodes[i, :] + 1, 0, i+1))) for i in range(conn_mat_nodes.shape[0])]
		
		# Create file
		output_file_name = file_name.replace('.mat', '.feb')
		with open(output_file_name, 'w') as out_file:
			out_file.writelines([xml_ver_str, fe_start_str, module_str, material_start_str, material_child_str, material_child_end_str])
			out_file.writelines([material_end_str, geometry_start_str, node_start_str])
			out_file.writelines(node_strings_formatted)
			out_file.writelines([node_end_str, element_start_str])
			out_file.writelines(elem_strings_formatted)
			out_file.writelines([element_end_str, geometry_end_str, fe_end_str])
		return(output_file_name)