import numpy as np
import scipy as sp
import math
from cardiachelpers import mathhelper
from matplotlib import path

def getLambda(c, e):
	"""Compute Lambda Values and Derivatives by Cubic Hermite Function"""
	
	l = []
	
	h00 = [1 - 3*(e_i**2) + 2*(e_i**3) for e_i in e]
	h10 = [e_i*((e_i-1)**2) for e_i in e]
	h01 = [(e_i**2)*(3-2*e_i) for e_i in e]
	h11 = [(e_i**2)*(e_i-1) for e_i in e]
	
	dh00 = [6*((e_i**2)-e_i) for e_i in e]
	dh10 = [3*(e_i**2)-4*e_i+1 for e_i in e]
	dh01 = [6*(e_i-e_i**2) for e_i in e]
	dh11 = [3*(e_i**2)-2*e_i for e_i in e]
	
	l1 = h00[0]*h00[1]*c[0,0] + h01[0]*h00[1]*c[1,0] + h00[0]*h01[1]*c[2,0] + h01[0]*h01[1]*c[3,0]
	l2 = h10[0]*h00[1]*c[0,3] + h11[0]*h00[1]*c[1,3] + h10[0]*h01[1]*c[2,3] + h11[0]*h01[1]*c[3,3]
	l3 = h00[0]*h10[1]*c[0,4] + h01[0]*h10[1]*c[1,4] + h00[0]*h11[1]*c[2,4] + h01[0]*h11[1]*c[3,4]
	l4 = h10[0]*h10[1]*c[0,5] + h11[0]*h10[1]*c[1,5] + h10[0]*h11[1]*c[2,5] + h11[0]*h11[1]*c[3,5]
	l = np.append(l, l1 + l2 + l3 + l4)
	
	dl1wrt1 = dh00[0]*h00[1]*c[0,0] + dh01[0]*h00[1]*c[1,0] + dh00[0]*h01[1]*c[2,0] + dh01[0]*h01[1]*c[3,0]
	dl2wrt1 = dh10[0]*h00[1]*c[0,3] + dh11[0]*h00[1]*c[1,3] + dh10[0]*h01[1]*c[2,3] + dh11[0]*h01[1]*c[3,3]
	dl3wrt1 = dh00[0]*h10[1]*c[0,4] + dh01[0]*h10[1]*c[1,4] + dh00[0]*h11[1]*c[2,4] + dh01[0]*h11[1]*c[3,4]
	dl4wrt1 = dh10[0]*h10[1]*c[0,5] + dh11[0]*h10[1]*c[1,5] + dh10[0]*h11[1]*c[2,5] + dh11[0]*h11[1]*c[3,5]
	l = np.append(l, dl1wrt1 + dl2wrt1 + dl3wrt1 + dl4wrt1)
	
	dl1wrt2 = h00[0]*dh00[1]*c[0,0] + h01[0]*dh00[1]*c[1,0] + h00[0]*dh01[1]*c[2,0] + h01[0]*dh01[1]*c[3,0]
	dl2wrt2 = h10[0]*dh00[1]*c[0,3] + h11[0]*dh00[1]*c[1,3] + h10[0]*dh01[1]*c[2,3] + h11[0]*dh01[1]*c[3,3]
	dl3wrt2 = h00[0]*dh10[1]*c[0,4] + h01[0]*dh10[1]*c[1,4] + h00[0]*dh11[1]*c[2,4] + h01[0]*dh11[1]*c[3,4]
	dl4wrt2 = h10[0]*dh10[1]*c[0,5] + h11[0]*dh10[1]*c[1,5] + h10[0]*dh11[1]*c[2,5] + h11[0]*dh11[1]*c[3,5]
	l = np.append(l, dl1wrt2 + dl2wrt2 + dl3wrt2 + dl4wrt2)
	
	dl1wrt12 = dh00[0]*dh00[1]*c[0,0] + dh01[0]*dh00[1]*c[1,0] + dh00[0]*dh01[1]*c[2,0] + dh01[0]*dh01[1]*c[3,0]
	dl2wrt12 = dh10[0]*dh00[1]*c[0,3] + dh11[0]*dh00[1]*c[1,3] + dh10[0]*dh01[1]*c[2,3] + dh11[0]*dh01[1]*c[3,3]
	dl3wrt12 = dh00[0]*dh10[1]*c[0,4] + dh01[0]*dh10[1]*c[1,4] + dh00[0]*dh11[1]*c[2,4] + dh01[0]*dh11[1]*c[3,4]
	dl4wrt12 = dh10[0]*dh10[1]*c[0,5] + dh11[0]*dh10[1]*c[1,5] + dh10[0]*dh11[1]*c[2,5] + dh11[0]*dh11[1]*c[3,5]
	l = np.append(l, dl1wrt12 + dl2wrt12 + dl3wrt12 + dl4wrt12)

	return(l)
	
def nearestNodalPoints(z2, z3, nodal_theta, size_nodal_theta, max_nodal_theta, nodal_mu, min_nodal_mu, max_nodal_mu):
	"""Find the 4 nearest nodal points
	Uses a convention from Hashima et al
	
	args:
		z2 (float): mu
		z3 (float): theta
		nodal_theta (array): Theta values for each node
		size_nodal_theta (int): Size of the nodal theta array
		max_nodal_theta (float): Maximum value in nodal theta
		nodal_mu (array): Mu values for each node
		min_nodal_mu (float): Minimum value in nodal mu
		max_nodal_mu (float): Maximum value in nodal mu
	"""
	e = [0, 0]
	# Special theta cases
	if z3 >= 2*math.pi: # If theta is above 2pi
		z3 -= 2*math.pi
		t13 = nodal_theta[1]
		t24 = nodal_theta[0]
		e[0] = (t13 - z3)/(t13 - t24)
		# Indices for theta
		corner13theta = 2
		corner24theta = 1
	elif z3 >= max_nodal_theta: # If theta = 0 is e[0] = 0
		t13 = 2*math.pi
		t24 = max_nodal_theta
		e[0] = (t13 - z3)/(t13 - t24)
		# Indices for theta
		corner13theta = 1
		corner24theta = size_nodal_theta
	else: # General theta case
		min_t = np.where(nodal_theta <= z3)[0].size
		t13 = nodal_theta[min_t]
		t24 = nodal_theta[min_t-1]
		e[0] = (t13 - z3) / (t13 - t24)
		# Indices for theta
		corner13theta = min_t + 1
		corner24theta = min_t
	# Special mu cases
	if z2 == min_nodal_mu: # Smallest mu value possible
		corner12mu = 1
		corner34mu = 2
	elif z2 == max_nodal_mu: # Largest mu value possible
		corner12mu = size_nodal_mu - 1
		corner34mu = size_nodal_mu
	else: # General mu case
		min_m = np.where(nodal_mu < z2)[0].size
		m12 = nodal_mu[min_m - 1]
		m34 = nodal_mu[min_m]
		e[1] = (z2 - m12)/(m34 - m12)
		corner12mu = min_m
		corner34mu = min_m + 1
	return([e, corner13theta, corner24theta, corner12mu, corner34mu])
	
def generateInd(size_nodal_mu, corner13theta, corner24theta, corner12mu, corner34mu, num_nodes):
	"""Generates the ind list based on nearest nodal points and the size of the nodal mu array
	
	args:
		size_nodal_mu (int): Size of nodal mu array
		corner13theta - corner34mu (ints): Indices of nearest nodal points
		num_nodes (int): Total number of nodes
	returns:
		ind (list): List of indices for nearest nodal points
	"""
	ind = []
	# Append nearest nodal points
	ind.append(size_nodal_mu*(corner13theta - 1) + corner12mu - 1)
	ind.append(size_nodal_mu*(corner24theta - 1) + corner12mu - 1)
	ind.append(size_nodal_mu*(corner13theta - 1) + corner34mu - 1)
	ind.append(size_nodal_mu*(corner24theta - 1) + corner34mu - 1)
	ind.append(ind[0] + num_nodes)
	ind.append(ind[1] + num_nodes)
	ind.append(ind[2] + num_nodes)
	ind.append(ind[3] + num_nodes)
	ind.append(ind[4] + num_nodes)
	ind.append(ind[5] + num_nodes)
	ind.append(ind[6] + num_nodes)
	ind.append(ind[7] + num_nodes)
	ind.append(ind[8] + num_nodes)
	ind.append(ind[9] + num_nodes)
	ind.append(ind[10] + num_nodes)
	ind.append(ind[11] + num_nodes)
	# Convert list from floats to ints
	ind = [int(ind_i) for ind_i in ind]
	return(ind)
	
def assignRegionNodes(nodes, region_contour, contour_edge_spacing, num_slices, focus):
	"""Generalized method to label which elements fall within a region contour.
	
	Region contour should be spaced evenly per slice, with an equal number of points on the outer layer and inner layer. The point distance between the circumferential extents of the region should be given with contour edge spacing.
	"""
	# Determine mu error buffer should be used
	err_val = 0
	
	# Convert nodes to cartesian for measurement
	nodes_prol_mu, nodes_prol_nu, nodes_prol_phi = mathhelper.cart2prolate(nodes[:, 0], nodes[:, 1], nodes[:, 2], focus)
	nodes_prol = np.column_stack((nodes_prol_mu, nodes_prol_nu, nodes_prol_phi))
	
	# Convert region to prolate
	region_mu, region_nu, region_phi = mathhelper.cart2prolate(region_contour[:, 0], region_contour[:, 1], region_contour[:, 2], focus)
	region_prol = np.column_stack((region_mu, region_nu, region_phi))
	region_prol_edges = region_prol[0::contour_edge_spacing, :]
	
	# Get polygonal path for the region edges
	region_prol_polygon = np.vstack((region_prol_edges[0::2, :], region_prol_edges[:0:-2, :], region_prol_edges[0, :]))
	region_polygon = path.Path(np.column_stack((region_prol_polygon[:, 2], region_prol_polygon[:, 1])))
	
	# Determine points inside region contour
	nodes_in_region = np.where(region_polygon.contains_points(np.column_stack((nodes_prol[:, 2], nodes_prol[:, 1]))))[0]
	
	# Get surface plots for inner and outer region surfaces
	base_list = list(range(contour_edge_spacing))
	sum_list = [[contour_edge_spacing*2*i]*contour_edge_spacing for i in range(num_slices)]
	region_inds_inner = []
	for i in range(len(sum_list)):
		region_inds_inner = np.append(region_inds_inner, np.add(base_list, sum_list[i]))
	region_inds_inner = [int(s_i) for s_i in region_inds_inner]
	region_inds_outer = [region_inds_inner_i + contour_edge_spacing for region_inds_inner_i in region_inds_inner]
	
	# Interpolate base on grid placement to get inner and outer values at each node
	inner_pt_vals = sp.interpolate.griddata(np.column_stack((region_prol[region_inds_inner, 2], region_prol[region_inds_inner, 1])), region_prol[region_inds_inner, 0], np.column_stack((nodes_prol[nodes_in_region, 2], nodes_prol[nodes_in_region, 1])), method='cubic')
	outer_pt_vals = sp.interpolate.griddata(np.column_stack((region_prol[region_inds_outer, 2], region_prol[region_inds_outer, 1])), region_prol[region_inds_outer, 0], np.column_stack((nodes_prol[nodes_in_region, 2], nodes_prol[nodes_in_region, 1])), method='cubic')
	mu_range = np.column_stack((inner_pt_vals, outer_pt_vals))
	
	# Error factor if needed
	mu_range_err = [err_val*abs(mu_range[i, 1] - mu_range[i, 0]) for i in range(mu_range.shape[0])]
	
	# Find which nodes are within the mu extent at the specified phi, nu points
	nodes_in_mu = np.where([((np.min(mu_range[i, :])-mu_range_err[i]) <= nodes_prol[nodes_in_region[i], 0]) & ((np.max(mu_range[i, :])+mu_range_err[i]) >= nodes_prol[nodes_in_region[i], 0]) for i in range(len(nodes_in_region))])[0]
	
	# Get final node values within the 3-d region
	final_node_inds = nodes_in_region[nodes_in_mu]
	
	return(final_node_inds)
	
def biCubicInterp(x_data, y_data, node_data, scale_der=0):
	"""Interpolate the x and y data to a bicubic fit
	
	args:
		x_data (array): x values to fit
		y_data (array): y values to fit
		node_data (array): Nodal values to use in the fit
		scale_der (int)
	
	returns:
		a (array): The interpolated values in order lambda, mu, theta, derivatives
	"""
	# Pull the mu and theta vectors from x, y, and node arrays
	new_mu_vec = x_data[0, :]
	new_theta_vec = y_data[:, 0]
	old_mu_vec = node_data[:, 0, 1]
	old_theta_vec = node_data[0, :, 2]
	
	# Pull the sizes of the arrays established earlier
	size_new_mu = new_mu_vec.size
	size_new_theta = new_theta_vec.size
	size_old_mu, size_old_theta, n = node_data.shape
	rads = math.pi/180
	
	# Set up A (output matrix of lambda, mu, theta, and derivatives, same order as node_data)
	a = np.zeros([size_new_mu, size_new_theta, n])
	a[:, :, 1] = np.transpose(x_data)
	a[:, :, 0] = np.transpose(y_data)
	
	for i in range(size_new_mu):
		for j in range(size_new_theta):
			# Pull the current mu and theta
			mu = new_mu_vec[i]
			theta = new_theta_vec[j]
			# Determine if you are currently at a nodal position
			found_nodal_point = False
			for k in range(size_old_mu):
				for m in range(size_old_theta):
					# If both mu and theta are in the old vectors, at a nodal point
					if (mu == old_mu_vec[k]) and (theta == old_theta_vec[m]):
						# Store the data in the a array if at a nodal point
						a[i, j, 2] = node_data[k, m, 0]
						a[i, j, 3:n] = node_data[k, m, 3:n]
						found_nodal_point = True
						break
			
			# If not at nodal position, do interpolation
			if not found_nodal_point:
				if i == 0:
					# Set min_t to the number of points in the old theta vector less than current theta
					min_t = np.where(old_theta_vec <= theta)[0].size
					# Set corner array based on nodal data
					corner = np.array([node_data[0, min_t, :], node_data[0, min_t-1, :], node_data[1, min_t, :], node_data[1, min_t, :]])
					# Calculate e array based on corner and theta values
					e = [(corner[0, 2]-theta)/(corner[0,2]-corner[1,2]), 0]
				elif j == (size_new_theta-1):
					# Set min_m to the number of point in the old mu vector less than current mu
					min_m = np.where(old_mu_vec < mu)[0].size
					# Set corner array based on nodal data
					corner = np.array([node_data[min_m-1, 1, :], node_data[min_m-1, 0, :], node_data[min_m, 1, :], node_data[min_m, 0, :]])
					# Calculate e array based on corner and mu values
					e = [1, (mu-corner[1,1])/(corner[3,1]-corner[1,1])]
				else:
					# Set min_t and min_m based on number of points in respective arrays less than current values
					min_t = np.where(old_theta_vec <= theta)[0].size
					min_m = np.where(old_mu_vec < mu)[0].size
					# Set corner array based on nodal data
					corner = np.array([node_data[min_m-1, min_t, :], node_data[min_m-1, min_t-1, :], node_data[min_m, min_t, :], node_data[min_m, min_t-1, :]])
					# Calculate e array based on corner, theta, and mu values
					e = [(corner[0, 2]-theta)/(corner[0, 2]-corner[1, 2]), (mu-corner[1, 1])/(corner[3, 1]-corner[1, 1])]
				
				if scale_der > 0:
					# Fill out the rest of the corner array
					corner[:, 3] *= (corner[0, 2]-corner[1, 2])*rads
					corner[:, 4] *= (corner[3, 1]-corner[1, 1])*rads
					corner[:, 5] *= (math.pi/180)*(corner[0, 2]-corner[1, 2])*(corner[3, 1]-corner[1, 1])*rads
				# Set a array points equal to lambda
				a[i, j, 2:n] = getLambda(corner, e)
				
				if scale_der > 0:
					# Modify the a array values based on corner values
					a[i, j, 3] /= (corner[0,2] - corner[1,2])*rads
					a[i, j, 4] /= (corner[3,1] - corner[1,1])*rads
					a[i, j, 5] /= (corner[0,2] - corner[1,2])*(corner[3,1]-corner[1,1])*(math.pi/180)*rads
	# Flip the a array first and third slices				
	temp = np.copy(a[:, :, 0])
	a[:, :, 0] = a[:, :, 2]
	a[:, :, 2] = temp
	return(a)
	
def generateDofData(dof, num_nodes, nodal_mu, nodal_theta, unsorted_nodal_mesh, size_nodal_mu):
	"""Return the unified dof_data array based on the passed inputs.
	
	This dof_data array is what is used to generate the IPNODE output file.
	"""
	dof_data = []
	for i in range(num_nodes):
		# Set m1 as the number of points in the unsorted nodal mesh greater than nodal mu.
		m1 = np.where(nodal_mu <= unsorted_nodal_mesh[i, 0])[0].size
		# Set t as the number of points in the unsorted nodal mesh greater than nodal theta.
		t = np.where(nodal_theta <= unsorted_nodal_mesh[i, 1])[0].size
		# Calculate indices and used those to pull from input dof and append to dof data.
		lam_ind = size_nodal_mu * (t-1) + m1 - 1
		d1 = lam_ind + num_nodes
		d2 = d1 + num_nodes
		d3 = d2 + num_nodes
		dof_data.append([dof[lam_ind], dof[d1], dof[d2], dof[d3]])
	return(np.array(dof_data))
	
def getStarterMesh(mesh_density):
	"""Return initial starter mesh based on mesh density."""
	# Based on the indicated mesh density, compose starter mesh columns
	if mesh_density == '4x2':
		first_coord = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
		second_coord = np.array([120, 120, 120, 120, 60, 60, 60, 60, 0, 0, 0, 0])
		third_coord = np.array([0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270])
		coord1_der1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		coord1_der2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		coord1_der3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	elif mesh_density == '4x4':
		first_coord = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
		second_coord = np.array([120, 120, 120, 120, 90, 90, 90, 90, 60, 60, 60, 60, 30, 30, 30, 30, 0, 0, 0, 0])
		third_coord = np.array([0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270])
		coord1_der1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		coord1_der2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		coord1_der3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	elif mesh_density == '4x8':
		first_coord = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
		second_coord = np.array([120, 120, 120, 120, 90, 90, 90, 90, 60, 60, 60, 60, 30, 30, 30, 30, 0, 0, 0, 0, 120, 120, 120, 120, 90, 90, 90, 90, 60, 60, 60, 60, 30, 30, 30, 30, 0, 0, 0, 0])
		third_coord = np.array([0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 45, 135, 225, 315, 45, 135, 225, 315, 45, 135, 225, 315, 45, 135, 225, 315, 45, 135, 225, 315])
		coord1_der1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		coord1_der2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		coord1_der3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	# Construct the full y array by stacking the columns
	y = np.column_stack((first_coord, second_coord, third_coord, coord1_der1, coord1_der2, coord1_der3))
	return(y)
	
def calcBasisDerivs(e, deriv_num, order=0):
	"""Compute derivatives of the bicubic hermite and lagrange basis function coefficients
	
	The order of the basis derivatives is the order of the time polynomial.
	
	args:
		e (array): The points at which to calculate the derivatives.
		deriv_num (int): Determines which derivatives to calculate
		order (int): Determines space-time derivative calculation
	returns:
		h (array): Array containing coefficient derivatives
	"""
	# 1D Shape Functions:
	h00 = [1 - 3*(e_i**2) + 2*(e_i**3) for e_i in e]
	h10 = [e_i*((e_i-1)**2) for e_i in e]
	h01 = [(e_i**2)*(3-2*e_i) for e_i in e]
	h11 = [(e_i**2)*(e_i-1) for e_i in e]
	
	# First Derivatives
	dh00 = [-6*e_i + 6*(e_i**2) for e_i in e]
	dh10 = [2*e_i*(e_i-1) + (e_i-1)**2 for e_i in e]
	dh01 = [-2*(e_i**2) + 2*e_i*(3-2*e_i) for e_i in e]
	dh11 = [(e_i**2) + 2*e_i*(e_i-1) for e_i in e]
	
	# Second Derivatives
	d2h00 = [12*e_i - 6 for e_i in e]
	d2h10 = [6*e_i - 4 for e_i in e]
	d2h01 = [-12*e_i + 6 for e_i in e]
	d2h11 = [6*e_i - 2 for e_i in e]
	
	# Assemble Spatial Derivative Coefficients
	if deriv_num == 0:
		h_init = [h00[0]*h00[1], h01[0]*h00[1], h00[0]*h01[1], h01[0]*h01[1], h10[0]*h00[1], h11[0]*h00[1], h10[0]*h01[1], h11[0]*h01[1], h00[0]*h10[1], h01[0]*h10[1], h00[0]*h11[1], h01[0]*h11[1], h10[0]*h10[1], h11[0]*h10[1], h10[0]*h11[1], h11[0]*h11[1]]
	elif deriv_num == 1:
		h_init = [dh00[0]*h00[1], dh01[0]*h00[1], dh00[0]*h01[1], dh01[0]*h01[1], dh10[0]*h00[1], dh11[0]*h00[1], dh10[0]*h01[1], dh11[0]*h01[1], dh00[0]*h10[1], dh01[0]*h10[1], dh00[0]*h11[1], dh01[0]*h11[1], dh10[0]*h10[1], dh11[0]*h10[1], dh10[0]*h11[1], dh11[0]*h11[1]]
	elif deriv_num == 2:
		h_init = [d2h00[0]*h00[1], d2h01[0]*h00[1], d2h00[0]*h01[1], d2h01[0]*h01[1], d2h10[0]*h00[1], d2h11[0]*h00[1], d2h10[0]*h01[1], d2h11[0]*h01[1], d2h00[0]*h10[1], d2h01[0]*h10[1], d2h00[0]*h11[1], d2h01[0]*h11[1], d2h10[0]*h10[1], d2h11[0]*h10[1], d2h10[0]*h11[1], d2h11[0]*h11[1]]
	elif deriv_num == 3:
		h_init = [h00[0]*dh00[1], h01[0]*dh00[1], h00[0]*dh01[1], h01[0]*dh01[1], h10[0]*dh00[1], h11[0]*dh00[1], h10[0]*dh01[1], h11[0]*dh01[1], h00[0]*dh10[1], h01[0]*dh10[1], h00[0]*dh11[1], h01[0]*dh11[1], h10[0]*dh10[1], h11[0]*dh10[1], h10[0]*dh11[1], h11[0]*dh11[1]]
	elif deriv_num == 4:
		h_init = [h00[0]*d2h00[1], h01[0]*d2h00[1], h00[0]*d2h01[1], h01[0]*d2h01[1], h10[0]*d2h00[1], h11[0]*d2h00[1], h10[0]*d2h01[1], h11[0]*d2h01[1], h00[0]*d2h10[1], h01[0]*d2h10[1], h00[0]*d2h11[1], h01[0]*d2h11[1], h10[0]*d2h10[1], h11[0]*d2h10[1], h10[0]*d2h11[1], h11[0]*d2h11[1]]
	elif deriv_num == 5:
		h_init = [dh00[0]*dh00[1], dh01[0]*dh00[1], dh00[0]*dh01[1], dh01[0]*dh01[1], dh10[0]*dh00[1], dh11[0]*dh00[1], dh10[0]*dh01[1], dh11[0]*dh01[1], dh00[0]*dh10[1], dh01[0]*dh10[1], dh00[0]*dh11[1], dh01[0]*dh11[1], dh10[0]*dh10[1], dh11[0]*dh10[1], dh10[0]*dh11[1], dh11[0]*dh11[1]]
		
	if order == 0:
		h = h_init
	else:
		# 0-order time derivative:
		
		# Normalized time coordinates for elemtn of this order
		t = [i/order for i in range(order+1)]
		num_t = order+1
		
		# Compute Lagrange Polynomial, 0-order time derivative
		lg = [1 for i in range(num_t)]
		lg = [lg[i]*(e[2]-t[j])/(t[i]-t[j]) for i in range(num_t) for j in range(num_t) if not(j==i)]
		
		# Rearrange to put Element nodes in first 2 pos
		if order > 1:
			temp = lg[1:num_t]
			lg[1] = lg[num_t]
			lg[2:num_t+1] = temp
			
		# Assemble First row of H (0-order time derivative)	
		h_temp = [lg[i]*h_init for i in range(num_t)]
		h[0, :] = h_temp
		
		# First order derivative:
		
		# Compute First-order time derivatives (velocity)
		vel = []
		for i in range(num_t):
			vel.append(0)
			for j in range(num_t):
				if j != i:
					vel_temp = 1
					for k in range(num_t):
						if k != j and k != i:
							vel_temp *= e[2] - t[k]
					vel[i] += vel_temp
			for m in range(num_t):
				if m != i:
					vel[i] /= t[i]-t[m]
		
		# Rearrange to put Elemtn nodes in first 2 pos
		if order > 1:
			temp = vel[1:num_t]
			vel[1] = vel[num_t]
			vel[2:num_t+1] = temp
			
		# Assemble Second row of H
		h_temp = [vel[i]*h_init for i in range(num_t)]
		h[1, :] = h_temp
		
		# Second order derivative:
		
		# Compute second-order time derivatives
		acc = []
		if order > 1:
			for i in range(num_t):
				acc.append(0)
				for j in range(num_t):
					if j != i:
						for k in range(num_t):
							if not k in [i, j]:
								acc_temp = 1
								for m in range(num_t):
									if not m in [k, j, i]:
										acc_temp *= e[2] - t[m]
								acc[i] += acc_temp
				for n in range(num_t):
					if n != i:
						acc[i] /= t[i]-t[n]
			temp = acc[1:num_t]
			acc[1] = acc[num_t]
			acc[2:num_t] = temp
			
			h_temp = [acc[i]*h_init for i in range(num_t)]
			
			h[2, :] = h_temp
	return(h)
	
def calcDamping(smooth_weights, num_gp, num_dof_total, num_dof_elem, size_nodal_mu, size_nodal_theta, num_nodes, num_dof_mesh, num_time_nodes=1):
	"""Calculate the global damping array from passed values
	
	Assumes that there are 5 derivatives (1, 11, 2, 22, 12)
	args:
		smooth_weights (array): Array output from reduced system solution
		num_gp (int)
		num_dof_total (int): Total number of degrees of freedom
		num_dof_elemn (int): Number of degrees of freedom per element
		size_nodal_mu (int): Size of the nodal_mu array
		size_nodal_theta (float): Size of the nodal_theta array
		num_nodes (int): Number of nodes
		num_dof_mesh (int): Number of degrees of freedom in the mesh
		num_time_nodes (int, default 1): Number of timepoints per node
	returns:
		global_damp (array): Array with global damping and mass matrix used for smoothing
	"""
	# Initialize the output matrix and convert size_nodal_theta to an integer
	size_nodal_theta = int(size_nodal_theta)
	global_damp = np.zeros([num_dof_total, num_dof_total])
	
	p = np.array([[0, 0, 0, 0, 0, 0, 0], [-0.2886751345948130, 0.2886751345948130, 0, 0, 0, 0, 0], [-0.3872983346207410, 0, 0.3872983346207410, 0, 0, 0, 0], [-0.4305681557970260, -0.1699905217924280, 0.1699905217924280, 0.4305681557970260, 0, 0, 0], [-0.4530899229693320, -0.2692346550528410, 0, 0.2692346550528410, 0.4530899229693320, 0, 0], [-0.4662347571015760, -0.3306046932331330, -0.1193095930415990, 0.1193095930415990, 0.3306046932331330, 0.4662347571015760, 0], [-0.4745539561713800, -0.3707655927996970, -0.2029225756886990, 0, 0.2029225756886990, 0.3707655927996970, 0.4745539561713800]]) + 0.5
	
	w = np.array([[1, 0, 0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0, 0, 0], [0.2777777777777780, 0.4444444444444440, 0.2777777777777780, 0, 0, 0, 0], [0.1739274225687270, 0.3260725774312730, 0.3260725774312730, 0.1739274225687270, 0, 0, 0], [0.1184634425280940, 0.2393143352496830, 0.2844444444444440, 0.2393143352496830, 0.1184634425280940, 0, 0], [0.0856622461895850, 0.1803807865240700, 0.2339569672863460, 0.2339569672863460, 0.1803807865240700, 0.0856622461895850, 0], [0.0647424830844350, 0.1398526957446390, 0.1909150252525600, 0.2089795918367350, 0.1909150252525600, 0.1398526957446390, 0.0647424830844350]])
	
	# Step through each element and integrate at Gauss points
	#	Same number of Gauss points are used for all elements
	for mu in range(size_nodal_mu-1):
		for theta in range(size_nodal_theta):
			# Calculate element number
			elem_num = (mu - 1)*size_nodal_theta + theta
			# Get the nearest theta and mu indices
			corner24theta = theta+1
			corner13theta = 1 if theta == size_nodal_theta-1 else corner24theta+1
			corner12mu = mu+1
			corner34mu = corner12mu + 1
			# Generate the index array of relevant degrees of freedom
			ind = generateInd(size_nodal_mu, corner13theta, corner24theta, corner12mu, corner34mu, num_nodes)
			# Step through each Gauss point
			for e1 in range(num_gp[0]):
				for e2 in range(num_gp[1]):
					# Calculate weight of gauss point and position
					wgp = w[num_gp[0]-1, e1]*w[num_gp[1]-1, e2]
					e = [p[num_gp[0]-1, e1], p[num_gp[1]-1, e2]]
					for der in range(5):
						# Get smoothing weighting factor for specific derivative
						wder = smooth_weights[elem_num, der]
						# Get the basis derivatives in the same order as ind
						h = calcBasisDerivs(e, der+1)
						for h1 in range(num_dof_elem):
							for h2 in range(num_dof_elem):
								for i in range(num_time_nodes):
									# Calculate the global damping matrix point
									global_damp[num_dof_mesh*(i-1) + ind[h1], num_dof_mesh*(i-1)+ind[h2]] += h[h1]*h[h2]*wgp*wder
									
	return(global_damp)
	
def getSmoothWeights(mesh_density, num_elem, unsorted_mesh_deg, nodal_mu_deg, nodal_theta_deg, size_nodal_theta):
	"""Parse the template files to get the smoothing weights
	
	args:
		ipfit_file (string): IPNODE template file
		elem_file (string): Starting mesh file
		num_elem (int): The number of elements
		unsorted_mesh_deg (array): The unsorted mesh, with angles in degrees
		nodal_mu_deg (array): Nodal mu values, based on angles in degrees
		nodal_theta_deg (array): Nodal theta values, with angles in degrees
		size_nodal_theta (array): Size of the nodal theta arrays
	returns:
		smooth_weights (array): The sorted mesh values
	"""
	# Pre-allocate the array for smooth_weights to allow indexing
	smooth_weights = np.zeros([int(num_elem), 5])
	# Generate columns for temp array
	if mesh_density == '4x2':
		col1 = np.array([0.01] * 8)
		col2 = np.array([0.02] * 8)
		col3 = np.array([0.01] * 8)
		col4 = np.array([0.02] * 8)
		col5 = np.array([0.04] * 8)
	elif mesh_density == '4x4':
		col1 = np.array([0.01] * 16)
		col2 = np.array([0.02] * 16)
		col3 = np.array([0.01] * 16)
		col4 = np.array([0.02] * 16)
		col5 = np.array([0.04] * 16)
	elif mesh_density == '4x8':
		col1 = np.array([0.1] * 32)
		col2 = np.array([0.2] * 32)
		col3 = np.array([0.1] * 32)
		col4 = np.array([0.2] * 32)
		col5 = np.array([0.4] * 32)
	# Form temp array from columns
	temp = np.column_stack((col1, col2, col3, col4, col5))
	# Generate Element Index Lists
	if mesh_density == '4x2':
		n1_list = [6, 7, 8, 5, 10, 11, 12, 9]
		n2_list = [5, 6, 7, 8, 9, 10, 11, 12]
	elif mesh_density == '4x4':
		n1_list = [6, 7, 8, 5, 10, 11, 12, 9, 14, 15, 16, 13, 18, 19, 20, 17]
		n2_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
	elif mesh_density == '4x8':
		n1_list = [25, 6, 26, 7, 27, 8, 28, 5, 29, 10, 30, 11, 31, 12, 32, 9, 33, 14, 34, 15, 35, 16, 36, 13, 37, 18, 38, 19, 39, 20, 40, 17]
		n2_list = [5, 25, 6, 26, 7, 27, 8, 28, 9, 29, 10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35, 16, 36, 17, 37, 18, 38, 19, 39, 20, 40]
	# Fill out smooth_weights array
	for i in range(len(n1_list)):
		# Pull n1 and n2 from the lists
		n1 = n1_list.pop(0)
		n2 = n2_list.pop(0)
		# Pull mu and theta based on n1 and n2
		mu = unsorted_mesh_deg[n1-1, 0]
		theta = unsorted_mesh_deg[n2-1, 1]
		# Get indices of nodal mu / theta array where less than mu / theta
		mu_ind = np.where(nodal_mu_deg <= mu)[0].size
		theta_ind = np.where(nodal_theta_deg <= theta)[0].size
		# Calculate e index based on mu and theta
		e_ind = int((mu_ind-1)*size_nodal_theta + theta_ind)
		# Copy last row in temp to smooth weights, then remove row from temp
		smooth_weights[e_ind-1] = temp[-1, :]
		temp = temp[:-1, :]
	return(smooth_weights)
	
def generalFit(dof_model, e, order=0):
	"""Compute lambda values by bicubic Hermite-Lagrange basis functions.
	
	args:
		dof_model (array): DOF of the model
		e (array): Local coordinates
		order (int): Temporal order
	returns:
		l (array): Lambda values
		h (array): Shape functions for local coordinates
	"""
	# 1-D Shape Functions
	h00 = [1 - 3*(e[0]**2) + 2*(e[0]**3), 1 - 3*(e[1]**2) + 2*(e[1]**3)]
	h10 = [e[0]*((e[0]-1)**2), e[1]*((e[1]-1)**2)]
	h01 = [(e[0]**2)*(3-2*e[0]), (e[1]**2)*(3-2*e[1])]
	h11 = [(e[0]**2)*(e[0]-1), (e[1]**2)*(e[1]-1)]
	
	# Assemble 2-D Shape Functions
	h_init = [h00[0]*h00[1], h01[0]*h00[1], h00[0]*h01[1], h01[0]*h01[1], h10[0]*h00[1], h11[0]*h00[1], h10[0]*h01[1], h11[0]*h01[1], h00[0]*h10[1], h01[0]*h10[1], h00[0]*h11[1], h01[0]*h11[1], h10[0]*h10[1], h11[0]*h10[1], h10[0]*h11[1], h11[0]*h11[1]]
	
	if order > 0: # Temporal Basis Functions
		# Normalize time coordinates
		t = [i/order for i in range(order+1)]
		num_t = order+1
		# Compute Lagrange polynomial
		lg = [1 for i in range(num_t)]
		lg = [lg[i]*(e[2]-t[j])/(t[i]-t[j]) for i in range(num_t) for j in range(num_t) if not(j==i)]
		# Rearrange to put element nodes in first 2 positions
		if order > 1:
			temp = lg[1:num_t]
			lg[1] = lg[order]
			lg[2:order] = temp
		# Assemble h and get l
		l = 0
		h = []
		for i in range(num_t):
			l = l + lg[i]*np.dot(h_init, dof_model[:, i])
			h.append([lg[i]*h_init_i for h_init_i in h_init])
	else: # Space-only fit
		h = h_init
		l = np.dot(h, dof_model)
	return([l, h])
	
def fitBicubicData(data, focus, mesh_density='4x2', smooth=True, constraints=True, compute_errors=True):
	"""Bicubic fit of x,y,z data to a prolate mesh
	
	args:
		data (array): x, y, z points arranged by column
		focus (float): the focal point for the prolate spheroid
		mesh_density (string): Determines mesh type
		smooth (bool): Whether or not to implement smoothing
		contraints (bool): Determines solution method
	returns:
		node_matrix (array): Array containing nodal prolate spheroid values
		rms_error (int): Error estimation for lambda
	"""
	
	if not mesh_density in ['4x2', '4x4', '4x8']:
		mesh_density = '4x2'
		print('Mesh density incorrect. Set to 4x2.')
	
	if constraints:
		if mesh_density == '4x2':
			c = np.array([[1, -1, 1, 0, 1, -1, 1, 0, 1], [4, 1, 1, 0, 1, -1, 1, 0, 1], [7, 1, 1, 0, 1, 1, -1, 0, 1], [10, 1, 1, 0, 1, 4, -1, 0, 1]])
		elif mesh_density == '4x4':
			c = np.array([[1, -1, 1, 0, 1, -1, 1, 0, 1], [6, 1, 1, 0, 1, -1, 1, 0, 1], [11, 1, 1, 0, 1, 1, -1, 0, 1], [16, 1, 1, 0, 1, 6, -1, 0, 1]])
		elif mesh_density == '4x8':
			c = np.array([[1, -1, 1, 0, 1, -1, 1, 0, 1], [6, 1, 1, 0, 1, -1, 1, 0, 1], [11, 1, 1, 0, 1, -1, 1, 0, 1], [16, 1, 1, 0, 1, -1, 1, 0, 1], [21, 1, 1, 0, 1, 1, -1, 0, 1], [26, 1, 1, 0, 1, 6, -1, 0, 1], [31, 1, 1, 0, 1, 11, -1, 0, 1], [36, 1, 1, 0, 1, 16, -1, 0, 1]])
	
	# Get starting mesh and organize
	nodal_mesh = getStarterMesh(mesh_density)
	unsorted_nodal_mesh_deg = nodal_mesh[:, 1:3]
	nodal_mesh = nodal_mesh[nodal_mesh[:, 0].argsort()]
	nodal_mesh = nodal_mesh[nodal_mesh[:, 1].argsort(kind='mergesort')]
	nodal_mesh = nodal_mesh[nodal_mesh[:, 2].argsort(kind='mergesort')]
	
	# Get Sizes
	m = nodal_mesh.shape[0]
	size_nodal_nu = np.where(nodal_mesh[:, 2] == 0)[0].size
	size_nodal_phi = m/size_nodal_nu
	num_elem = size_nodal_phi*(size_nodal_nu-1)
	
	# Put into radians
	rads = math.pi/180
	nodal_nu_deg = nodal_mesh[0:size_nodal_nu, 1]
	nodal_phi_deg = nodal_mesh[0::size_nodal_nu, 2]
	nodal_nu = nodal_nu_deg*rads
	nodal_phi = nodal_phi_deg*rads
	
	# Build initial guess vector
	init_guess = nodal_mesh[:, 0].tolist()
	init_guess.extend(nodal_mesh[:, 3])
	init_guess.extend(nodal_mesh[:, 4])
	init_guess.extend(nodal_mesh[:, 5])
	num_dof_total = len(init_guess)
	num_dof_elem = 16

	if smooth:
		gp_limit = 7
		num_gp = [3, 3]
		if num_gp[0] > gp_limit or num_gp[1] > gp_limit:
			raise(NameError)
	
	# Initialize
	pts_elem = [0] * int(num_elem)
	global_stiff = [[0 for i in range(int(num_dof_total))] for j in range(int(num_dof_total))]
	global_rhs = [0] * int(num_dof_total)
	lhs_matrix = [[0] * int(num_dof_total)] * int(num_dof_total)
	
	# Read Data Points:
	data_x = data[:, 0]
	data_y = data[:, 1]
	data_z = data[:, 2]
	count = data.shape[0]
	data_w = np.ones((count, 3))
	# Convert to Prolate
	data_size = 0
	min_nodal_nu = min(nodal_nu)
	max_nodal_nu = max(nodal_nu)
	max_nodal_phi = max(nodal_phi)
	for i in range(count):
		ind = []
		mu, nu, phi = mathhelper.cart2prolate(data_x[i], data_y[i], data_z[i], focus)
		if nu >= min_nodal_nu and nu <= max_nodal_nu:
			data_size += 1
			e, corner13phi, corner24phi, corner12nu, corner34nu = nearestNodalPoints(nu, phi, nodal_phi, size_nodal_phi, max_nodal_phi, nodal_nu, min_nodal_nu, max_nodal_nu)
			element_number = int((corner12nu-1)*size_nodal_phi + corner24phi)
			pts_elem[element_number - 1] = pts_elem[element_number - 1] + 1
			# Build dof vector.
			# 	ind is the index vector, of length 16.
			ind = generateInd(size_nodal_nu, corner13phi, corner24phi, corner12nu, corner34nu, m)

			dof_model = np.array(init_guess)[ind]
			
			lam_model, h = generalFit(dof_model, e)
			
			lam_diff = mu - lam_model

			for h1 in range(num_dof_elem):
				global_rhs[ind[h1]] = global_rhs[ind[h1]] + h[h1]*lam_diff*data_w.flatten()[i]
				for h2 in range(num_dof_elem):
					global_stiff[ind[h1]][ind[h2]] += h[h1]*h[h2]*data_w.flatten()[i]
	
	# Add smoothing if requested
	if smooth:
		smooth_weights = getSmoothWeights(mesh_density, num_elem, unsorted_nodal_mesh_deg, nodal_nu_deg, nodal_phi_deg, size_nodal_phi)
		global_damp = calcDamping(smooth_weights, num_gp, num_dof_total, num_dof_elem, size_nodal_nu, size_nodal_phi, m, num_dof_total)
		lhs_matrix = global_stiff + global_damp
	else:
		lhs_matrix = global_stiff
	
	# Solve for displacement
	if constraints:
		displacement = solveReducedSystem(lhs_matrix, global_rhs, c, num_dof_total, m, num_dof_total)
	else:
		displacement = np.linalg.solve(lhs_matrix, global_rhs)
	
	# Fitted dof values sorted by mu and theta
	optimized_dof = np.add(init_guess, displacement)
	
	# Organize DOF data as though it is written to and read from IPNODE file
	dof_data = generateDofData(optimized_dof, m, nodal_nu_deg, nodal_phi_deg, unsorted_nodal_mesh_deg, size_nodal_nu)
	
	# Get the 72x6 matrix with nodal values
	# 	Columns are in the order: lambda, mu, theta, dlds1, dlds2, d2lds1ds2
	node_matrix = np.append(dof_data[:, 0].reshape([dof_data.shape[0], 1]), np.append(unsorted_nodal_mesh_deg, dof_data[:, 1:], axis=1), axis=1)
	
	# Perform the error estaimate
	if compute_errors:
		# Initialize global error
		err = 0
		
		# Iterate through points again to estimate error
		for i in range(count):
			mu, nu, phi = mathhelper.cart2prolate(data_x[i], data_y[i], data_z[i], focus)
			if nu >= min_nodal_nu and nu < max_nodal_nu:
				e, corner13phi, corner24phi, corner12nu, corner34nu = nearestNodalPoints(nu, phi, nodal_phi, size_nodal_phi, max_nodal_phi, nodal_nu, min_nodal_nu, max_nodal_nu)
				ind = generateInd(size_nodal_nu, corner13phi, corner24phi, corner12nu, corner34nu, m)
				dof_model = optimized_dof[ind]
				
				lam_model, _ = generalFit(dof_model, e)
				
				err += ((lam_model - mu)*data_w.flatten()[i])**2
		# Get the RMS Error in Lambda
		rms_err = math.sqrt(err / data_size)
	else:
		rms_err = 0
	return([node_matrix, rms_err])
	
def solveReducedSystem(lhs_matrix, global_rhs, c, num_dof_total, nn, num_dof_mesh, num_time_nodes=1):
	"""Use the constraints encoded in c to reduce linear system of equations and solve
	
	args:
		lhs_matrix (array): The left-hand side of the equation
		global_rhs (array): The right-hand side of the equation
		c (array): Array of constraints on the equation
		num_dof_total (int): Total number of degrees of freedom
		nn
		num_dof_mesh (int): Number of dofs in the mesh
		num_time_nodes (int): How many timepoints to use for each node
	returns:
		displacement (array): Solution to the equation between lhs and rhs
	"""
	# Set up the mapping vector v
	v = np.zeros([num_dof_total, 2])
	v[:, 0] = range(num_dof_total)
	v += 1
	
	# Arrange degrees of freedom
	m = c.shape[0]
	k = num_dof_total
	
	for i in range(m):
		for j in range(1,8,2):
			if c[i, j] > 0: # Coupling
				k -= num_time_nodes
				mult = c[i, j+1]
				search_flag = True
				target = c[i, j]
				# Cycle until you reach the end of a link
				while search_flag:
					p = np.where(target == c[:, 0])[0]
					if c[p, j] == -1:
						for tnn in range(num_time_nodes):
							v[int(num_dof_mesh*tnn+c[i, 0]+(((j+1)/2)-1)*nn)-1, 0] = num_dof_mesh*tnn+c[p,0]+(((j+1)/2)-1)*nn
							v[int(num_dof_mesh*tnn+c[i, 0]+(((j+1)/2)-1)*nn)-1, 1] = mult
						search_flag = False
					else:
						target = c[p, j]
						mult = mult*c[p, j+1]
			elif c[i, j] == 0: # Fixing
				k -= num_time_nodes
				for tnn in range(num_time_nodes):
					v[int(num_dof_mesh*tnn+c[i, 0]+(((j+1)/2)-1)*nn)-1, 0] = 0
	# Eliminate zeros and redundancies
	true_dof = np.unique(v[np.nonzero(v[:, 0]), 0][0])
	
	# Reduction and Solve:
	red_rhs = np.zeros(k)
	red_lhs = np.zeros([k, k])
	ind1 = 0
	
	for i in range(num_dof_total):
		p1 = v[i, 0]
		mult1 = v[i, 1]
		
		if p1 != 0: # Ensure p1 is unfixed
			if p1 == i+1: # Ensure p1 is a free degree of freedom
				red_rhs[ind1] += global_rhs[i]
				ind2 = 0
				for j in range(num_dof_total):
					p2 = v[j, 0]
					mult2 = v[j, 1]
					if p2 != 0: # Ensure p2 is unfixed
						if p2 == j+1: # Ensure p2 is a free degree of freedom
							red_lhs[ind1, ind2] += lhs_matrix[i, j]
							ind2 += 1
						else: # If p2 is coupled:
							m = np.where(p2 == true_dof)[0]
							red_lhs[ind1, m] += lhs_matrix[i, j]*mult2
				ind1 += 1
			else: # If p1 is coupled:
				m = np.where(p1 == true_dof)[0]
				red_rhs[m] += global_rhs[i]*mult1
				ind2 = 0
				for j in range(num_dof_total):
					p2 = v[j, 0]
					mult2 = v[j, 1]
					if p2 != 0: # Ensure p2 is unfixed
						if p2 == j+1: # Ensure p2 is a free degree of freedom
							red_lhs[m, ind2] += lhs_matrix[i, j]*mult1
							ind2 += 1
						else: # If p2 is coupled:
							n = np.where(p2 == true_dof)[0]
							red_lhs[m, n] += lhs_matrix[i, j]*mult1*mult2
	# Solving the reduced system
	red_disp = np.linalg.solve(red_lhs, red_rhs)
	
	# Reassemble
	displacement = []
	for i in range(num_dof_total):
		p = v[i, 0]
		mult = v[i, 1]
		if p != 0: # P is unfixed
			ind = np.where(p == true_dof)[0]
			displacement.append((red_disp[ind]*mult)[0])
		else: # P is fixed
			displacement.append(0)

	return(displacement)