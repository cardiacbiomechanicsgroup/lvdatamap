import numpy as np
import math
import matplotlib.pyplot as mplt
from mpl_toolkits.mplot3d import Axes3D
from cardiachelpers import meshhelper
from cardiachelpers import mathhelper
from cardiachelpers import stackhelper
import subprocess

def segmentRender(all_data_endo, all_data_epi, apex_pt, basal_pt, septal_pts, origin, transform, landmarks=True, ax=None, scar=None):
	"""Display the segmentation contours and user-indicated points
	
	args:
		all_data_endo (array): All endocardial contour data including slice values.
		all_data_epi (array): All epicardial contour data including slice values.
		apex_pt (array): Apical point
		basal_pt (array): Basal point
		septal_pts (array): Septal points (including midpoint)
		landmarks (bool): Whether to plot the pinpoints.
		ax (mplotlib3d axes): Axes on which to plot (default None)
	returns:
		ax (mplotlib3d axes): The axes of the contour
	"""
	if not ax:
		fig = mplt.figure()
		ax = fig.add_subplot(111, projection='3d')
	all_data_endo = np.array(all_data_endo).squeeze()
	all_data_epi = np.array(all_data_epi).squeeze()
	data_endo = stackhelper.rotateDataCoordinates(all_data_endo[:, :3], apex_pt, basal_pt, septal_pts)[0]
	data_epi = stackhelper.rotateDataCoordinates(all_data_epi[:, :3], apex_pt, basal_pt, septal_pts)[0]
	# Subtract origin and transform data
	apex_transform = np.dot((apex_pt - origin), np.transpose(transform))
	basal_transform = np.dot((basal_pt - origin), np.transpose(transform))
	septal_transform1 = np.dot((septal_pts[0, :] - origin), np.transpose(transform))
	septal_transform2 = np.dot((septal_pts[1, :] - origin), np.transpose(transform))
	septal_transform3 = np.dot((septal_pts[2, :] - origin), np.transpose(transform))
	# Set up bins as the unique data in all_data_endo third column (the slices)
	bins = np.unique(all_data_endo[:, 3])
	for jz in range(bins.size):
		# Get the indices that match the current bin and append then append the first value
		endo_tracing = np.where(all_data_endo[:, 3] == bins[jz])[0]
		endo_tracing = np.append(endo_tracing, endo_tracing[0])
		#print(tracing)
		# Pull x, y, z from endo and epi and plot
		x = data_endo[endo_tracing, 2]
		y = data_endo[endo_tracing, 1]
		z = data_endo[endo_tracing, 0]
		ax.plot(x, y, -z, 'y-')
		# Epi plotting
		epi_tracing = np.where(all_data_epi[:, 3] == bins[jz])[0]
		epi_tracing = np.append(epi_tracing, epi_tracing[0])
		x = data_epi[epi_tracing, 2]
		y = data_epi[epi_tracing, 1]
		z = data_epi[epi_tracing, 0]
		ax.plot(x, y, -z, 'c-')
	if landmarks:
		# Plot the apex, basal, and septal points
		ab = np.array([apex_transform, basal_transform])
		si = np.array([septal_transform2, septal_transform3])
		
		ax.plot(ab[:, 2], ab[:, 1], -ab[:, 0], 'k-.')
		ax.scatter(si[:, 2], si[:, 1], -si[:, 0], 'bo', s=50)
		ax.scatter(septal_transform1[2], septal_transform1[1], -septal_transform1[0], 'ro', s=50)
	
	return(ax)
	
def surfaceRender(nodal_mesh, focus, ax=None):
	"""Plot surface mesh on optionally-passed axes
	
	args:
		nodal_mesh (array): Mesh to be plotted
		ax (mplot3d axes object): Axes on which to plot the mesh
	returns:
		ax (mplot3d axes object): Axes containing the surface plot contained in a figure
	"""
	# If no axes were passed, generate new set of axes
	if not ax:
		fig = mplt.figure()
		ax = fig.add_subplot(111, projection='3d')

	# Sort the mesh by first 3 columns
	nodal_mesh = nodal_mesh[nodal_mesh[:, 0].argsort()]
	nodal_mesh = nodal_mesh[nodal_mesh[:, 1].argsort(kind='mergesort')]
	nodal_mesh = nodal_mesh[nodal_mesh[:, 2].argsort(kind='mergesort')]
	
	# Set up number of divisions and calculate e for each division (as a ratio)
	num_div = 20
	e = [i/num_div for i in range(num_div + 1)]
	# Convert angular values from degrees to radians
	rads = math.pi/180
	nodal_mesh[:, 1:3] *= rads
	# Store the shapes and sizes of the mesh values
	m = nodal_mesh.shape[0]
	size_nodal_nu = np.where(nodal_mesh[:, 2] == 0)[0].size
	size_nodal_phi = m/size_nodal_nu
	# Get the mu and theta values from the mesh
	nodal_nu = nodal_mesh[:size_nodal_nu, 1]
	nodal_phi = nodal_mesh[::size_nodal_nu, 2]
	# Convert apex node from prolate to cartesian, then plot with scatter
	if min(nodal_nu) == 0:
		x, y, z = mathhelper.prolate2cart(nodal_mesh[0, 0], nodal_mesh[0, 1], nodal_mesh[0, 2], focus)
		ax.scatter(z, y, -x)
		start_nu = 1
	else:
		start_nu = 0
	# Plot circumferential element boundaries
	for i in range(start_nu, size_nodal_nu):
		for j in range(int(size_nodal_phi)):
			# Define nodal values for interpolation
			if j == size_nodal_phi-1:
				ind0 = i
				p0 = 2*math.pi
			else:
				ind0 = (j+1)*size_nodal_nu + i
				p0 = nodal_phi[j+1]
			ind1 = (j)*size_nodal_nu + i
			p1 = nodal_phi[j]
			# Get mu and dM/dm1
			m0 = nodal_mesh[ind0, 0]
			dm0 = nodal_mesh[ind0, 3]
			m1 = nodal_mesh[ind1, 0]
			dm1 = nodal_mesh[ind1, 3]
			# Convert to cartesian
			n0x, n0y, n0z = mathhelper.prolate2cart(nodal_mesh[ind0, 0], nodal_mesh[ind0, 1], nodal_mesh[ind0, 2], focus)
			# Plot the node
			ax.scatter(n0z, n0y, -n0x)
			# Plot the arc segments
			for k in range(2, len(e)):
				# Determine starting point to use
				if k == 2:
					pt_x, pt_y, pt_z = n0x, n0y, n0z
				else:
					pt_x, pt_y, pt_z = x_here, y_here, z_here
				# Get lambda
				hm0 = 1 - 3*(e[k]**2) + 2*(e[k]**3)
				hdm0 = e[k]*(e[k] - 1)**2
				hm1 = (e[k]**2)*(3 - 2*e[k])
				hdm1 = (e[k]**2)*(e[k] - 1)
				m = hm0 * m0 + hdm0 * dm0 + hm1 * m1 + hdm1 * dm1
				# Get theta
				p_here = p0 - e[k]*(p0 - p1)
				# Convert to cartesian
				x_here, y_here, z_here = mathhelper.prolate2cart(m, nodal_nu[i], p_here, focus)
				# Create vectors
				x = np.append(pt_x, x_here)
				y = np.append(pt_y, y_here)
				z = np.append(pt_z, z_here)
				# Plot segments
				ax.plot(z, y, -x, 'k-.')
	# Plot longitudinal element boundaries
	for i in range(int(size_nodal_phi)):
		for j in range(size_nodal_nu-1):
			# Define nodal values needeed for interpolation
			ind0 = i*size_nodal_nu + j
			ind1 = ind0 + 1
			n0 = nodal_nu[j]
			n1 = nodal_nu[j+1]
			# Get lambda and dL/de2
			m0 = nodal_mesh[ind0, 0]
			dm0 = nodal_mesh[ind0, 4]
			m1 = nodal_mesh[ind1, 0]
			dm1 = nodal_mesh[ind1, 4]
			# Convert nodal points to cartesian
			n0x, n0y, n0z = mathhelper.prolate2cart(nodal_mesh[ind0, 0], nodal_mesh[ind0, 1], nodal_mesh[ind0, 2], focus)
			# Plot arc
			for k in range(2, len(e)):
				# Determine point to use
				if k == 2:
					pt_x, pt_y, pt_z = n0x, n0y, n0z
				else:
					pt_x, pt_y, pt_z = x_here, y_here, z_here
				# Get lambda
				hm0 = 1 - 3*(e[k]**2) + 2*(e[k]**3)
				hdm0 = e[k]*(e[k] - 1)**2
				hm1 = (e[k]**2)*(3 - 2*e[k])
				hdm1 = (e[k]**2)*(e[k] - 1)
				m = hm0 * m0 + hdm0 * dm0 + hm1 * m1 + hdm1 * dm1
				# Get nu
				n_here = n0 + e[k]*(n1-n0)
				# Convert to cartesian
				x_here, y_here, z_here = mathhelper.prolate2cart(m, n_here, nodal_phi[i], focus)
				# Append the vectors for plotting
				x = np.append(pt_x, x_here)
				y = np.append(pt_y, y_here)
				z = np.append(pt_z, z_here)
				# Plot the segment
				ax.plot(z, y, -x, 'k-.')
				
	return(ax)
	
def displayScarTrace(scar, origin, transform, ax=None):
	"""Plots scar trace overlay onto a passed axis.
	"""
	if not ax:
		fig = mplt.figure()
		ax = fig.add_subplot(111, projection='3d')
	for scar_slice in scar:
		# Append the first point to the end to make a circular contour
		cur_scar = np.append(scar_slice, np.expand_dims(scar_slice[0, :], 0), axis=0)
		# Transform the data using the same transformation as endo/epi contours
		data_scar = np.dot((cur_scar - np.array([origin for i in range(cur_scar.shape[0])])), np.transpose(transform))
		# Plot
		x = data_scar[:, 2]
		y = data_scar[:, 1]
		z = data_scar[:, 0]
		ax.plot(x, y, -z, 'r-')
	return(ax)

def displayDensePts(dense_pts, dense_slices, origin, transform, dense_displacement_all=False, dense_plot_quiver=0, timepoint=-1, ax=None):
	"""Shows DENSE pointts and (optionally) displacements in a 3D graph.
	"""
	# If no axes were passed, generate axes
	if not ax:
		fig = mplt.figure()
		ax = fig.add_subplot(111, projection='3d')
	
	# Pull appropriate timepoint (or set to False if displacement is undesired)
	if timepoint >= 0:
		try:
			dense_displacement = dense_displacement_all[timepoint]
		except(IndexError):
			warnings.warn('Timepoint not in range of values! Displacement will be ignored.')
			timepoint = -1
			dense_displacement = False
		except(TypeError):
			warnings.warn('Displacement not passed, but timepoint requested! Displacement will be ignored.')
			dense_displacement = False
	else:
		dense_displacement = False
	
	for i in range(len(dense_slices)):
		cur_slice = dense_slices[i]
		cur_dense_pts = np.column_stack((dense_pts[i], [cur_slice - origin[2]]*dense_pts[i].shape[0]))
		data_dense = np.dot(cur_dense_pts, np.transpose(transform))
		x = data_dense[:, 2]
		y = data_dense[:, 1]
		z = data_dense[:, 0]
		ax.scatter(x, y, -z, ',')
		
		if dense_displacement:
			dense_displacement_slice = dense_displacement[i]
			if dense_plot_quiver == 1:
				ax.quiver(x, y, -z, dense_displacement_slice[:, 0], dense_displacement_slice[:, 1], [0]*dense_displacement_slice.shape[0])
	return(ax)
	
def nodeRender(nodes, ax=None):
	"""Display the nodes passed in as a 3D scatter plot."""
	if not ax:
		fig = mplt.figure()
		ax = fig.add_subplot(111, projection='3d')
	x = nodes[:, 2]
	y = nodes[:, 1]
	z = nodes[:, 0]
	ax.scatter(x, y, -z)
	return(ax)
	
def displayMeshPostview(file_name, executable_name):
	"""Launch PostView with specific file selected.
	
	The string for the subprocess must point to your installation of PostView.
	"""
	p = subprocess.Popen([executable_name, file_name])
	return(p)
	
def plotListData(input_list, title_list, ax=None):
	if not ax:
		fig = mplt.figure()
		ax = fig.add_subplot(111)
	for input_vals in input_list:
		x_vals = list(range(len(input_vals)))
		ax.plot(x_vals, input_vals)
	ax.legend(title_list)
	return(ax)