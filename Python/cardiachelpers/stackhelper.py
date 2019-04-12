import numpy as np
import scipy as sp
import scipy.stats as spstats
import math
import cardiachelpers.mathhelper as mathhelper

def rotateDataCoordinates(points, apex_pt, basal_pt, septal_pts):
	"""Reorganize endo and epi data for processing.
	
	args:
		all_data_endo (array): Endo data from MRI model
		all_data_epi (array): Epi data from MRI model
		apex_pt (array): Apical point selected from long-axis data
		basal_pt (array): Basal point selected from long-axis data
		septal_pts (array): Septal points selected from short-axis data
	returns:
		data_endo (array): Modified endo contour
		data_epi (array): Modified epi contour
		focus (float): The focal point for prolate coordinates
		transform_basis (array): Combined vector from the 3 calculated basis vectors
		origin (array): Point indicating the origin point
	"""
	#endo = all_data_endo[:, 0:3]
	#epi = all_data_epi[:, 0:3]
	points = np.array(points)
	apex_pt = np.array(apex_pt)
	basal_pt = np.array(basal_pt)
	septal_pts = np.array(septal_pts)
	
	calcNorm = lambda arr_in: np.sqrt(np.sum(np.square(arr_in)))
	
	# Calculate first basis vector, (Base - Apex)/Magnitude
	c = apex_pt - basal_pt
	c_norm = calcNorm(c)
	e1_basis = [c1 / c_norm for c1 in c]
	
	# Calculate Origin Location
	origin = basal_pt + c/3
	
	# Calculate Focus Length based on c_norm
	focus = (2*c_norm/3)/(math.cosh(1))
	num_points = points.shape[0]
	#epi_points = epi.shape[0]
	
	# Calculate Second basis vector using plane intersects, septal point, and e1
	d1 = septal_pts[0, :] - origin
	d2 = d1 - [np.dot(d1, e1_basis)*e1_elem for e1_elem in e1_basis]
	e2_basis = d2 / calcNorm(d2)
	
	# Calculate third basis vector from the first 2 basis vectors
	e3 = np.cross(e1_basis, e2_basis)
	e3_basis = e3 / calcNorm(e3)
	
	# Set up transform basis from the 3 calculated basis vectors
	transform_basis = np.array([e1_basis, e2_basis, e3_basis])
	
	# Set up the modified endo and epi contours
	#print(points)
	rot_points = np.dot((points - np.array([origin for i in range(num_points)])), np.transpose(transform_basis))
	#data_epi = np.dot((epi - np.array([origin for i in range(epi_points)])), np.transpose(transform_basis))
	
	# Append extra identifying data to the modified contours
	#data_endo = np.append(data_endo, np.reshape(all_data_endo[:, 3], [all_data_endo.shape[0], 1]), axis=1)
	#data_epi = np.append(data_epi, np.reshape(all_data_epi[:, 3], [all_data_epi.shape[0], 1]), axis=1)
	return([rot_points, focus, transform_basis, origin])

def getContourFromStack(endo_stack, epi_stack, sastruct, rv_insertion_pts, septal_slice, apex_base_pts, scar_stack=np.empty([0])):
	"""Using the stacks, construct the endo and epi contours in proper format, abs points, axis center, and (if passed) scar
	
	Uses several matrix transformations to convert the stack data into the finalized contours.
	args:
		endo_stack (array): Original endocardial stack
		epi_stack (array): Original epicardial stack
		sastruct (dict): Setstruct from the SEGMENT MAT file
		rv_insertion_pts (array): RV points indicated in the SEGMENT file
		septal_slice (integer): The slice containing the rv points
		apex_base_pts (array): The points from the long-axis image indicating apex and base
		scar_stack (array): Original scar point stack
	returns:
		abs_shifted (array): Shifted apex, basal, and septal points
		endo (array): Modified endocardial contours
		epi (array): Modified epicardial contours
		axis_center (array): Center of slice by apex-base axis calculation
		scar_all (array): Shifted scar points to align with new endo and epi contours
	"""
	# Pull elements from passed args
	kept_slices = sastruct['KeptSlices']
	apex_pts = apex_base_pts[-2].reshape([1, 3])
	base_pts = apex_base_pts[-1].reshape([1, 3])
	septal_pts = rv_insertion_pts[(2, 0, 1), :]
	# Calculate the z-orientation and store it in the m array
	cine_z_orientation = np.cross(sastruct['ImageOrientation'][0:3], sastruct['ImageOrientation'][3:6])
	cine_m = np.array([sastruct['ImageOrientation'][3:6], sastruct['ImageOrientation'][0:3], cine_z_orientation])
	# Multiply the xyz parts of the stack by the m array and store
	transform_endo = np.transpose(cine_m@np.transpose(endo_stack[:, 0:3]))
	transform_epi = np.transpose(cine_m@np.transpose(epi_stack[:, 0:3]))
	transform_abs = np.transpose(cine_m@np.transpose(np.append(apex_pts, np.append(base_pts, septal_pts, axis=0), axis=0)))
	if scar_stack.size:
		scar_z_orientation = np.cross(sastruct['ImageOrientation'][0:3], sastruct['ImageOrientation'][3:6])
		scar_m = np.array([sastruct['ImageOrientation'][3:6], sastruct['ImageOrientation'][0:3], scar_z_orientation])
		transform_scar = np.transpose(scar_m@np.transpose(scar_stack[:, 0:3]))
	# Calculate the apex-base elements from the transformed abs points
	ab_dist = transform_abs[1, :] - transform_abs[0, :]
	ab_x = [transform_abs[0, 0] + (ab_dist[0]*(item/100)) for item in list(range(101))]
	ab_y = [transform_abs[0, 1] + (ab_dist[1]*(item/100)) for item in list(range(101))]
	ab_z = [transform_abs[0, 2] + (ab_dist[2]*(item/100)) for item in list(range(101))]
	# Generate a list of z values (by slice)
	z_loc = [transform_endo[np.where(endo_stack[:, 4] == cur_slice)[0][0], 2] for cur_slice in kept_slices]
	# Calculate the m-array for the each slice and store in a list
	m_slices = [(z_loc_cur - transform_abs[0,2])/ab_dist[2] for z_loc_cur in z_loc]
	# Calculate the apex-base axis based on the slice m values and ab_dist
	ba_axis_x = [m_slice * ab_dist[0] + transform_abs[0, 0] for m_slice in m_slices]
	ba_axis_y = [m_slice * ab_dist[1] + transform_abs[0, 1] for m_slice in m_slices]
	ba_axis_intercept = np.transpose(np.array([ba_axis_x, ba_axis_y, z_loc]))
	#Set up lists before appending values
	slice_center = []
	center_axis_diff = []
	endo_shifted = [None] * transform_endo.shape[0]
	epi_shifted = [None] * transform_epi.shape[0]
	axis_center = []
	center_axis_diff = []
	if scar_stack.size:
		scar_shifted = [None] * transform_scar.shape[0]
	# Iterate through and calculate the new endo and epi values
	for i in range(len(kept_slices)):
		# Pull the current slice and find which values in the stack are in the correct slice
		cur_slice = kept_slices[i]
		slice_endo_inds = np.where(endo_stack[:, 4] == cur_slice)[0]
		slice_epi_inds = np.where(epi_stack[:, 4] == cur_slice)[0]
		# Get the center of the slice for both epicardial center (slice) and apex-base line (axis)
		slice_center.append(np.mean(transform_epi[slice_epi_inds, :], axis=0))
		axis_center.append(ba_axis_intercept[i, :])
		# Calculate the difference between slice center and axis center
		center_axis_diff.append(slice_center[i] - axis_center[i])
		# Shift the slices by the difference in centers and shift, to align the centers
		for j in slice_endo_inds:
			endo_shifted[j] = transform_endo[j] - center_axis_diff[i]
		for j in slice_epi_inds:
			epi_shifted[j] = transform_epi[j] - center_axis_diff[i]
		if scar_stack.size:
			slice_scar_inds = np.where(scar_stack[:, 4] == cur_slice)[0]
			for j in slice_scar_inds:
				scar_shifted[j] = transform_scar[j] - center_axis_diff[i]
	# Get the new septal slice by calculating the adjustment from the topmost slice
	septal_slice_new = int(septal_slice[0][0] - endo_stack[0, 4] + 1)
	# Calculate the array to subtract from transform_abs to get the shifted apex, basal, and septal points
	sub_arr = np.array([[0, 0, 0], [0, 0, 0], [center_axis_diff[septal_slice_new][0], center_axis_diff[septal_slice_new][1], 0], 
		[center_axis_diff[septal_slice_new][0], center_axis_diff[septal_slice_new][1], 0], [center_axis_diff[septal_slice_new][0], center_axis_diff[septal_slice_new][1], 0]])
	abs_shifted = transform_abs - sub_arr
	
	# Select data and transform to array
	endo = [np.array(endo_shifted)[np.where(endo_stack[:, 4] == jz)[0]] for jz in range(1, 1+int(max(endo_stack[:, 4])))]
	epi = [np.array(epi_shifted)[np.where(epi_stack[:, 4] == jz)[0]] for jz in range(1, 1+int(max(endo_stack[:, 4])))]
	if scar_stack.size:
		scar_all = [np.array(scar_shifted)[np.where(scar_stack[:, 4] == jz)[0]] for jz in range(1, 1+int(max(endo_stack[:, 4])))]
		return([abs_shifted, endo, epi, axis_center, scar_all])
	else:
		return([abs_shifted, endo, epi, axis_center])

def transformStack(setstruct, slice_number=0, layer='endo'):
	"""Perform the actual stack rotation and calculate the rotation matrix.
	
	args:
		setstruct (dict): setstruct output from the SEGMENT file
		slice_number (int): The slice number to be used for transformation
		layer (string): Which layer to run through transformation
	"""
	# Pull x_pix, y_pix, and set run_xyz based on layer selection
	if layer == 'endo':
		x_pix = setstruct['endo_x'][:,slice_number,:]
		y_pix = setstruct['endo_y'][:,slice_number,:]
		run_xyz = True
	elif layer == 'epi':
		x_pix = setstruct['epi_x'][:,slice_number,:]
		y_pix = setstruct['epi_y'][:,slice_number,:]
		run_xyz = True
	elif layer == 'mask':
		x_pix = setstruct['mask_x'][slice_number]
		y_pix = setstruct['mask_y'][slice_number]
		run_xyz = np.isnan(sum(sum(x_pix))) < 1
	elif layer == 'long':
		run_xyz = False
	else:
		print('Incorrect Layer Selection. Defaulting to endo.')
		x_pix = setstruct['endo_x'][:,slice_number,:]
		y_pix = setstruct['endo_y'][:,slice_number,:]
		run_xyz = True
	# If endo, epi, or scar where there are no NaN values:
	if run_xyz:
		# Round the x_pix and y_pix arrays
		x_pix_round = np.round(x_pix)
		y_pix_round = np.round(y_pix)
		# Set up lists that need to be altered during the loop
		perim_length = [None] * x_pix_round.shape[0]
		xy_pts = [None] * x_pix_round.shape[0]
		for i in range(x_pix_round.shape[0]):
			# If the layer isn't scar and there are NaN values in the rounded x_pix
			#		Set the current point to all NaN and skip the rest of the loop
			if (not layer == 'mask') and (np.any(np.isnan(x_pix_round[:,i]))):
				xy_pts[i] = [np.nan, np.nan, np.nan]
				continue
			# Concatentate x_pix and y_pix into a single array
			xy_pix_round = np.array([x_pix_round[i,:].tolist(), y_pix_round[i,:].tolist()])
			# If the layer is scar, set the current perim_length point to the 2nd dimension size of xy_pix_round
			#		Otherwise, grab the unique points and then the size of the 2nd dimension
			if layer == 'mask':
				perim_length[i] = xy_pix_round.shape[1]
			else:
				perim_length[i] = ((np.unique(xy_pix_round,axis=1)).shape)[1]
			# Set perim points as linearly-spaced series of points from 0 to 1, with x_pix.shape[1]+1 number of points
			perim_pts = np.linspace(0,1,x_pix.shape[1]+1)
			# The interp points should be based on the value stored earlier in perim_length, +1
			interp_perim_pts = np.linspace(0,1,perim_length[i]+1)
			# Convert x_pix and y_pix to lists
			x_pix_arr = x_pix[i,:].tolist()
			y_pix_arr = y_pix[i,:].tolist()
			# Put the x_pix and y_pix lists into a new array together
			pix_arr = np.array([x_pix_arr, y_pix_arr])
			if layer == 'mask':
				# Sort the pix_arr array, putting 0 values at the bottom of the list
				#		Primary sort is along the second column, secondary sort along the first
				#		The conversion to and from NaN puts 0 values at the end of the list instead of the front
				pix_arr = pix_arr.astype(float)
				pix_arr[pix_arr == 0] = np.nan
				pix_arr = pix_arr[:, np.argsort(pix_arr[1], kind='mergesort')]
				pix_arr[np.isnan(pix_arr)] = 0
				pix_arr = pix_arr.astype(int)
			# Set perim_xy_pts to be pix arr, with the first point repeated at the end
			pix_append = np.reshape(pix_arr[:, 0], [2,1])
			perim_xy_pts = np.append(pix_arr.transpose(),pix_append.transpose(),axis=0)
			# Define a cubic interpolation function based on perim_pts and perim_xy_pts
			interp_func = sp.interpolate.interp1d(perim_pts,perim_xy_pts,kind='cubic',axis=0)
			# Run the interpolation function on interp_perim_pts to get interp_xy_pts
			#		This is the xy points interpolated along the new spacing
			interp_xy_pts = interp_func(interp_perim_pts)
			# Store the interpolated xy points, minus the last value (repeated from first)
			xy_pts[i] = np.array(interp_xy_pts[0:interp_xy_pts.shape[0]-1,:])
	'''# Pull values from the setstruct dict
	x_resolution = setstruct['ResolutionX']
	y_resolution = setstruct['ResolutionY']
	image_position = setstruct['ImagePosition']
	image_orientation = setstruct['ImageOrientation']
	# Pull the image orientation in x and y, then the z is the cross-product
	x_image_orientation = image_orientation[3:6]
	y_image_orientation = image_orientation[0:3]
	z_image_orientation = np.cross(y_image_orientation, x_image_orientation)
	slice_thickness = setstruct['SliceThickness']
	slice_gap = setstruct['SliceGap']'''
	# Set the z offset (always 0 in long-axis)
	if layer == 'long':
		z_offset = 0
	else:
		z_offset = slice_number
	if run_xyz:
		xyz_pts = xy_pts
		# If z points are used, add a new column to xyz_pts
		#		This column is entirely equal to the z_offset
		for i in range(x_pix_round.shape[0]):
			z_pix = -z_offset*np.ones([perim_length[i],1])
			if (layer == 'mask') or (not np.any(np.isnan(xy_pts[i].flatten()))):
				z_pix = z_pix.reshape([z_pix.shape[0], 1])
				xyz_pts[i] = np.append(xyz_pts[i], z_pix, axis=1)
				xyz_shape = xyz_pts[i].shape
	'''# Set t_o as a 4x4 identity matrix except the final column is [-1, -1, 0, 1]
	t_o = np.identity(4)
	t_o[:,3] = [-1, -1, 0, 1]
	# Set s_eye as an identity matrix except the first 3 points on the diagonal are:
	#		x_resolution, y_resolution, slice_thickness+slice_gap
	s_eye = np.identity(4)
	s_eye[0,0] = x_resolution
	s_eye[1,1] = y_resolution
	s_eye[2,2] = slice_thickness + slice_gap
	# Set r_eye as a 4x4 identity matrix except the upper right corner is a 3x3 transposed orientation matrix
	r_eye = np.identity(4)
	r_eye[0:3,0:3] = np.transpose([x_image_orientation[:], y_image_orientation[:], z_image_orientation[:]])
	# Set t_ipp to an identity matrix except the first 3 points of the final column are the image position
	t_ipp = np.identity(4)
	t_ipp[0:3,3] = image_position
	# Multiply t_ipp, r_eye, s_eye, and t_o and store as m_arr
	m_arr = t_ipp@r_eye@s_eye@t_o'''
	m_arr = _generateTransformMatrix(setstruct)
	if run_xyz:
		for i in range(x_pix_round.shape[0]):
			# As long as there are no NaN values:
			if ~np.any(np.isnan(xyz_pts[i].flatten())):
				try:
					# Append a column of ones and multiply xyz_pts by the array defined above
					mult_arr = np.transpose(np.append(xyz_pts[i], np.ones([xyz_pts[i].shape[0], 1]), axis=1))
					X = np.transpose(m_arr@mult_arr)
				except:
					print('Error encountered.')
					continue
				# Remove the column of ones at the end and store in xyz_pts
				X = X[:,0:3]
				xyz_pts[i] = X
	if layer == 'mask':
		# Return values if scar is the current layer: xyz_pts, m_arr
		if not run_xyz:
			xyz_pts[0] = [None, None, None]
		return([xyz_pts, m_arr])
	if layer == 'epi':
		# Return values if epi is the current layer: xyz_pts, m_arr
		return([xyz_pts, m_arr])
	# Set values before use
	pp_slice = None
	time_id = None
	cur_arr = np.array(setstruct['EndoPinX'][:])
	# Set z_offset to 0 if EndoPins have no timepoint changes
	if len(cur_arr.shape) < 2:
		x_pinpts = np.array(setstruct['EndoPinX'][:])
		y_pinpts = np.array(setstruct['EndoPinY'][:])
		z_offset_pp = 0
	else:
		# Set the timepoint based on where the endo pinpoints are non-zero
		time_slice = np.where(cur_arr)
		# If there is more than one timepoint, choose the first one
		#		The first dimension in cur_arr is time, the second is slices
		if len(time_slice) > 1:
			time_id = time_slice[0][0]
			pp_slice = time_slice[1][0]
		else:
			# If there aren't multiple timepoints, just take the slice
			time_id = None
			pp_slice= time_slice[0][0]
		z_offset_pp = pp_slice
		# Pull the pinpoints from the structures based on time data
		if time_id == None:
			x_pinpts = cur_arr[pp_slice]
			y_pinpts = np.array(setstruct['EndoPinY'])[pp_slice]
		else:
			x_pinpts = cur_arr[time_id][pp_slice]
			y_pinpts = np.array(setstruct['EndoPinY'])[time_id][pp_slice]
	# Round pinpoints and append the z offset
	pinpts_round = [np.round(x_pinpts), np.round(y_pinpts)]
	z_pix = -z_offset_pp * np.ones([len(x_pinpts)])
	pinpts_round.append(z_pix)
	# Append a column of ones, multiply by the m_arr as defined above, and remove ones
	pinpts_round.append(np.ones([len(x_pinpts)]))
	pp = np.transpose(np.array(m_arr)@np.array(pinpts_round))
	pp = pp[:,0:3]
	if layer == 'long':
		# Return if layer is long-axis: pinpoints (pp), m_arr
		returnList = [pp, m_arr]
	else:
		# Return if layer is endocardial: xyz_pts, pinpoints (pp), m_arr
		returnList = [xyz_pts, pp, m_arr]
	return(returnList)

def prepTransformedStack(transformed_stack, time_indices, j = 0):
	"""Process output from transformStack to append identifying information.
	
	args:
		transformed_stack: The full output from the transformStack function.
		cxyz: Array to which to append the output array
		time_indices: Which time points should be used
		j: The current slice
	
	returns:
		cxyz (array): Newly lengthened array containing formatted data from the stack
	"""
	# Pull the appropriate element from the stack
	Xd = transformed_stack[0]
	cxyz = np.array([])
	# Iterate through the 
	for k in time_indices:
		# Pull timepoint data
		Xd_k = Xd[k]
		# If every element is NaN, skip this timepoint
		if np.all(np.isnan(Xd_k)):
			continue
		# Store the slice index as an appropriately-shaped array for appending
		slice_indices = j*np.ones([Xd_k.shape[0], 1])
		# Append arrays together
		cxyz_append = np.append(Xd_k, time_indices[k]*np.ones([Xd_k.shape[0], 1]), axis=1)
		cxyz_append2 = np.append(cxyz_append, slice_indices, axis=1)
		cxyz = np.append(cxyz, cxyz_append2)

	return(cxyz)

def removeEmptySlices(setstruct, endo_x):
	"""Remove time points and slices with no contours
	
	args:
		setstruct: SEGMENT structure in the original short-axis file.
		endo_x: The x-values of the endocardial contours from SEGMENT.
		
	returns:
		list kept_slices: List of slices that have contours.
		list time_id: Which time points have been segmented.
	"""
	
	# Start with a full list of all slices
	kept_slices = np.arange(setstruct['ZSize']) + 1
	
	# Find where the slices have not received a contour trace and remove them
	no_trace = np.sum(np.isnan(endo_x), axis=2)
	delete_slices = no_trace != 0
	delete_slices = np.sum(delete_slices, axis=0) == delete_slices.shape[0]
	kept_slices = kept_slices[~delete_slices]
	time_id = np.where(no_trace[:,kept_slices[0] - 1] == 0)
	return([kept_slices, time_id])

def rotateStack(setstruct, slice_labels, layer='endo', axial_flag=False):
	"""Rotates Short-Axis Stack based on septum location.
	
	args:
		setstruct (dict): The structure from the SEGMENT data file
		slice_labels (list): The list of slices that are included in the stack
		layer (string): Which layer is being rotated (endo, epi, scar)
		axial_flag (bool): Check if image orientation is already correct
	
	returns:
		cxyz (array): The transformed and rotated contour
		m_arr (array): The multiplication array used in the transform process
		Pd (array): Unknown
		heartrate (array): Heartrate for each slice taken
	"""
	
	# Set up initial variables for future use.
	cxyz = np.array([])
	
	# Determine orientation of image and exit if it is correct
	if axial_flag:
		if (abs(abs(setstruct['ImageOrientation'][0])-1) >= 1E-7 or
			  abs(abs(setstruct['ImageOrientation'][4])-1) >= 1E-7):
			print('Image Orientation Correct')
			return(0)
	
	# Rotate stacks and format data for return.
	time_indices = range(setstruct['TSize'])
	for j in slice_labels:
		# Pass each slice to be transformed and reformat for return
		transformed_stack = transformStack(setstruct, j-1, layer)
		cxyz_slice = prepTransformedStack(transformed_stack, time_indices, j)
		cxyz = np.append(cxyz, cxyz_slice)
		# Track heartrate during each slice acquisition
		#heartrate = np.append(heartrate, [hr, slice_counter])
		# Determine what values to return based on the layer selected.
		if layer == 'epi' or layer == 'mask':
			m_arr = transformed_stack[1]
		else:
			Pd = transformed_stack[1]
			m_arr = transformed_stack[2]
	# Reshape the transformed stack as a nx5 array
	cxyz = cxyz.reshape([int(cxyz.size/5), 5])
	# Set the returned data based on the layer
	if layer == 'epi' or layer == 'mask':
		returnList = [cxyz, m_arr]
	else:
		returnList = [cxyz, Pd, m_arr]
	return(returnList)

def getMaskContour(mask_endo_stack, mask_epi_stack, mask_insertion_pts, mask_struct, mask_septal_slice, mask, apex_base_pts, transmural_filter=0.1, interp_vals=True, elim_secondary=True):
	"""Generic import for binary mask overlays onto a contour stack.
	
	The mask input should be a binary mask overlay aligned with the struct variable.
	The data is returned in the form of a ratio of wall thickness at angle bins.
	
	args:
		mask_endo_stack (array): The endo stack variable from the mask stack import
		mask_epi_stack (array): The epi stack variable from the mask stack import
		mask_insertion_pts (array): The insertion points from the mask stack import
		mask_struct (dict): The 'struct' variable returned from stack import
		mask_septal_slice (int): The septal slice returned from stack import
		mask (array): The binary mask that determines the location of the regions of interest
		transmural_filter (float): A variable that indicates the minimal transmurality to keep
		interp_vals (bool): Determine whether or not single-bin gaps should be interpolated
		elim_secondary (bool): Determine whether or not non-contiguous, smaller regions should be removed
		
	returns:
		mask_abs (array): The apex-base-septal points array from stack rotation and transformation
		mask_endo (array): The endocardial contour of the mask structure
		mask_epi (array): The epicardial contour of the mask structure
		mask_ratio (array): The inner and outer contours as a ratio of wall thickness, binned using polar angles to differentiate segments
		mask_slices (list): The list of slices that contained regions of interest for the mask
	"""
	# Get the mask XY values
	cxyz_mask, kept_slices, mask_slices = getMaskXY(mask, mask_struct)
	
	# Convert the stacks
	mask_abs, mask_endo, mask_epi, axis_center, all_mask = getContourFromStack(mask_endo_stack, mask_epi_stack, mask_struct, mask_insertion_pts, mask_septal_slice, apex_base_pts, cxyz_mask)
	
	# Get polar values and wall thickness
	mask_endo_polar, mask_epi_polar, mask_polar = convertSlicesToPolar(kept_slices, mask_endo, mask_epi, all_mask, scar_flag=True)
	wall_thickness = np.append(np.expand_dims(mask_endo_polar[:, :, 1], axis=2), np.expand_dims(mask_epi_polar[:, :, 3] - mask_endo_polar[:, :, 3], axis=2), axis=2)
	
	# Calculate ratio through wall based on angle binning
	inner_distance = mask_polar[:, :, 3] - mask_endo_polar[:, :, 3]
	outer_distance = mask_polar[:, :, 4] - mask_endo_polar[:, :, 3]
	inner_ratio = np.expand_dims(inner_distance/wall_thickness[:, :, 1], axis=2)
	outer_ratio = np.expand_dims(outer_distance/wall_thickness[:, :, 1], axis=2)
	mask_ratio = np.append(np.expand_dims(wall_thickness[:, :, 0], axis=2), np.append(inner_ratio, outer_ratio, axis=2), axis=2)
	
	# If desired, eliminate regions outside of transmurality lower limit
	if transmural_filter:
		mask_ratio[np.isnan(mask_ratio)] = 0
		low_trans = np.where(mask_ratio[:, :, 2] - mask_ratio[:, :, 1] < transmural_filter)
		mask_ratio[low_trans[0], low_trans[1], 1:] = np.nan
	
	# Interpolate single-bin gaps, if desired (for contiguous traces)
	if interp_vals:
		for i in range(mask_ratio.shape[0]):
			mask_slice = mask_ratio[i, :, :]
			mask_slice_nans = np.where(np.isnan(mask_slice[:, 1]))[0]
			mask_slice_nan_iso = [(((mask_slice_nans_i + 1) % mask_slice.shape[0]) not in mask_slice_nans) & (((mask_slice_nans_i - 1) % mask_slice.shape[0]) not in mask_slice_nans) for mask_slice_nans_i in mask_slice_nans]
			mask_slice_nan_ind = mask_slice_nans[mask_slice_nan_iso]
			for mask_ind in mask_slice_nan_ind:
				mask_inner_adj = [mask_slice[(mask_ind - 1) % mask_slice.shape[0], 1], mask_slice[(mask_ind + 1) % mask_slice.shape[0], 1]]
				mask_outer_adj = [mask_slice[(mask_ind - 1) % mask_slice.shape[0], 2], mask_slice[(mask_ind + 1) % mask_slice.shape[0], 2]]
				mask_inner_mean = np.mean(mask_inner_adj)
				mask_outer_mean = np.mean(mask_outer_adj)
				mask_slice[mask_ind, 1] = mask_inner_mean
				mask_slice[mask_ind, 2] = mask_outer_mean
		mask_ratio[i, :, :] = mask_slice
		
	# Eliminate small, non-contiguous regions, if desired (for a single trace)
	if elim_secondary:
		for i in range(mask_ratio.shape[0]):
			mask_slice = mask_ratio[i, :, :]
				# If there is no scar on this slice, go to next slice
			if np.all(np.isnan(mask_slice[:, 1])):
				continue
			# Pull the scar indices where there is no nan value
			mask_slice_nonan = np.where(~np.isnan(mask_slice[:, 1]))[0]
			# Calculate the differences between each value in the index array, and append the difference between the last and first points
			gap_dist = np.diff(mask_slice_nonan)
			gap_dist = np.append(gap_dist, mask_slice_nonan[0] + mask_slice.shape[0] - mask_slice_nonan[-1])
			# If there is only 1 gap, then there is only one scar contour, so continue to next slice
			if np.count_nonzero(gap_dist > 1) == 1:
				gap_dist[gap_dist > 1] = 1
			if np.all(gap_dist == 1):
				continue
			# The case where there are multiple non-contiguous scar traces:
			for j in range(math.floor(np.count_nonzero(gap_dist > 1)/2)):
				# Get the indices around the non-contiguous regions
				ind1 = (np.where(gap_dist > 1)[0][0] + 1) % (len(gap_dist))
				ind2 = (np.where(gap_dist > 1)[0][1] + 1) % (len(gap_dist))
				if ind1 > ind2:
					lower_index = ind2
					upper_index = ind1
				else:
					lower_index = ind1
					upper_index = ind2
				# Split the list, then set the longer list (main scar trace) as the value (essentially remove the smaller scar trace)
				slice_u2l = mask_slice_nonan[:lower_index].tolist() + mask_slice_nonan[upper_index:].tolist()
				slice_l2u = mask_slice_nonan[lower_index:upper_index].tolist()
				if len(slice_u2l) > len(slice_l2u):
					mask_slice_nonan = np.array(slice_u2l)
				else:
					mask_slice_nonan = np.array(slice_l2u)
				# Recalculate gap distance
				gap_dist = np.diff(mask_slice_nonan)
				if mask_slice_nonan[-1] == mask_slice.shape[0] - 1 and mask_slice_nonan[0] == 0:
					gap_dist = np.append(gap_dist, 1)
				if np.count_nonzero(gap_dist > 1) == 1:
					gap_dist[gap_dist > 1] = 1
			# Set up temporary arrays to pull the main slice
			new_mask_inner = np.empty(mask_slice.shape[0])
			new_mask_inner[:] = np.NAN
			new_mask_outer = new_mask_inner.copy()
			new_mask_inner[mask_slice_nonan] = mask_slice[mask_slice_nonan, 1]
			new_mask_outer[mask_slice_nonan] = mask_slice[mask_slice_nonan, 2]
			# Reassign the scar contour, overwriting non-contiguous traces with NaN
			mask_slice[:, 1] = new_mask_inner
			mask_slice[:, 2] = new_mask_outer
			mask_ratio[i, :, :] = mask_slice
	
	# Translate Endo and Epicardial contours back to cartesian
	mask_endo_cart, mask_epi_cart = shiftPolarCartesian(mask_endo_polar, mask_epi_polar, mask_endo, mask_epi, mask_slices, axis_center, wall_thickness)
	avg_wall_thickness = np.mean(wall_thickness[:, :, 1])
	
	return([mask_abs, mask_endo, mask_epi, mask_ratio, mask_slices])
	
def getMaskXY(mask, kept_slices):
	"""General form to get binary masks (such as scar data) as xy overlays.
	"""
	mask_pts = np.array(np.where(mask)) + 1
	mask_slices = np.array(list(set(mask_pts[0, :])))
	mask_pts[0] -= 1
	mask_x = [None]*mask.shape[0]
	mask_y = [None]*mask.shape[0]
	for i in mask_slices:
		temp_x = mask_pts[1, np.where(mask_pts[0, :] == i-1)]
		temp_y = mask_pts[2, np.where(mask_pts[0, :] == i-1)]
		mask_x[i-1] = temp_x
		mask_y[i-1] = temp_y
	return([mask_x, mask_y])

def convertSlicesToPolar(slices, endo, epi, scar=None, scar_flag=False, num_bins = 50):
	"""Convert an epicardial and endocardial contour (and scar data) from cartesian to polar coordinates
	
	args:
		slices (list): the slices that are being processed
		endo (array): the cartesian points of the endocardial contour
		epi (array): the cartesian points of the epicardial contour
		scar (array): the cartesian points dictating scar position
		scar_flag (boolean): boolean indicating presence of scar data
		num_bins (integer): number of angles in the range of -pi to pi
	returns:
		endo_polar (array): polar coordinates of the endocardial contour
		epi_polar (array): polar coordinates of the endocardial contour
		scar_polar (array): polar coordinates of the inner and outer scar contour
	"""
	# Set up initial variables, including binned angle values
	angles = np.linspace(-math.pi, math.pi, num_bins)
	endo_polar = []
	epi_polar = []
	scar_polar = []
	
	# Iterate through each slice and calculate polar values
	for i in slices-1:
		# Grab data from current slice
		cur_endo = endo[i]
		cur_epi = epi[i]
		if scar_flag:
			cur_scar = scar[i]
			cur_scar_shift = cur_scar - np.mean(cur_epi, axis=0) if cur_scar.size > 0 else cur_scar
		# Shift the contours by the average epicardial point
		cur_endo_shift = cur_endo - np.mean(cur_epi, axis=0)
		cur_epi_shift = cur_epi - np.mean(cur_epi, axis=0)
		# Convert the shifted cartesian points to polar
		theta_endo, rho_endo = mathhelper.cart2pol(cur_endo_shift[:, 0], cur_endo_shift[:, 1])
		theta_epi, rho_epi = mathhelper.cart2pol(cur_epi_shift[:, 0], cur_epi_shift[:, 1])
		# Define a subfunction that returns a lambda function, which provides the indices in angles where theta falls
		def getIndices(j): return lambda theta: np.where((angles[j] <= theta) & (angles[j+1] > theta))[0].tolist()
		# Shifts theta_endo from 0:2*pi to -pi:pi
		theta_endo = [te_i if te_i < math.pi else te_i-2*math.pi for te_i in theta_endo]
		theta_epi = [te_i if te_i < math.pi else te_i-2*math.pi for te_i in theta_epi]
		# Get the indices (in angles) where theta falls for endo and epi contours
		endo_idx = [getIndices(j)(theta_endo) for j in range(angles.size-1)]
		epi_idx = [getIndices(j)(theta_epi) for j in range(angles.size-1)]
		if scar_flag:
			theta_scar, rho_scar = mathhelper.cart2pol(cur_scar_shift[:, 0], cur_scar_shift[:, 1]) if cur_scar.size > 0 else [np.nan, np.nan]
			if cur_scar.size > 0: theta_scar = [ts_i if ts_i < math.pi else ts_i-2*math.pi for ts_i in theta_scar]
			scar_idx = [getIndices(j)(theta_scar) if len(getIndices(j)(theta_scar)) > 0 else [] for j in range(angles.size-1)]
			scar_bin = [[angles[j], np.mean(angles[j:j+2]), angles[j+1], min(rho_scar[scar_idx[j]]), max(rho_scar[scar_idx[j]])] if len(scar_idx[j]) > 0 else [angles[j], np.mean(angles[j:j+2]), angles[j+1], np.nan, np.nan] for j in range(angles.size-1)]
		# Create the list for the current slice containing: [angle bin min, angle bin average, angle bin max, average contour rho in that bin (nan if no contour point in that bin)]
		#	Creates this list for both endo and epi. For scar, the bin indicates minimum and maximum rho, instead of average rho
		endo_bin = [[angles[j], np.mean(angles[j:j+2]), angles[j+1], np.mean(rho_endo[endo_idx[j]])] if len(endo_idx[j]) > 0 else [angles[j], np.mean(angles[j:j+2]), angles[j+1], np.nan] for j in range(angles.size-1)]
		epi_bin = [[angles[j], np.mean(angles[j:j+2]), angles[j+1], np.mean(rho_epi[epi_idx[j]])] if len(endo_idx[j]) > 0 else [angles[j], np.mean(angles[j:j+2]), angles[j+1], np.nan] for j in range(angles.size-1)]
		# Append the current slice to the global polar matrix
		endo_polar.append(np.array(endo_bin))
		epi_polar.append(np.array(epi_bin))
		if scar_flag: scar_polar.append(np.array(scar_bin))
			
	# Convert lists to arrays:
	endo_polar = np.array(endo_polar)
	epi_polar = np.array(epi_polar)
	scar_polar = np.array(scar_polar)
	
	# Interpolate nan values in endo and epi polar arrays:
	for cur_slice in range(endo_polar.shape[0]):
		for angle in range(endo_polar.shape[1]):
			angle_less = (angle - 1) % endo_polar.shape[1]
			angle_more = (angle + 1) % endo_polar.shape[1]
			if np.isnan(endo_polar[cur_slice, angle, 3]):
				endo_polar[cur_slice, angle, 3] = (endo_polar[cur_slice, angle_less, 3] + endo_polar[cur_slice, angle_more, 3])/2
			if np.isnan(epi_polar[cur_slice, angle, 3]):
				epi_polar[cur_slice, angle, 3] = (epi_polar[cur_slice, angle_less, 3] + epi_polar[cur_slice, angle_more, 3])/2

	return([endo_polar, epi_polar, scar_polar])

def shiftPolarCartesian(endo_polar, epi_polar, endo, epi, kept_slices, axis_center, wall_thickness):
	"""Shift polar array into cartesian coordinates
	
	args:
		endo_polar (array): polar endocardial contour
		epi_polar (array): polar epicardial contour
		endo (array): endocardial points from getEndoEpiFromStack
		epi (array): epicardial pionts from getEndoEpiFromStack
		kept_slices (list): the slices with contours from SEGMENT
		axis_center (array): the center point of the apex-base axis in each slice
		wall_thickness (array): wall thickness (polar) in each angle for each slice
	"""
	cine_endo = []
	cine_epi = []
	for slices in kept_slices-1:
	
		# Convert Endo Stack from Polar to Cartesian:
		endo_pol_rho = endo_polar[slices, :, 3]
		endo_pol_theta = endo_polar[slices, :, 1]
		x_endo_cart, y_endo_cart = mathhelper.pol2cart(endo_pol_theta, endo_pol_rho)
		endo_z = [endo[slices][0, 2] for i in range(len(x_endo_cart))]
		
		# Convert Epi Stack from Polar to Cartesian:
		epi_pol_rho = epi_polar[slices, :, 3]
		epi_pol_theta = epi_polar[slices, :, 1]
		x_epi_cart, y_epi_cart = mathhelper.pol2cart(epi_pol_theta, epi_pol_rho)
		epi_z = [epi[slices][0, 2] for i in range(len(x_epi_cart))]
		
		# Shift Points back from previous Polar Origin Shift:
		polar_center = np.mean([x_epi_cart, y_epi_cart], axis=1)
		center_diff = [polar_center[0] - axis_center[slices][0], polar_center[1] - axis_center[slices][1]]
		x_endo_cart -= center_diff[0]
		y_endo_cart -= center_diff[1]
		x_epi_cart -= center_diff[0]
		y_epi_cart -= center_diff[1]

		wall = wall_thickness[slices, :, 1]
		cine_endo.append([x_endo_cart, y_endo_cart, endo_z, wall])
		cine_epi.append([x_epi_cart, y_epi_cart, epi_z, wall])
	
	# Swap the axes and reshape to get format the endo and epi traces
	temp_cine_endo = np.swapaxes(np.array(cine_endo), 1, 2)
	new_cine_endo = temp_cine_endo.reshape([temp_cine_endo.shape[0]*temp_cine_endo.shape[1], temp_cine_endo.shape[2]])
	
	temp_cine_epi = np.swapaxes(np.array(cine_epi), 1, 2)
	new_cine_epi = temp_cine_epi.reshape([temp_cine_epi.shape[0]*temp_cine_epi.shape[1], temp_cine_epi.shape[2]])

	return([new_cine_endo, new_cine_epi])

def interpShortScar(num_bins, epi_prol, endo_prol, pts_prol, epi_rot, endo_rot, pts_rot):
	theta_bins = np.linspace(0, 2*math.pi, num_bins+1)
	theta_centers = [(theta_bins[i] + theta_bins[i+1])/2 for i in range(len(theta_bins) - 1)]
	scar_width_combined = [None]*len(epi_prol)
	theta_centers_combined = [None]*len(epi_prol)
	scar_pts_combined = [None]*len(epi_prol)
	for slice_num in range(len(epi_prol)):
		wall_scar, interp_theta_centers, scar_pts = __wallScarCalculations(num_bins, epi_prol[slice_num], endo_prol[slice_num], pts_prol[slice_num], epi_rot[slice_num], endo_rot[slice_num], pts_rot[slice_num], theta_bins, theta_centers)
		theta_centers_combined[slice_num] = np.column_stack((interp_theta_centers, theta_centers))
		scar_width_combined[slice_num] = wall_scar
		scar_pts_combined[slice_num] = scar_pts
	return([theta_centers_combined, scar_width_combined, scar_pts_combined])

def interpLongScar(num_bins, epi_prol, endo_prol, pts_prol, epi_rot, endo_rot, pts_rot):
	mu_bins = np.linspace(0, 120*math.pi/180, num_bins+1)
	mu_centers = [(mu_bins[i] + mu_bins[i+1])/2 for i in range(len(mu_bins) - 1)]
	wall_scar_combined = [None]*len(epi_prol)
	interp_surface_combined = [None]*len(epi_prol)
	scar_pts_combined = [None]*len(epi_prol)
	for slice_num in range(len(epi_prol)):
		mid_point = np.mean(epi_prol[slice_num][:, 2])
		edges = [mid_point-math.pi, mid_point, mid_point+math.pi]
		endo_edge_index = mathhelper.getBinValues(endo_prol[slice_num][:, 2], edges)[1]
		epi_edge_index = mathhelper.getBinValues(epi_prol[slice_num][:, 2], edges)[1]
		pts_edge_index = mathhelper.getBinValues(pts_prol[slice_num][:, 2], edges)[1]

		wall_scar_slice = np.array([])
		interp_surface_slice = np.array([])
		scar_pts_slice = np.array([])
		for bin_val in range(len(edges)-1):
			endo_prol_bin = endo_prol[slice_num][np.where([endo_edge_val == bin_val for endo_edge_val in endo_edge_index])[0], :]
			epi_prol_bin = epi_prol[slice_num][np.where([epi_edge_val == bin_val for epi_edge_val in epi_edge_index])[0], :]
			pts_prol_bin = pts_prol[slice_num][np.where([pts_edge_val == bin_val for pts_edge_val in pts_edge_index])[0], :]
			endo_rot_bin = endo_rot[slice_num][np.where([endo_edge_val == bin_val for endo_edge_val in endo_edge_index])[0], :]
			epi_rot_bin = epi_rot[slice_num][np.where([epi_edge_val == bin_val for epi_edge_val in epi_edge_index])[0], :]
			pts_rot_bin = pts_rot[slice_num][np.where([pts_edge_val == bin_val for pts_edge_val in pts_edge_index])[0], :]
			
			wall_scar, interp_mu_centers, scar_pts = __wallScarCalculations(num_bins, epi_prol_bin, endo_prol_bin, pts_prol_bin, epi_rot_bin, endo_rot_bin, pts_rot_bin, mu_bins, mu_centers, long_axis=True)
			epi_surf_stack = np.column_stack((interp_mu_centers[:, 0], mu_centers, interp_mu_centers[:, 1]))
			interp_surface_slice = np.append(interp_surface_slice, epi_surf_stack, axis=0) if interp_surface_slice.size else epi_surf_stack
			wall_scar_slice = np.append(wall_scar_slice, wall_scar, axis=0) if wall_scar_slice.size else wall_scar
			scar_pts_slice = np.append(scar_pts_slice, scar_pts, axis=0) if scar_pts_slice.size else scar_pts
		wall_scar_combined[slice_num] = wall_scar_slice
		interp_surface_combined[slice_num] = interp_surface_slice
		scar_pts_combined[slice_num] = scar_pts_slice
	return([interp_surface_combined, wall_scar_combined, scar_pts_combined])
	
def _generateTransformMatrix(setstruct):
	# Pull values from the setstruct dict
	x_resolution = setstruct['ResolutionX']
	y_resolution = setstruct['ResolutionY']
	image_position = setstruct['ImagePosition']
	image_orientation = setstruct['ImageOrientation']
	# Pull the image orientation in x and y, then the z is the cross-product
	x_image_orientation = image_orientation[3:6]
	y_image_orientation = image_orientation[0:3]
	z_image_orientation = np.cross(y_image_orientation, x_image_orientation)
	slice_thickness = setstruct['SliceThickness']
	slice_gap = setstruct['SliceGap']
	t_o = np.identity(4)
	t_o[:,3] = [-1, -1, 0, 1]
	# Set s_eye as an identity matrix except the first 3 points on the diagonal are:
	#		x_resolution, y_resolution, slice_thickness+slice_gap
	s_eye = np.identity(4)
	s_eye[0,0] = x_resolution
	s_eye[1,1] = y_resolution
	s_eye[2,2] = slice_thickness + slice_gap
	# Set r_eye as a 4x4 identity matrix except the upper right corner is a 3x3 transposed orientation matrix
	r_eye = np.identity(4)
	r_eye[0:3,0:3] = np.transpose([x_image_orientation[:], y_image_orientation[:], z_image_orientation[:]])
	# Set t_ipp to an identity matrix except the first 3 points of the final column are the image position
	t_ipp = np.identity(4)
	t_ipp[0:3,3] = image_position
	# Multiply t_ipp, r_eye, s_eye, and t_o and store as m_arr
	m_arr = t_ipp@r_eye@s_eye@t_o
	return(m_arr)
	
def __wallScarCalculations(num_bins, epi_prol, endo_prol, pts_prol, epi_rot, endo_rot, pts_rot, angle_bins, angle_centers, long_axis=False):
	# Set up binning interpolation
	interp_function = sp.interpolate.interp1d(epi_prol[:, 1], epi_prol[:, [0, 2]], axis=0, kind='linear', fill_value='extrapolate') if long_axis else sp.interpolate.interp1d(epi_prol[:, 2], epi_prol[:, :2], axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
	interp_centers = interp_function(angle_centers)

	# Identify up indices for each bin
	_, epi_bin_index = mathhelper.getBinValues(epi_prol[:, 1], angle_bins) if long_axis else mathhelper.getBinValues(epi_prol[:, 2], angle_bins)
	_, endo_bin_index = mathhelper.getBinValues(endo_prol[:, 1], angle_bins) if long_axis else mathhelper.getBinValues(endo_prol[:, 2], angle_bins)
	_, pts_bin_index = mathhelper.getBinValues(pts_prol[:, 1], angle_bins) if long_axis else mathhelper.getBinValues(pts_prol[:, 2], angle_bins)
	
	# Calculate wall thickness
	wall_scar = np.empty((num_bins, 3))
	wall_scar[:] = np.nan
	for bin_num in range(num_bins):
		endo_indices = np.where(np.array(endo_bin_index) == bin_num)[0]
		epi_indices = np.where(np.array(epi_bin_index) == bin_num)[0]
		endo_point = np.nanmean(endo_rot[endo_indices, :], axis=0) if endo_indices.size else np.full([1, 3], np.nan)
		epi_point = np.nanmean(epi_rot[epi_indices, :], axis=0) if epi_indices.size else np.full([1, 3], np.nan)
		wall_scar[bin_num, 0] = math.sqrt(np.sum(np.square(epi_point - endo_point)))
	
	# Interpolate the wall thickness in bins without points
	wall_interp_inds = np.array(np.where(~np.isnan(wall_scar[:, 0]))[0])
	wall_nan_inds = np.array(np.where(np.isnan(wall_scar[:, 0]))[0])
	angle_centers_interp = [angle_centers[wall_interp_ind] for wall_interp_ind in wall_interp_inds]
	wall_thickness_interp = [wall_scar[wall_interp_ind, 0] for wall_interp_ind in wall_interp_inds]
	angle_centers_nans = [angle_centers[wall_nan_ind] for wall_nan_ind in wall_nan_inds]
	wall_thickness_interp_eq = sp.interpolate.interp1d(angle_centers_interp, wall_thickness_interp, fill_value='extrapolate')
	interp_thickness = wall_thickness_interp_eq(angle_centers_nans)
	wall_scar[wall_nan_inds, 0] = interp_thickness
	
	scar_pts = np.empty((num_bins, 6))
	scar_pts[:] = np.nan
	# Calculate scar penetration depth
	for bin_num in range(num_bins):
		scar_indices = np.where(np.array(pts_bin_index) == bin_num)[0]
		if len(scar_indices) > 2:
			scar_points = np.append(pts_rot[scar_indices, :], pts_prol[scar_indices, :], axis=1)
			scar_pts_ordered = scar_points[np.argsort(scar_points[:, 3]), :]
			endo_scar_pt = scar_pts_ordered[0, 0:3]
			epi_scar_pt = scar_pts_ordered[-1, 0:3]
			scar_pts[bin_num, 0:3] = endo_scar_pt
			scar_pts[bin_num, 3:6] = epi_scar_pt
			wall_scar[bin_num, 1] = math.sqrt(np.sum(np.square(epi_scar_pt - endo_scar_pt)))/wall_scar[bin_num, 0]
			wall_scar[bin_num, 2] = math.sqrt(np.sum(np.square(epi_point - epi_scar_pt)))/wall_scar[bin_num, 0]
	return(wall_scar, interp_centers, scar_pts)