import numpy as np
import scipy.io as spio
import cardiachelpers.stackhelper as stackhelper
import cardiachelpers.mathhelper as mathhelper

def loadmat(filename):
	"""
	this function should be called instead of direct spio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	
	This and all sub-functions are based on jpapon's answer here:
		https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
	"""
	data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return(_check_keys(data))

def getTimeIndices(contour, time_pts, timepoint=0):
	"""Generate indices of the contour that match with the indicated time point.
	
	args:
		contour: The endo or epi contour from getEndoEpiFromStack
		time_pts: The list of all timepoints. The fourth column from endo/epi_stack
		timepoint: The timepoint of interest (passed an an index)
	returns:
		all_slice_inds (list): A list of lists, indicating indices per slice that fall in the desired timepoint.
	"""
	# Get all indices that match the time point
	time_selected = np.where(time_pts == timepoint)[0]
	# Get the possible range of indices for each slice
	#	Initial stack is a single list, contour is split by slice, so must correct for that
	contour_slice_inds = [0] + [contour_i.shape[0] for contour_i in contour]
	contour_slice_range = np.cumsum(contour_slice_inds)
	# Set up list to store slice index lists
	all_slice_inds = [None] * len(contour)
	# Iterate through contour and pull per-slice timepoints
	for i in range(len(contour)):
		cur_slice_timepts = [time_selected_i - contour_slice_range[i] for time_selected_i in time_selected if contour_slice_range[i] <= time_selected_i < contour_slice_range[i+1]]
		all_slice_inds[i] = cur_slice_timepts
	return(all_slice_inds)
	
def _check_keys(dict_pass):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict_pass:
		if isinstance(dict_pass[key], spio.matlab.mio5_params.mat_struct):
			dict_pass[key] = _todict(dict_pass[key])
	return(dict_pass)        

def _todict(matobj):
	"""
	A recursive function which constructs from matobjects nested dictionaries
	"""
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, spio.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		elif isinstance(elem,np.ndarray):
			dict[strg] = _tolist(elem)
		else:
			dict[strg] = elem
	return(dict)

def _tolist(ndarray):
	"""
	A recursive function which constructs lists from cellarrays 
	(which are loaded as numpy ndarrays), recursing into the elements
	if they contain matobjects.
	"""
	elem_list = []            
	for sub_elem in ndarray:
		if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
			elem_list.append(_todict(sub_elem))
		elif isinstance(sub_elem,np.ndarray):
			elem_list.append(_tolist(sub_elem))
		else:
			elem_list.append(sub_elem)
	return(elem_list)

def importLongAxis(long_axis_file):
	"""Imports data from a long-axis file with pinpoints for apex and basal locations
	
	args:
		long_axis_file (string): MAT file from SEGMENT with long-axis data
	returns:
		apex_base_pts (array): The points indicating the apex and basal points indicated in the file
	"""
	# Load the file using custom loadmat function
	long_axis_data = loadmat(long_axis_file)
	# Pull the setstruct data from the global structure
	lastruct = long_axis_data['setstruct']
	# Get the apex and basal points from stack transformation
	apex_base_pts, m_arr = stackhelper.transformStack(lastruct, layer='long')
	return(apex_base_pts)
	
def importStack(short_axis_file, timepoint=0, ignore_pinpts=False):
	"""Imports the short-axis file and formats data from it.
	
	Data is imported using the custom loadmat function
	to open the struct components appropriately. All short-axis
	data is imported during this function.
	
	args:
		short_axis_file: File for the short-axis data and segmentation.
	
	returns:
		array cxyz_sa_endo: Endocardial contour stack
		array cxyz_sa_epi: Epicardial contour stack
		rv_insertion_pts: The endocardial pinpoints indicating location where RV epicardium intersects LV epicardium
		setstruct: The MATLAB structure contained within the short-axis file (part of SEGMENT's output)
		septal_slice: The slices containing the RV insertion pinpoints
	"""
	
	# Import and format the short axis stack and pull relevant variables from the structure.
	short_axis_data = loadmat(short_axis_file)
	setstruct = short_axis_data['setstruct']
	endo_x = np.array(setstruct['EndoX'])
	endo_y = np.array(setstruct['EndoY'])
	epi_x = np.array(setstruct['EpiX'])
	epi_y = np.array(setstruct['EpiY'])
	# Data can be varying dimensions, so this ensures that arrays are reshaped
	#    into the same dimensionality and adjusts axis order for improved human readability
	if endo_x.ndim >= 3:
		endo_x = np.swapaxes(endo_x, 0, 1)
		endo_x = np.swapaxes(endo_x, 2, 1)
		endo_y = np.swapaxes(endo_y, 0, 1)
		endo_y = np.swapaxes(endo_y, 2, 1)
		epi_x = np.swapaxes(epi_x, 0, 1)
		epi_x = np.swapaxes(epi_x, 2, 1)
		epi_y = np.swapaxes(epi_y, 0, 1)
		epi_y = np.swapaxes(epi_y, 2, 1)
	else:
		if endo_x.ndim < 2:
			endo_x = np.expand_dims(endo_x, 1)
			endo_y = np.expand_dims(endo_y, 1)
			epi_x = np.expand_dims(epi_x, 1)
			epi_y = np.expand_dims(epi_y, 1)
		endo_x = endo_x.transpose()
		endo_y = endo_y.transpose()
		epi_x = epi_x.transpose()
		epi_y = epi_y.transpose()
		shape = endo_x.shape
		endo_x = endo_x.reshape(1, shape[0], shape[1])
		endo_y = endo_y.reshape(1, shape[0], shape[1])
		epi_x = epi_x.reshape(1, shape[0], shape[1])
		epi_y = epi_y.reshape(1, shape[0], shape[1])

	# Process the setstruct to get time points and slices that were segmented
	kept_slices, time_id = stackhelper.removeEmptySlices(setstruct, endo_x)
	kept_slices = np.array(kept_slices)
	if not ignore_pinpts:
		setstruct, septal_slice = _findRVInsertionPts(setstruct, time_id, timepoint, endo_x, endo_y)
		'''
		time_id = np.squeeze(np.array(time_id))
		endo_pin_x = np.array(setstruct['EndoPinX'])
		endo_pin_y = np.array(setstruct['EndoPinY'])
	
		# If more than 1 timepoint is passed, use the indicated timepoint at call
		if time_id.size > 1:
			try:
				time_id = np.where(endo_pin_x)[0][timepoint]
			except(IndexError):
				print('Invalid timepoint selected. Adjusting to initial timepoint.')
				time_id = np.where(endo_pin_x)[0][0]
	
		# Ensure that the pinpoint arrays are the correct dimensionality    
		if endo_pin_x.ndim == 1:
			endo_pin_x = endo_pin_x.reshape(1, endo_pin_x.shape[0])
		if endo_pin_y.ndim == 1:
			endo_pin_y = endo_pin_y.reshape(1, endo_pin_y.shape[0])
		
		# Finds the slice where the pinpoints are placed and treats it as the septal slice
		findRVSlice = lambda pin_x: np.where([np.sum(pin_x[:, cur_slice][0]) for cur_slice in range(pin_x.shape[1])])
		septal_slice = findRVSlice(endo_pin_x)
	
		# Extract the x and y pinpoints for the current contour
		x_pins = np.array(endo_pin_x[time_id, septal_slice][0][0])
		y_pins = np.array(endo_pin_y[time_id, septal_slice][0][0])
		endo_pins = np.array([x_pins, y_pins]).transpose()
		
		# Calculate the Septal Mid-Point from the pinpoints
		sept_pt = mathhelper.findMidPt(endo_pins, time_id, septal_slice, endo_x, endo_y)

		# Add the midpoint to the x and y pinpoint list and add it back to setstruct
		#        This part requires somewhat complex list comprehensions to reduce clutter and due to the complexity of the data format
		new_endo_pin_x = [np.append(cur_endo_pin_x, sept_pt[0]).tolist() if cur_endo_pin_x else cur_endo_pin_x for cur_endo_pin_x in endo_pin_x .flatten()]
		new_endo_pin_y = [np.append(cur_endo_pin_y, sept_pt[1]).tolist() if cur_endo_pin_y else cur_endo_pin_y for cur_endo_pin_y in endo_pin_y.flatten()]
		endo_pin_x = np.reshape(new_endo_pin_x, endo_pin_x.shape)
		endo_pin_y = np.reshape(new_endo_pin_y, endo_pin_y.shape)
	
		# Store relevant variables in the setstruct dictionary for use downstream
		setstruct['EndoPinX'] = endo_pin_x
		setstruct['EndoPinY'] = endo_pin_y
		'''
	setstruct['KeptSlices'] = kept_slices
	setstruct['endo_x'] = endo_x
	setstruct['endo_y'] = endo_y
	setstruct['epi_x'] = epi_x
	setstruct['epi_y'] = epi_y
	
	# Rotate the endo and epi contours (and pinpoints with the endo contour)
	cxyz_sa_endo, rv_insertion_pts, _ = stackhelper.rotateStack(setstruct, kept_slices, layer='endo')
	cxyz_sa_epi, _ = stackhelper.rotateStack(setstruct, kept_slices, layer='epi')
	
	# Define return list based on pinpoint running
	return_list = [cxyz_sa_endo, cxyz_sa_epi, rv_insertion_pts, setstruct] if ignore_pinpts else [cxyz_sa_endo, cxyz_sa_epi, rv_insertion_pts, setstruct, septal_slice]
	return(return_list)
	
def _findRVInsertionPts(setstruct, time_id, timepoint, endo_x, endo_y):
	"""Calculate RV Insertion point location and calculates a midpoint, also determines the septal slice.
	
	This is intended as a helper function that simply allows separation between 
	"""
	time_id = np.squeeze(np.array(time_id))
	endo_pin_x = np.array(setstruct['EndoPinX'])
	endo_pin_y = np.array(setstruct['EndoPinY'])

	# If more than 1 timepoint is passed, use the indicated timepoint at call
	if time_id.size > 1:
		try:
			time_id = np.where(endo_pin_x)[0][timepoint]
		except(IndexError):
			print('Invalid timepoint selected. Adjusting to initial timepoint.')
			time_id = np.where(endo_pin_x)[0][0]

	# Ensure that the pinpoint arrays are the correct dimensionality    
	if endo_pin_x.ndim == 1:
		endo_pin_x = endo_pin_x.reshape(1, endo_pin_x.shape[0])
	if endo_pin_y.ndim == 1:
		endo_pin_y = endo_pin_y.reshape(1, endo_pin_y.shape[0])
	
	# Finds the slice where the pinpoints are placed and treats it as the septal slice
	findRVSlice = lambda pin_x: np.where([np.sum(pin_x[:, cur_slice][0]) for cur_slice in range(pin_x.shape[1])])
	septal_slice = findRVSlice(endo_pin_x)

	# Extract the x and y pinpoints for the current contour
	x_pins = np.array(endo_pin_x[time_id, septal_slice][0][0])
	y_pins = np.array(endo_pin_y[time_id, septal_slice][0][0])
	endo_pins = np.array([x_pins, y_pins]).transpose()
	
	# Calculate the Septal Mid-Point from the pinpoints
	sept_pt = mathhelper.findMidPt(endo_pins, time_id, septal_slice, endo_x, endo_y)

	# Add the midpoint to the x and y pinpoint list and add it back to setstruct
	#        This part requires somewhat complex list comprehensions to reduce clutter and due to the complexity of the data format
	new_endo_pin_x = [np.append(cur_endo_pin_x, sept_pt[0]).tolist() if cur_endo_pin_x else cur_endo_pin_x for cur_endo_pin_x in endo_pin_x .flatten()]
	new_endo_pin_y = [np.append(cur_endo_pin_y, sept_pt[1]).tolist() if cur_endo_pin_y else cur_endo_pin_y for cur_endo_pin_y in endo_pin_y.flatten()]
	endo_pin_x = np.reshape(new_endo_pin_x, endo_pin_x.shape)
	endo_pin_y = np.reshape(new_endo_pin_y, endo_pin_y.shape)

	# Store relevant variables in the setstruct dictionary for use downstream
	setstruct['EndoPinX'] = endo_pin_x
	setstruct['EndoPinY'] = endo_pin_y
	
	return(setstruct, septal_slice)