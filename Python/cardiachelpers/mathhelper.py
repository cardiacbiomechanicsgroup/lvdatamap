import numpy as np
import scipy as sp
import math

def pol2cart(theta, rho):
	"""Convert polar (theta, rho) coordinates to cartesian (x, y) coordinates"""
	x = rho * np.cos(theta)
	y = rho * np.sin(theta)
	return ([x, y])

def cart2pol(x, y):
	"""Convert cartesian (x,y) coordinates to polar (theta, rho) coordinates"""
	rho = np.sqrt(np.square(x) + np.square(y))
	theta = np.arctan2(y,x)
	theta = np.where(theta < 0, theta + 2*np.pi, theta)
	return np.array([theta, rho])

def prolate2cart(m, n, p, focus):
	"""Convert passed lambda, mu, theta from prolate to cartesian based on focus."""
	x = focus * np.cosh(m) * np.cos(n)
	y = focus * np.sinh(m) * np.sin(n) * np.cos(p)
	z = focus * np.sinh(m) * np.sin(n) * np.sin(p)
	return([x, y, z])
	
def cart2prolate(x, y, z, focus):
	"""Convert passed x, y, z from cartesian to prolate based on focus."""
	# Checks if the data is a matrix to set the loop.
	matrix_flag = False
	if isinstance(x, np.ndarray):
		# Store the shape of the passed array, then flatten them
		input_shape = x.shape
		x = np.reshape(x, x.size, 1)
		y = np.reshape(y, y.size, 1)
		z = np.reshape(z, z.size, 1)
		len_x = x.size
		matrix_flag = True
	else:
		len_x = 1
	m = np.zeros((len_x, 1))
	n = np.zeros((len_x, 1))
	p = np.zeros((len_x, 1))
	# Loop through the vaues and perform equations to convert to prolate.
	for jz in range(len_x):
		# Pull values from array if arrays were passed
		if matrix_flag:
			x1 = x[jz]
			x2 = y[jz]
			x3 = z[jz]
		else:
			x1 = x
			x2 = y
			x3 = z
		a1 = x1**2 + x2**2 + x3**2 - focus**2
		a2 = math.sqrt((a1**2)+4*(focus**2)*((x2**2)+(x3**2)))
		a3 = 2*(focus**2)
		a4 = max([(a1+a2)/a3, 0])
		a5 = max([(a2-a1)/a3, 0])
		a6 = math.sqrt(a4)
		a7 = min([math.sqrt(a5), 1])
		a8 = math.asin(a7) if abs(a7) <= 1 else 0
		if abs(a7) > 1: print('SLH_CMI_C2P: A8 is zero')
		if x3==0 or a6==0 or a7==0:
			a9 = 0
		else:
			a9 = x3 / (focus*a6*a7) if abs(a6*a7)>0 else 0
		a9 = math.pi/2 if a9 >= 1 else -math.pi/2 if a9 <= -1 else math.asin(a9)
		# Set the prolate values lambda (z1), mu (z2), and theta (z3)
		z1 = math.log(a6 + math.sqrt(a4+1))
		z2 = a8 if x1 >= 0 else math.pi - a8
		z3 = math.fmod(a9+2*math.pi, 2*math.pi) if x2 >= 0 else math.pi-a9
		# Store the singular values into the array
		if matrix_flag:
			m[jz] = z1
			n[jz] = z2
			p[jz] = z3
		else:
			m = z1
			n = z2
			p = z3
	# Reshape the mu, nu, and phi arrays based on the input arrays
	if matrix_flag:
		m = np.reshape(m, input_shape, order='F')
		n = np.reshape(n, input_shape, order='F')
		p = np.reshape(p, input_shape, order='F')
	return([m, n, p])
	
def findMidPt(endo_pins, time_id, septal_slice, endo_x, endo_y):
	"""Calculate the mid-septal point based on the pinpoints already placed
	
	args:
		endo_pins: array containing the rv insertion pinpoints
		time_id: which time point to use for calculation
		septal_slice: which slice to use for calculation
		endo_x: the x values defining the endocardial contour
		endo_y: the y values defining the endocardial contour
	returns:
		array mid_pt: The septal midpoint between the two other pinpoints
	"""
	if endo_pins.ndim < 2:
		endo_pins = np.expand_dims(endo_pins, 0)
	# Get mean point between the 2 pinpoints.
	mean_pt = np.mean(endo_pins, axis=0).reshape([2, 1])

	# Calculate the perpindicular line between the two points (just slope, no intercept)
	slope = (endo_pins[1,1] - endo_pins[0,1])/(endo_pins[1,0] - endo_pins[0,0])
	perp_slope = -1/slope
	
	# Get the current slice and shift it by the mean point
	cur_slice = np.array([endo_x[time_id, septal_slice, :], endo_y[time_id, septal_slice, :]])
	cur_shape = cur_slice.shape
	cur_slice = cur_slice.reshape(cur_shape[0], cur_shape[3])
	cur_slice = cur_slice - mean_pt
	
	# Convert the slice into polar values
	polar_coords = cart2pol(cur_slice[0,:], cur_slice[1,:])[:,1:]
	
	# Get the theta values for the perpindicular line (polar theta)
	perp_dot = np.dot([1, perp_slope], [1, 0])
	calcNorm = lambda arr_in: np.sqrt(np.sum(np.square(arr_in)))
	perp_norm = calcNorm([1, 1*perp_slope])
	th1 = np.arccos(perp_dot/perp_norm)
	th2 = th1 + np.pi;
	
	# Calculate the rho values for the two theta values by interpolation
	r_interp = sp.interpolate.interp1d(polar_coords[0,:], polar_coords[1,:])
	r1 = r_interp(th1)
	r2 = r_interp(th2)
	r = r1 if r1<r2 else r2
	theta = th1 if r1<r2 else th2
	
	# Reconvert the interpolated rho and theta to cartesian
	mid_pt = (pol2cart(theta, r) + mean_pt.reshape([1,2])).reshape(2)
	return(mid_pt)

def getBinValues(values_in, bin_edges):
	bin_index = np.full((len(values_in)), np.nan) if isinstance(values_in, list) else np.full((values_in.size), np.nan)
	bin_counts = [None]*(len(bin_edges) - 1)
	value_list = values_in if isinstance(values_in, list) else values_in.flatten()
	for cur_bin in range(len(bin_edges) - 1):
		bin_bot = bin_edges[cur_bin]
		bin_top = bin_edges[cur_bin + 1]
		
		items_in_bin = np.where(np.logical_and(np.greater_equal(value_list, bin_bot), np.less(value_list, bin_top)))[0]
		bin_counts[cur_bin] = len(items_in_bin)
		bin_index[items_in_bin] = cur_bin
	#if bin_index.size:
		#bin_index[np.where(np.greater(value_list, max(bin_edges)))[0]] = max(bin_index)
		#bin_index[np.where(np.less(value_list, min(bin_edges)))[0]] = min(bin_index)
	return(bin_counts, list(bin_index.astype(int)))
		
def getAngleRange(angles):
	"""Get the leftmost and rightmost values from a passed series of angles.
	
	The angles should be circular, with a total possible range of 2*pi. The purpose of this function
	is to allow finding the circular extent of angles that cross the origin point.
	
	args:
		angles (float arr-like): The angles for which ranges are being determined.
	returns:
		angle_min (float): The "minimum" angle (clockwise)
		angle_max (float): The "maximum" angle (clockwise)
		direction (bool): True if the scar does not pass through the origin (angle flip point)
	"""
	# Get maximum and minimum angle values and subtract
	angle_max = np.max(angles)
	angle_min = np.min(angles)
	init_range = angle_max - angle_min
	# If the range is less than 2*pi, you don't cross the origin
	if init_range < 6:
		direction = True
		return([angle_min, angle_max, direction])
	else:
		# Sort angles from minimum -> maximum, append initial value to the end, increased by 2*pi
		angles_sorted = np.sort(angles)
		angles_sorted = np.append(angles_sorted, angles_sorted[0] + 2*math.pi)
		# Calculate the moving differential
		angles_diff = [angles_sorted[i+1] - angles_sorted[i] for i in range(len(angles))]
		# The "true minimum" (most counterclockwise angle) is immediately after the largest gap
		angle_min = angles_sorted[np.argmax(angles_diff, axis=0) + 1]
		angle_max = angles_sorted[np.argmax(angles_diff, axis=0)]
		# Determine directionality to ensure that scar values are between the minimum and maximum appropriately
		direction = np.all(np.bitwise_and(angles >= angle_min, angles <= angle_max))
	return([angle_min, angle_max, direction])