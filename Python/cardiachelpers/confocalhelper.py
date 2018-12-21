import scipy.ndimage
import numpy as np
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
from cardiachelpers import importhelper
from cardiachelpers import mathhelper
from skimage import io
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.feature
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path
import warnings
import math
from scipy import interpolate as spinterp
Image.MAX_IMAGE_PIXELS = None

def splitChannels(images_in, pull_channel=-1):
	"""Splits an image (or multiple images) to its different channels, then returns a nested list of the channels.
	"""
	# Determine if images in are a list or a single image
	if isinstance(images_in, list):
		split_images = [None]*len(images_in)
		# Iterate through images
		for i in range(len(images_in)):
			im = images_in[i]
			# Convert the image to an array and get the number of bands
			image_arr = np.array(im)
			num_channels = len(im.getbands())
			# Slice array based on channels selected (channel = -1 indicates all channels)
			if pull_channel >= 0 and pull_channel < num_channels:
				# Rebuild an image from the sliced array to isolate the channel
				split_images[i] = Image.fromarray(image_arr[:, :, pull_channel], mode=im.mode)
			else:
				# Create a new image for each channel and store as a list
				split_by_band = [None]*num_channels
				for chan_ind in range(num_channels):
					split_by_band[chan_ind] = Image.fromarray(image_arr[:, :, chan_ind], mode=im.mode)
				split_images[i] = split_by_band
		return(split_images)
	else:
		# Convert the image to an array and get the number of bands
		image_arr = np.array(images_in)
		num_channels = len(images_in.getbands())
		# Slice array based on channels selected (channel = -1 indicates all channels)
		if pull_channel >= 0 and pull_channel < num_channels:
			# Rebuild image from the sliced array to isolate the channel
			split_by_band = Image.fromarray(image_arr[:, :, pull_channel], mode=images_in.mode)
		else:
			# Create a new image for each channel and store as a list
			split_by_band = [None]*num_channels
			for chan_ind in range(num_channels):
				split_by_band[chan_ind] = Image.fromarray(image_arr[:, :, chan_ind], mode=images_in.mode)
		return(split_by_band)
		
def splitImageFrames(image_in):
	"""Splits an image (or multiple images as a list) to its different frames, and returns a list containing the images.
	"""
	# Determine if images in are a list or a single image
	if isinstance(image_in, list):
		full_images = []
		# Iterate through images, creating a sublist of frames for each image
		for image in image_in:
			split_image = [None]*image.n_frames
			# Iterate through frames and copy each frame independently, converting to RGB
			for i in range(image.n_frames):
				image.seek(i)
				split_image[i] = image.copy()
			full_images.append(split_image)
		return(full_images)
	else:
		split_image = [None]*image_in.n_frames
		# Iterate through frames and copy each frame independently, converting to RGB
		for i in range(image_in.n_frames):
			image_in.seek(i)
			split_image[i] = image_in.copy()
		return(split_image)

def stitchImages(images_in, image_x_inds, image_y_inds, overlap=0.1, stitched_type=False, save_pos=False):
	"""Piece images together based on x and y indices to form a single large image.
	"""
	# Get the maximum number of images in each direction
	x_range = max(image_x_inds)
	y_range = max(image_y_inds)
	# Pull the first image to be stitched
	first_sublist = images_in[0]
	while isinstance(first_sublist, list):
		first_sublist = first_sublist[0]
	# Get the size of the tiling from the input image
	if isinstance(first_sublist, Image.Image):
		tile_size = first_sublist.size
	else:
		return(False)
	# Create an image based on the width of the input image that should hold the full stitched image
	if not stitched_type:
		stitched_type = first_sublist.mode
	stitched_image = Image.new(stitched_type, (int(x_range*(1-overlap)*tile_size[0]), int(y_range*(1-overlap)*tile_size[1])))
	# Iterating through each image, copy the information to the stitched image based on position
	for image_num in range(len(images_in)):
		# Calculate X and Y positions for the image
		x_pos = int((x_range-image_x_inds[image_num])*(1-overlap)*tile_size[0])
		y_pos = int(image_y_inds[image_num]*(1-overlap)*tile_size[1])
		# Select the appropriate image to use
		cur_image = images_in[image_num]
		while isinstance(cur_image, list):
			cur_image = cur_image[0]
		if not isinstance(cur_image, Image.Image):
			return(False)
		# Paste the image into the large stitched image
		stitched_image.paste(cur_image, (x_pos, y_pos))
	# Determine if the image is being saved to disc or returned
	if save_pos:
		try:
			# Save the image to the indicated file
			stitched_image.save(save_pos)
			return(True)
		except Exception as e:
			raise(e)
	else:
		# Don't save the image, but return it as an Image object
		return(stitched_image)

def stitchImagesAbsolute(images_in, image_x_pos, image_y_pos, stitched_type=False, save_pos=False):
	x_max_image = np.where(image_x_pos == max(image_x_pos))[0][0]
	y_max_image = np.where(image_y_pos == max(image_y_pos))[0][0]
	x_size = max(image_x_pos) + images_in[x_max_image].size[0]
	y_size = max(image_y_pos) + images_in[y_max_image].size[1]
	# Pull the first image to be stitched
	first_sublist = images_in[0]
	while isinstance(first_sublist, list):
		first_sublist = first_sublist[0]
	# Get the size of the tiling from the input image
	if isinstance(first_sublist, Image.Image):
		tile_size = first_sublist.size
	else:
		return(False)
	# Create an image based on the width of the input image that should hold the full stitched image
	if not stitched_type:
		stitched_type = first_sublist.mode
	stitched_image = Image.new(stitched_type, (x_size, y_size))
	for image_num, cur_image in enumerate(images_in):
		x_pos = image_x_pos[image_num]
		y_pos = image_y_pos[image_num]
		stitched_image.paste(cur_image, (x_pos, y_pos))
	if save_pos:
		try:
			# Save the image to the indicated file
			stitched_image.save(save_pos)
			return(True)
		except Exception as e:
			raise(e)
	else:
		# Don't save the image, but return it as an Image object
		return(stitched_image)
		
def getImagePositions(image_files):
	"""Small function to pull image position data from Volocity-exported TIF files.
	"""
	# Set which data is desired
	data_categories = ['XLocationMicrons', 'YLocationMicrons', 'XCalibrationMicrons', 'YCalibrationMicrons']
	
	# Create array to store all categorical data for each image
	image_positions = np.empty((len(image_files), len(data_categories)))
	for file_num, tif_file in enumerate(image_files):
		with open(tif_file, encoding='utf8', errors='ignore') as temp_file:
			file_lines = temp_file.readlines()
			for line in file_lines:
				# Line split by '=' represents a property (if length = 2)
				line_split = line.split('=')
				if len(line_split) == 2:
					# If the category is in the list of desired categories, store it by appropriate column
					if line_split[0] in data_categories:
						image_positions[file_num, data_categories.index(line_split[0])] = float(line_split[1])
		
	# Create a dict object to represent which data is in each column
	column_dict = {data_categories[i] : i for i in range(len(data_categories))}
	return([image_positions, column_dict])

def getImageGrid(image_files, image_locs, locs_dict):
	"""Generate the x and y indices for image placement in stitching.
	
	X and Y indices are determined based on image data indicating absolute position and the "buckets" indicated
	image locations, pulled earlier from the files.
	"""
	# Determine how to find x and y positions
	x_col = locs_dict['XLocationMicrons']
	y_col = locs_dict['YLocationMicrons']
	# Pull absolute x and y locations from columns
	locs_x = image_locs[:, x_col]
	locs_y = image_locs[:, y_col]
	# Determine the possible slots for x and y data
	x_slots = np.unique(np.round(locs_x))
	y_slots = np.unique(np.round(locs_y))
	# Determine the relative positions for each image
	img_x_inds = [np.where(np.round(locs_x[i]) == x_slots)[0] for i in range(locs_x.shape[0])]
	img_y_inds = [np.where(np.round(locs_y[i]) == y_slots)[0] for i in range(locs_y.shape[0])]
	
	return(np.column_stack((img_x_inds, img_y_inds)))
	
def compressImages(images_in, image_scale=0.5):
	"""Resize the raw images in the model, to allow easier manipulation and display.
	
	Resets the compressed_images field on-call, to allow only one set of compressed images per instance.
	
	args:
		image_scale (float): Determines the ratio of new image size to old image size
	"""
	# Determine if input images are a list
	if isinstance(images_in, list):
		# Determine image size based on the ratio to the input image
		compressed_images = [None]*len(images_in)
		new_size = [int(image_scale*dimension) for dimension in images_in[0].size]
		# Iterate through images and resize using Lanczos reconstruction
		for image_num in range(len(images_in)):
			compressed_images[image_num] = images_in[image_num].resize(new_size, Image.LANCZOS) if not(image_scale == 1) else images_in[image_num]
		return(compressed_images)
	else:
		# Establish new size from ratio and resize image using Lanczos reconstruction
		new_size = [int(image_scale*dimension) for dimension in images_in.size]
		compressed_image = images_in.resize(new_size, Image.LANCZOS) if not(image_scale == 1) else images_in
		return(compressed_image)
		
def readImageGrid(file_name):
	"""Read image grid information from a file instead of pulling it from image file data.
	"""
	im_grid = np.empty([])
	with open(file_name) as grid_file:
		# Iterate through each line of the input file
		for file_line in grid_file.readlines():
			# Split the line string using ',' as a delimiter, stripping whitespace and converting to int
			cur_inds = np.array([int(ind.strip()) for ind in file_line.split(',')])
			# Either create the new array or append locations to the growing array row-by-row
			if im_grid.ndim:
				im_grid = np.vstack((im_grid, cur_inds))
			else:
				im_grid = cur_inds
	return(im_grid)
	
def writeImageGrid(image_grid, file_name):
	"""Write image grid information to a file, to allow easier access.
	"""
	# Generate a blank file to use for grid storage
	open(file_name, 'w').close()
	with open(file_name, 'w') as grid_file:
		# Iterate through rows in the locations
		for row in range(image_grid.shape[0]):
			# Pull respective x and y indices from the input grid array
			x_ind = image_grid[row, 0]
			y_ind = image_grid[row, 1]
			# Write the locations to the file with a ',' separator and move to a new line
			grid_file.write(str(x_ind) + ',' + str(y_ind) + '\n')
	return(True)
	
def splitForeground(image_file):
	image_in = Image.open(image_file)
	image_in = compressImages(image_in)
	mask = Image.new('1', image_in.size)
	mask, contour = _getThresholdMask(image_in)
		
	save_dir, im_filename = os.path.split(image_file)
	
	cur_file = im_filename.split('.')[0] + '_FGMask.png'
	filename = os.path.join(save_dir, cur_file)
	plt.imsave(filename, mask, cmap=cm.gray)
	return([filename, contour])
	
def openModelImage(image_file):
	"""Opens a 32-bit image stack representing a full stack of confocal slices for a heart.
	"""
	full_arr = io.imread(image_file)
	im_list = [Image.fromarray(full_arr[i, :, :]) for i in range(full_arr.shape[0])] if full_arr.ndim > 2 else Image.fromarray(full_arr)
	return(im_list)

def skeletonizeImage(input_image):
	"""Create a 1-pixel-wide trace of a larger thicker path.
	"""
	# Convert to binary image and skeletonize
	binary_array = np.array(input_image) > 0
	skeleton_contour = skimage.morphology.skeletonize(binary_array)
	return(skeleton_contour)
	
def contourMaskImage(mask_image):
	"""Using a foreground mask image, create a trace of the boundaries where foreground and background meet.
	"""
	edge_contour = skimage.feature.canny(np.array(mask_image), sigma=2)
	return(edge_contour)
	
def splitImageObjects(binary_arr):
	"""Get the connected objects from an image, and gather the two large paths.
	
	Returns:
		endo_path: The smaller of the two objects.
		epi_path: The larger of the two objects.
		labeled_arr: The input array, labeled with integer values indicating object connectivity.
	"""
	# Create the binary structure to indicate connectivity in corners and edges
	connecting_mat = scipy.ndimage.generate_binary_structure(2, 2)
	# Label the array for object connectivity, get the first two objects
	labeled_arr, num_features = scipy.ndimage.label(binary_arr, structure=connecting_mat)
	path_1 = np.array(np.where(labeled_arr == 1)).swapaxes(0, 1)
	path_2 = np.array(np.where(labeled_arr == 2)).swapaxes(0, 1)
	# Set endo and epi paths based on path size (epi is larger)
	endo_path = path_1 if path_1.size < path_2.size else path_2
	epi_path = path_1 if path_1.size > path_2.size else path_2
	return(endo_path, epi_path, labeled_arr)

def smoothPathTrace(pt_path):
	"""Attempt to interpolate the points around the path to form a smoother path trace.
	"""
	center_pt = np.mean(pt_path, axis=0)
	shifted_path = np.array([pt_path[i, :] - center_pt for i in range(pt_path.shape[0])])
	
	interp_theta_rads = np.linspace(0, 2*math.pi, 360)
	averaging_theta_rads = np.linspace(0, 2*math.pi, 30)
	averaging_theta_bins = [[averaging_theta_rads[i], averaging_theta_rads[i+1]] for i in range(len(averaging_theta_rads)-1)]
	
	path_theta, path_rho = mathhelper.cart2pol(shifted_path[:, 0], shifted_path[:, 1])
	
	#path_interp_eq = spinterp.interp1d(path_theta, path_rho, kind='linear', bounds_error=False, fill_value='extrapolate')
	path_theta_sort_inds = np.argsort(path_theta, kind='mergesort')
	path_interp_eq = spinterp.UnivariateSpline(path_theta[path_theta_sort_inds], path_rho[path_theta_sort_inds], k=1)
	
	path_interp_rho = path_interp_eq(interp_theta_rads)

	smoothed_x, smoothed_y = mathhelper.pol2cart(interp_theta_rads, path_interp_rho)
	smoothed_x = np.append(smoothed_x, smoothed_x[0])
	smoothed_y = np.append(smoothed_y, smoothed_y[0])
	
	return([smoothed_x, smoothed_y])
	
def orderPathTrace(pt_path):
	center_pt = np.mean(pt_path, axis=0)
	shifted_path = np.array([pt_path[i, :] - center_pt for i in range(pt_path.shape[0])])
	
	path_theta, path_rho = mathhelper.cart2pol(shifted_path[:, 0], shifted_path[:, 1])
	
	path_pt_order = np.argsort(path_theta, kind='mergesort')
	
	pt_path_ordered = pt_path[path_pt_order, :]
	return(pt_path_ordered)
	
def formatContourForModel(pt_list):
	num_slices = len(pt_list)
	max_path_length = max([pt_arr.shape[0] for pt_arr in pt_list])
	path_x = np.full([1, num_slices, max_path_length], np.nan)
	path_y = np.full([1, num_slices, max_path_length], np.nan)
	for slice_num in range(num_slices):
		pt_arr = pt_list[slice_num]
		for pt_ind in range(pt_arr.shape[0]):
			path_x[0, slice_num, pt_ind] = pt_arr[pt_ind, 0]
			path_y[0, slice_num, pt_ind] = pt_arr[pt_ind, 1]
	return(path_x, path_y)
	
def _getThresholdMask(image_in):
	image_arr = np.array(image_in)
	
	if np.any(image_arr):
		thresh = skimage.filters.threshold_minimum(image_arr)
		mask = image_arr > thresh
		fr_filter_mask = skimage.filters.frangi(mask)
		fr_filter_thresh = skimage.filters.threshold_otsu(fr_filter_mask)*0.25
		contours = skimage.measure.find_contours(fr_filter_mask, fr_filter_thresh)
		fr_thresh_mask = fr_filter_mask > fr_filter_thresh
		return([fr_thresh_mask, contours])
	else:
		return([image_arr, [None]])