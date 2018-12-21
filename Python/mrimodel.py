# -*- coding: utf-8 -*-
"""
Contains all class definitions and imports necessary to implement MRI segmentation imports
and alignments. Built based on MRI Processing MATLAB pipeline by Thien-Khoi Phung.

Created on Fri Jul 21 11:27:52 2017

@author: cdw2be
"""

# This is just a test comment to verify branch success.

# Imports
import math
import scipy as sp
import numpy as np
import matplotlib.pyplot as mplt
from mpl_toolkits.mplot3d import Axes3D
from cardiachelpers import importhelper
from cardiachelpers import stackhelper
from cardiachelpers import mathhelper
# Call this to set appropriate printing to view all data from arrays
np.set_printoptions(threshold=np.inf)
np.core.arrayprint._line_width = 160

class MRIModel():

	"""Contains contours based on multiple MRI modalities.

	Class containing contours based on multiple MRI modalities. Can be used to
	import Cine Black-Blood stacks and LGE stacks. Use Long-Axis image to determine
	vertical slice orientation.
	
	Attributes:
		scar (bool): Whether or not the model has an LGE data set
		dense (bool): Whether or not the model has a DENSE data set
		*_endo (array): Endocardial contour for the data set indicated by *
		*_epi (array): Epicardial contour for the data set indicated by *
		*_apex_pt (array): Apex point for the data set indicated by *
		*_basal_pt (array): Basal point for the data set indicated by *
		*_septal_pts (array): RV Insertion points selected in segment for the data set indicated by *
		*_slices (list): List of slices that were traced in SEGMENT for the data set indicated by *
		scar_ratio (array): Ratio of scar versus wall thickness by angle bins
		aligned_scar (array): The scar contour mapped from LGE to cine stack
	"""
	
	def __init__(self, cine_file, la_file, sa_scar_file=None, la_scar_files=None, dense_file=None):
		"""Initialize new MRI model and select applicable files.
		
		Args:
			scar: Determines whether to import an LGE image stack.
			dense: Determines whether to import a DENSE image stack.
			
		Returns: New instance of MRIModel
		"""
		self.cine_file = cine_file
		self.long_axis_file = la_file
		if sa_scar_file:
			self.sa_scar_file = sa_scar_file
			self.scar = True
			if la_scar_files:
				self.la_scar_files = la_scar_files
				self.la_scar = True
			else:
				self.la_scar_files = None
				self.la_scar = False
		else:
			self.sa_scar_file = None
			self.scar = False
		
		if dense_file:
			self.dense_file = dense_file
			self.dense = True
		else:
			self.dense_file = None
			self.dense = False
		
		# Define variables used in all models
		self.apex_base_pts = importhelper.importLongAxis(self.long_axis_file)
		
		self.cine_endo = []
		self.cine_epi = []
		self.cine_apex_pt = []
		self.cine_basal_pt = []
		self.cine_septal_pts = []
		self.cine_slices = []
		
		# Set up scar-specific variables
		self.lge_endo = []
		self.lge_epi = []
		self.scar_ratio = []
		self.lge_apex_pt = []
		self.lge_basal_pt = []
		self.lge_septal_pts = []
		self.scar_slices = []
		self.aligned_scar = []
		
		# Set up DENSE-specific variables
		self.dense_endo = []
		self.dense_epi = []
		self.dense_pts = []
		self.dense_displacement = []
		self.dense_slices = []
		self.dense_aligned_pts = []
		self.dense_aligned_displacement = []
		self.radial_strain = []
		self.circumferential_strain = []
	
	def importCine(self, timepoint=0):
		"""Import the black-blood cine stack.

		Returns:
			boolean: True if import was successful.
		"""
		# Import the Black-Blood cine (short axis) stack
		endo_stack, epi_stack, self.rv_insertion_pts, sastruct, septal_slice = importhelper.importStack(self.cine_file, timepoint)
		
		_, endo_for_slice, epi_for_slice, _ = stackhelper.getContourFromStack(endo_stack, epi_stack, sastruct, self.rv_insertion_pts, septal_slice, self.apex_base_pts)
		self.endo_slices = [np.unique(np.round(endo_for_slice[i][:, 2], 2))[0] for i in range(len(endo_for_slice))]
		self.epi_slices = [np.unique(np.round(epi_for_slice[i][:, 2], 2))[0] for i in range(len(epi_for_slice))]
		
		kept_slices = sastruct['KeptSlices']

		apex_pt = self.apex_base_pts[0, :]
		base_pt = self.apex_base_pts[1, :]
		center_septal_pt = np.expand_dims(self.rv_insertion_pts[2, :], 0)
		
		time_pts = np.unique(endo_stack[:, 3])
		endo = [None]*time_pts.size
		epi = [None]*time_pts.size
		
		# Format endo and epi contour points
		for i, time_pt in enumerate(time_pts):
			endo_by_time = endo_stack[np.where(endo_stack[:, 3] == time_pt)[0], :]
			epi_by_time = epi_stack[np.where(epi_stack[:, 3] == time_pt)[0], :]
			endo_time_by_slice = [None]*len(kept_slices)
			epi_time_by_slice = [None]*len(kept_slices)
			for j, slice_num in enumerate(kept_slices):
				endo_time_by_slice[j] = endo_by_time[np.where(endo_by_time[:, 4] == slice_num)[0], :3]
				endo_slice_column = endo_by_time[np.where(endo_by_time[:, 4] == slice_num)[0], 4]
				epi_time_by_slice[j] = epi_by_time[np.where(epi_by_time[:, 4] == slice_num)[0], :3]
				epi_slice_column = epi_by_time[np.where(epi_by_time[:, 4] == slice_num)[0], 4]
				endo_time_by_slice[j] = np.column_stack((endo_time_by_slice[j], endo_slice_column))
				epi_time_by_slice[j] = np.column_stack((epi_time_by_slice[j], epi_slice_column))
			endo[i] = endo_time_by_slice
			epi[i] = epi_time_by_slice

		cine_endo_rotate = [None]*len(endo)
		cine_epi_rotate = [None]*len(epi)
		
		# Rotate endo and epicardial points and reformat the rotated points.
		for time_pt in range(len(cine_endo_rotate)):
			endo_rotate_timepts = [None]*len(endo[time_pt])
			epi_rotate_timepts = [None]*len(epi[time_pt])
			for slice_num in range(len(endo_rotate_timepts)):
				endo_rotate_timepts[slice_num], _, self.transform_basis, _ = stackhelper.rotateDataCoordinates(endo[time_pt][slice_num][:, :3], apex_pt, base_pt, center_septal_pt)
				epi_rotate_timepts[slice_num], _, self.transform_basis, _ = stackhelper.rotateDataCoordinates(epi[time_pt][slice_num][:, :3], apex_pt, base_pt, center_septal_pt)
				endo_rotate_timepts[slice_num] = np.column_stack((endo_rotate_timepts[slice_num], endo[time_pt][slice_num][:, 3]))
				epi_rotate_timepts[slice_num] = np.column_stack((epi_rotate_timepts[slice_num], epi[time_pt][slice_num][:, 3]))
			cine_endo_rotate[time_pt] = endo_rotate_timepts
			cine_epi_rotate[time_pt] = epi_rotate_timepts
		self.rv_insertion_pts_rot = stackhelper.rotateDataCoordinates(self.rv_insertion_pts, apex_pt, base_pt, center_septal_pt)[0]
		self.abs_pts_rot = stackhelper.rotateDataCoordinates(self.apex_base_pts, apex_pt, base_pt, center_septal_pt)[0]

		cine_endo_arrs = [None]*len(endo)
		cine_epi_arrs = [None]*len(epi)
		cine_endo_rotate_arrs = [None]*len(cine_endo_rotate)
		cine_epi_rotate_arrs = [None]*len(cine_epi_rotate)
		
		# Establish list of arrays to format
		for time_ind, endo_timept in enumerate(endo):
			epi_timept = epi[time_ind]
			endo_rotate_timept = cine_endo_rotate[time_ind]
			epi_rotate_timept = cine_epi_rotate[time_ind]
			cine_endo_arrs[time_ind] = np.vstack(endo_timept)
			cine_endo_rotate_arrs[time_ind] = np.vstack(endo_rotate_timept)
			cine_epi_arrs[time_ind] = np.vstack(epi_timept)
			cine_epi_rotate_arrs[time_ind] = np.vstack(epi_rotate_timept)

		# Store class fields based on calculated values:
		self.cine_endo_rotate = cine_endo_rotate_arrs
		self.cine_epi_rotate = cine_epi_rotate_arrs
		self.cine_endo = cine_endo_arrs
		self.cine_epi = cine_epi_arrs
		self.cine_apex_pt = self.abs_pts_rot[0]
		self.cine_basal_pt = self.abs_pts_rot[1]
		self.cine_septal_pts = self.rv_insertion_pts_rot
		
		return(True)
		
	def importLGE(self):
		"""Import the LGE MRI stack.
		
		Create an endocardial and epicardial contour based on the LGE file stack.
		Additionally imports scar traces and stores them as scar ratio.
		Scar ratio is the ratio of the scar contour edges compared to wall thickness.
			
		Returns:
			boolean: True if the import was successful.
		"""
		scar_endo_stack, scar_epi_stack, scar_insertion_pts, scarstruct, scar_septal_slice = importhelper.importStack(self.sa_scar_file)
		
		# Prepare variables imported from file.
		scar_auto = np.array(scarstruct['Scar']['Auto'])
		scar_manual = np.array(scarstruct['Scar']['Manual'])
		
		# Form a combination array that combines the automatic scar recognition with manual adjustments (including manual erasing)
		scar_combined = np.add(scar_auto, scar_manual) > 0
		
		# The new array needs to have axes adjusted to align with the format (z, x, y) allowing list[n] to return a full slice
		scar_combined = np.swapaxes(np.swapaxes(scar_combined, 1, 2), 0, 1)
		scar_slices = np.where([np.any(scar_combined[slice_num, :, :]) for slice_num in range(scar_combined.shape[0])])[0]
		self.scar_combined = scar_combined
		
		scar_x, scar_y = stackhelper.getMaskXY(scar_combined, scarstruct['KeptSlices'])
		
		scarstruct['mask_x'] = scar_x
		scarstruct['mask_y'] = scar_y
		
		scar_pt_stack, scar_m = stackhelper.rotateStack(scarstruct, scar_slices+1, layer='mask')

		apex_pt = self.apex_base_pts[0, :]
		base_pt = self.apex_base_pts[1, :]
		center_septal_pt = np.expand_dims(self.rv_insertion_pts[2, :], 0)
		
		scar_endo = [None]*len(scar_slices)
		scar_epi = [None]*len(scar_slices)
		scar_pts = [None]*len(scar_slices)
		self.lge_endo_rotate = [None]*len(scar_endo)
		self.lge_epi_rotate = [None]*len(scar_epi)
		self.lge_pts_rotate = [None]*len(scar_pts)
		
		for i, slice_num in enumerate(scar_slices):
			scar_endo[i] = scar_endo_stack[np.where(scar_endo_stack[:, 4] == slice_num+1)[0], :3]
			scar_epi[i] = scar_epi_stack[np.where(scar_epi_stack[:, 4] == slice_num+1)[0], :3]
			scar_pts[i] = scar_pt_stack[np.where(scar_pt_stack[:, 4] == slice_num+1)[0], :3]
			self.lge_endo_rotate[i], _, self.transform_basis, self.origin = stackhelper.rotateDataCoordinates(scar_endo[i], apex_pt, base_pt, center_septal_pt)
			self.lge_epi_rotate[i], _, self.transform_basis, _ = stackhelper.rotateDataCoordinates(scar_epi[i], apex_pt, base_pt, center_septal_pt)
			self.lge_pts_rotate[i], _, _, _ = stackhelper.rotateDataCoordinates(scar_pts[i], apex_pt, base_pt, center_septal_pt)
		
		# Store instance fields
		self.lge_septal_pts = scar_insertion_pts
		self.lge_endo = scar_endo
		self.lge_epi = scar_epi
		self.scar_slices = scar_slices
		
		return(True)
		
	def importScarLA(self):
		"""Import Long-Axis LGE Images and contours.
		"""
		# Set up import variables based on number of long-axis files.
		scar_la_endo = [None]*len(self.la_scar_files)
		scar_la_epi = [None]*len(self.la_scar_files)
		scar_la_pinpts = [None]*len(self.la_scar_files)
		scar_la_struct = [None]*len(self.la_scar_files)
		scar_pt_stack = [None]*len(self.la_scar_files)

		for i, la_scar_file in enumerate(self.la_scar_files):
			# Import the long-axis-based LGE information
			scar_la_endo[i], scar_la_epi[i], scar_la_pinpts[i], scar_la_struct[i] = importhelper.importStack(la_scar_file, ignore_pinpts=True)
			
			# Define scar combination
			scar_auto = np.array(scar_la_struct[i]['Scar']['Auto'])
			scar_manual = np.array(scar_la_struct[i]['Scar']['Manual'])
			scar_combined_full = np.expand_dims(np.add(scar_auto, scar_manual) > 0, axis=0)
			
			scar_x, scar_y = stackhelper.getMaskXY(scar_combined_full, [0])
			
			scar_la_struct[i]['mask_x'] = scar_x
			scar_la_struct[i]['mask_y'] = scar_y
			
			scar_pt_stack[i], _ = stackhelper.rotateStack(scar_la_struct[i], [1], layer='mask')
		
		self.lge_la_endo_rotate = [None]*len(self.la_scar_files)
		self.lge_la_epi_rotate = [None]*len(self.la_scar_files)
		self.lge_la_pts_rotate = [None]*len(self.la_scar_files)
		
		apex_pt = self.apex_base_pts[0, :]
		base_pt = self.apex_base_pts[1, :]
		center_septal_pt = np.expand_dims(self.rv_insertion_pts[2, :], 0)
		
		# Rotate and save long-axis lge contours
		for slice_num in range(len(self.la_scar_files)):
			scar_endo = scar_la_endo[slice_num][:, :3]
			scar_epi = scar_la_epi[slice_num][:, :3]
			scar_pts = scar_pt_stack[slice_num][:, :3]
			
			self.lge_la_endo_rotate[slice_num] = stackhelper.rotateDataCoordinates(scar_endo, apex_pt, base_pt, center_septal_pt)[0]
			self.lge_la_epi_rotate[slice_num] = stackhelper.rotateDataCoordinates(scar_epi, apex_pt, base_pt, center_septal_pt)[0]
			self.lge_la_pts_rotate[slice_num] = stackhelper.rotateDataCoordinates(scar_pts, apex_pt, base_pt, center_septal_pt)[0]
		
	def importDense(self):
		"""Imports DENSE MR data from the file established at initialization.
		"""
		
		dense_endo = [None]*len(self.dense_file)
		dense_epi = [None]*len(self.dense_file)
		dense_pts = [None]*len(self.dense_file)
		slice_locations = [None]*len(self.dense_file)
		dense_displacement = False
		radial_strain = False
		circumferential_strain = False
		
		for i in range(len(self.dense_file)):
			dense_file = self.dense_file[i]
			# Extract Contour Information from DENSE Mat file
			dense_data = importhelper.loadmat(dense_file)
			slice_location = dense_data['SequenceInfo'][0, 0].SliceLocation
			slice_locations[i] = slice_location
			epi_dense = np.array(dense_data['ROIInfo']['RestingContour'][0])
			endo_dense = np.array(dense_data['ROIInfo']['RestingContour'][1])
			
			# Append slice location to DENSE contour data
			endo_slice_col = np.array([slice_location] * endo_dense.shape[0]).reshape([endo_dense.shape[0], 1])
			epi_slice_col = np.array([slice_location] * epi_dense.shape[0]).reshape([epi_dense.shape[0], 1])
			endo_dense = np.append(endo_dense, endo_slice_col, axis=1)
			epi_dense = np.append(epi_dense, epi_slice_col, axis=1)
			
			# Interpolate data to get an equal number of points for each contour
			endo_interp_func = sp.interpolate.interp1d(np.arange(0, 1+1/(endo_dense.shape[0]-1), 1/(endo_dense.shape[0]-1)), endo_dense, axis=0, kind='cubic')
			epi_interp_func = sp.interpolate.interp1d(np.arange(0, 1+1/(epi_dense.shape[0]-1), 1/(epi_dense.shape[0]-1)), epi_dense, axis=0, kind='cubic')
			endo_interp = endo_interp_func(np.arange(0, 80/79, 1/79))
			epi_interp = epi_interp_func(np.arange(0, 80/79, 1/79))
			
			# Pull timepoints from DENSE
			dense_timepoints = len(dense_data['DisplacementInfo']['dX'][0])
			
			# Shift the DENSE endo and epi contours by the epicardial mean
			endo_shift = endo_interp[:, :2] - np.mean(epi_interp[:, :2], axis=0)
			epi_shift = epi_interp[:, :2] - np.mean(epi_interp[:, :2], axis=0)
			endo_shift_theta, endo_shift_rho = mathhelper.cart2pol(endo_shift[:, 0], endo_shift[:, 1])
			epi_shift_theta, epi_shift_rho = mathhelper.cart2pol(epi_shift[:, 0], epi_shift[:, 1])
			dense_endo[i] = endo_shift
			dense_epi[i] = epi_shift
			
			# Shift the entire pixel array by the same epicardial mean
			dense_x = dense_data['DisplacementInfo']['X'] - np.mean(epi_interp[:, 0])
			dense_y = dense_data['DisplacementInfo']['Y'] - np.mean(epi_interp[:, 1])
			dense_z = [slice_location]*len(dense_x)
			
			dense_pts[i] = np.column_stack((dense_x, dense_y, dense_z))
			
			all_dense_theta, all_dense_rho = mathhelper.cart2pol(dense_x, dense_y)
			
			# Get displacement / strain info and store as a 2-D array
			dense_dx = np.array(dense_data['DisplacementInfo']['dX'])
			dense_dy = np.array(dense_data['DisplacementInfo']['dY'])
			dense_dz = np.array(dense_data['DisplacementInfo']['dZ'])
			dense_radial = np.array(dense_data['StrainInfo']['RR'])
			dense_circumferential = np.array(dense_data['StrainInfo']['CC'])
			
			# Add DENSE displacement and strain slices by time
			if not dense_displacement:
				dense_displacement = [None] * dense_dx.shape[1]
				radial_strain = [None] * dense_radial.shape[1]
				circumferential_strain = [None]*dense_circumferential.shape[1]
				for i in range(dense_dx.shape[1]):
					dense_displacement[i] = np.column_stack((dense_dx[:, i], dense_dy[:, i], dense_dz[:, i]))
				for i in range(dense_radial.shape[1]):
					radial_strain[i] = dense_radial[:, i]
				for i in range(dense_circumferential.shape[1]):
					circumferential_strain[i] = dense_circumferential[:, i]
			else:
				for i in range(dense_dx.shape[1]):
					cur_disp = np.column_stack((dense_dx[:, i], dense_dy[:, i], dense_dz[:, i]))
					dense_displacement[i] = [dense_displacement[i], cur_disp]
				for i in range(dense_radial.shape[1]):
					cur_rad_strain = dense_radial[:, i]
					radial_strain[i] = [radial_strain[i], cur_rad_strain]
				for i in range(dense_circumferential.shape[1]):
					cur_circ_strain = dense_circumferential[:, i]
					circumferential_strain[i] = [circumferential_strain[i], cur_circ_strain]
		self.dense_endo = dense_endo
		self.dense_epi = dense_epi
		self.dense_pts = dense_pts
		self.dense_displacement = dense_displacement
		self.dense_slices = slice_locations
		self.radial_strain = radial_strain
		self.circumferential_strain = circumferential_strain
		return(True)
	
	def alignScar(self, timepoint=0):
		"""Scar alignment designed to include long-axis scar data.
		"""
		# Interpolate the scar contours circumferentially using 50 evenly-spaced angle bins
		self.interp_epi_surf, self.wall_scar = stackhelper.interpShortScar(50, self.lge_epi_prol, self.lge_endo_prol, self.lge_pts_prol, self.lge_epi_rotate, self.lge_endo_rotate, self.lge_pts_rotate)
		# Remove nan values from the list of scar transmuralities
		for slice_num in range(len(self.wall_scar)):
			temp_slice = self.wall_scar[slice_num]
			temp_slice[np.isnan(temp_slice[:, 1]), 1] = 0
			self.wall_scar[slice_num] = temp_slice
		# Interpolate scar contours based on the long-axis LGE contours
		self.interp_epi_la_surf, self.wall_scar_la = stackhelper.interpLongScar(20, self.lge_la_epi_prol, self.lge_la_endo_prol, self.lge_la_pts_prol, self.lge_la_epi_rotate, self.lge_la_endo_rotate, self.lge_la_pts_rotate)
		# Remove nan values from the list of long-axis LGE transmuralities
		for slice_num in range(len(self.wall_scar_la)):
			temp_slice = self.wall_scar_la[slice_num]
			temp_slice[np.isnan(temp_slice[:, 1]), 1] = 0
			self.wall_scar_la[slice_num] = temp_slice
		temp_data_arr = np.empty([0, 6])
		# Stack the slices into single arrays
		for slice_num in range(len(self.wall_scar)):
			temp_slice_arr = np.column_stack((self.interp_epi_surf[slice_num], self.wall_scar[slice_num]))
			temp_data_arr = np.vstack((temp_data_arr, temp_slice_arr))
		for slice_num in range(len(self.wall_scar_la)):
			temp_slice_arr = np.column_stack((self.interp_epi_la_surf[slice_num], self.wall_scar_la[slice_num]))
			temp_data_arr = np.vstack((temp_data_arr, temp_slice_arr))
		nan_rows = ~np.isnan(temp_data_arr[:, 1])
		interp_data = temp_data_arr[nan_rows, :]
		interp_data = interp_data[:, [2, 1, 4, 5]]
		interp_data_inc = np.column_stack((interp_data[:, 0]+2*math.pi, interp_data[:, 1:]))
		interp_data_dec = np.column_stack((interp_data[:, 0]-2*math.pi, interp_data[:, 1:]))
		interp_data_complete = np.vstack((interp_data, interp_data_inc, interp_data_dec))
		self.interp_data = interp_data_complete
	
	def convertDataProlate(self, focus):
		"""Convert all data from a rotated axis into prolate spheroid coordinates for further alignment.
		"""
		# Convert cine endocardial and epicardial traces
		cine_endo_prol = [None]*len(self.cine_endo_rotate)
		cine_epi_prol = [None]*len(self.cine_epi_rotate)
		
		for time_pt in range(len(self.cine_endo_rotate)):
			cine_endo_rot_time = self.cine_endo_rotate[time_pt]
			cine_epi_rot_time = self.cine_epi_rotate[time_pt]
			cine_endo_prol[time_pt] = np.column_stack(tuple(mathhelper.cart2prolate(cine_endo_rot_time[:, 0], cine_endo_rot_time[:, 1], cine_endo_rot_time[:, 2], focus)))
			cine_epi_prol[time_pt] = np.column_stack(tuple(mathhelper.cart2prolate(cine_epi_rot_time[:, 0], cine_epi_rot_time[:, 1], cine_epi_rot_time[:, 2], focus)))
		
		self.cine_endo_prol = cine_endo_prol
		self.cine_epi_prol = cine_epi_prol

		# Convert Short-Axis LGE endocardial, epicardial, and scar-point traces
		lge_endo_prol = [None]*len(self.lge_endo_rotate)
		lge_epi_prol = [None]*len(self.lge_epi_rotate)
		lge_pts_prol = [None]*len(self.lge_pts_rotate)
		
		for i in range(len(self.lge_endo_rotate)):
			lge_endo_prol_list = mathhelper.cart2prolate(self.lge_endo_rotate[i][:, 0], self.lge_endo_rotate[i][:, 1], self.lge_endo_rotate[i][:, 2], focus)
			lge_epi_prol_list = mathhelper.cart2prolate(self.lge_epi_rotate[i][:, 0], self.lge_epi_rotate[i][:, 1], self.lge_epi_rotate[i][:, 2], focus)
			lge_pts_prol_list = mathhelper.cart2prolate(self.lge_pts_rotate[i][:, 0], self.lge_pts_rotate[i][:, 1], self.lge_pts_rotate[i][:, 2], focus)
			
			lge_endo_prol[i] = np.column_stack(tuple(lge_endo_prol_list))
			lge_epi_prol[i] = np.column_stack(tuple(lge_epi_prol_list))
			lge_pts_prol[i] = np.column_stack(tuple(lge_pts_prol_list))
			
		self.lge_endo_prol = lge_endo_prol
		self.lge_epi_prol = lge_epi_prol
		self.lge_pts_prol = lge_pts_prol
		
		# Convert long-axis LGE endocardial, epicardial, and scar-point traces
		lge_la_endo_prol = [None]*len(self.lge_la_endo_rotate)
		lge_la_epi_prol = [None]*len(self.lge_la_epi_rotate)
		lge_la_pts_prol = [None]*len(self.lge_la_pts_rotate)
		
		for i in range(len(self.lge_la_endo_rotate)):
			lge_la_endo_prol_list = mathhelper.cart2prolate(self.lge_la_endo_rotate[i][:, 0], self.lge_la_endo_rotate[i][:, 1], self.lge_la_endo_rotate[i][:, 2], focus)
			lge_la_epi_prol_list = mathhelper.cart2prolate(self.lge_la_epi_rotate[i][:, 0], self.lge_la_epi_rotate[i][:, 1], self.lge_la_epi_rotate[i][:, 2], focus)
			lge_la_pts_prol_list = mathhelper.cart2prolate(self.lge_la_pts_rotate[i][:, 0], self.lge_la_pts_rotate[i][:, 1], self.lge_la_pts_rotate[i][:, 2], focus)
			
			lge_la_endo_prol[i] = np.column_stack(tuple(lge_la_endo_prol_list))
			lge_la_epi_prol[i] = np.column_stack(tuple(lge_la_epi_prol_list))
			lge_la_pts_prol[i] = np.column_stack(tuple(lge_la_pts_prol_list))
			
		self.lge_la_endo_prol = lge_la_endo_prol
		self.lge_la_epi_prol = lge_la_epi_prol
		self.lge_la_pts_prol = lge_la_pts_prol
	
	def alignScarCine(self, timepoint=0):
		"""A method of aligning scar and cine data
		"""
		# If a timepoint is passed, pull the cine from that point
		cine_endo = self.cine_endo[timepoint]
		cine_epi = self.cine_epi[timepoint]
		# Get slice values to section the endo / epi array by slice
		slice_indices = sorted(np.unique(cine_endo[:, 2], return_index=True)[1])
		slice_vals = cine_endo[slice_indices, 2]
		# Set up angle bins
		num_bins = self.scar_ratio.shape[1] + 1
		angles = np.linspace(-math.pi, math.pi, num_bins)
		angles2 = angles[1:]
		angles2 = np.append(angles2, angles[0])
		angles = np.column_stack((angles, angles2))[:-1]
		# Iterate through slices and convert to polar
		full_scar_contour = []
		for i in range(len(slice_vals)):
			# Get indices for the current slice
			cur_slice_ind = np.where(cine_endo[:, 2] == slice_vals[i])[0]
			# Pull current slice endocardial and epicardial cartesian contours
			cur_slice_endo = cine_endo[cur_slice_ind, :]
			cur_slice_epi = cine_epi[cur_slice_ind, :]
			
			# Get the slice center and shift by that value (center slices at 0)
			slice_center = np.mean(cur_slice_epi, axis=0)
			endo_x = cur_slice_endo[:, 0] - slice_center[0]
			endo_y = cur_slice_endo[:, 1] - slice_center[1]
			epi_x = cur_slice_epi[:, 0] - slice_center[0]
			epi_y = cur_slice_epi[:, 1] - slice_center[1]
			
			# Convert the cartesian contours to polar
			endo_theta, endo_rho = mathhelper.cart2pol(endo_x, endo_y)
			epi_theta, epi_rho = mathhelper.cart2pol(epi_x, epi_y)
			endo_theta = [cur_slice_theta_i - 2*np.pi if cur_slice_theta_i > np.pi else cur_slice_theta_i for cur_slice_theta_i in endo_theta]
			epi_theta = [cur_slice_theta_i - 2*np.pi if cur_slice_theta_i > np.pi else cur_slice_theta_i for cur_slice_theta_i in epi_theta]
			# Get rho values for each angle bin based on theta
			endo_bin_inds = [np.where((endo_theta > angles[i, 0]) & (endo_theta <= angles[i, 1]))[0].tolist() for i in range(angles.shape[0])]
			epi_bin_inds = [np.where((epi_theta > angles[i, 0]) & (epi_theta <= angles[i, 1]))[0].tolist() for i in range(angles.shape[0])]
			endo_rho_mean = [np.mean(endo_rho[endo_bin_inds_i]) for endo_bin_inds_i in endo_bin_inds]
			epi_rho_mean = [np.mean(epi_rho[epi_bin_inds_i]) for epi_bin_inds_i in epi_bin_inds]
			
			# Get the current scar slice
			cur_scar = self.scar_ratio[i, :, :]
			
			# Adjust any values less than 0 or greater than 1 in the ratio
			with np.errstate(invalid='ignore'):
				nonan_inds = np.where(~np.isnan(cur_scar[:, 1]))[0].tolist()
				for j in range(len(nonan_inds)):
					# Check if the value is less than 0
					if cur_scar[nonan_inds[j], 1] <= 0:
						# Set it equal to the average of the adjacent values (to create a smooth scar trace)
						if j == 0:
							cur_scar[nonan_inds[j], 1] = np.mean([cur_scar[nonan_inds[j], 2], cur_scar[nonan_inds[j+1], 1]])
						elif j == len(nonan_inds) - 1:
							cur_scar[nonan_inds[j], 1] = np.mean([cur_scar[nonan_inds[j-1], 1], cur_scar[nonan_inds[j], 2]])
						else:
							cur_scar[nonan_inds[j], 1] = np.mean([cur_scar[nonan_inds[j-1], 1], cur_scar[nonan_inds[(j+1) % (len(nonan_inds)-1)], 1]])
					# Check if the value is greater than 1
					if cur_scar[nonan_inds[j], 2] >= 1:
						# Set it equal to the average of the adjacent values (to create a smooth scar trace)
						if j == 0:
							cur_scar[nonan_inds[j], 2] = np.mean([cur_scar[nonan_inds[j], 1], cur_scar[nonan_inds[j+1], 2]])
						elif j == len(nonan_inds) - 1:
							cur_scar[nonan_inds[j], 2] = np.mean([cur_scar[nonan_inds[j-1], 2], cur_scar[nonan_inds[j], 1]])
						else:
							cur_scar[nonan_inds[j], 2] = np.mean([cur_scar[nonan_inds[j-1], 2], cur_scar[nonan_inds[(j+1) % (len(nonan_inds)-1)], 2]])
			
			# Get the scar inner and outer rho values based on endo and epi rho values
			scar_inner_rho = [endo_rho_mean[j] + cur_scar[j, 1] * (epi_rho_mean[j] - endo_rho_mean[j]) for j in range(cur_scar.shape[0])]
			scar_outer_rho = [endo_rho_mean[j] + cur_scar[j, 2] * (epi_rho_mean[j] - endo_rho_mean[j]) for j in range(cur_scar.shape[0])]
			# Convert the scar values to cartesian
			scar_inner_x, scar_inner_y = mathhelper.pol2cart(cur_scar[:, 0], scar_inner_rho)
			scar_outer_x, scar_outer_y = mathhelper.pol2cart(cur_scar[:, 0], scar_outer_rho)
			# If there is no scar trace here, just move to the next slice and append an empty array for the current contour
			if np.all(np.isnan(scar_inner_x)):
				full_scar_contour.append(np.array([]))
				continue
			
			# Extract the contour and remove NaN values
			# Roll the array so that the first point of the contour is the non-nan value immediately after nan
			nan_boundary = np.where([np.isnan(scar_inner_x[j]) & ~np.isnan(scar_inner_x[j+1]) for j in range(scar_inner_x.size - 1)])[0]
			scar_inner_x = np.roll(scar_inner_x, -nan_boundary[0]-1)
			# Find the end of the contour and slice, removing NaNs
			num_boundary = np.where([~np.isnan(scar_inner_x[j]) & np.isnan(scar_inner_x[j+1]) for j in range(scar_inner_x.size - 1)])[0]
			scar_inner_x = scar_inner_x[:num_boundary[0]+1]
			# Roll each other contour
			scar_outer_x = np.roll(scar_outer_x, -nan_boundary[0]-1)[:num_boundary[0]+1]
			scar_inner_y = np.roll(scar_inner_y, -nan_boundary[0]-1)[:num_boundary[0]+1]
			scar_outer_y = np.roll(scar_outer_y, -nan_boundary[0]-1)[:num_boundary[0]+1]
			# Construct combined arrays in trace order, inner -> outer in a loop
			scar_x = np.append(scar_inner_x, scar_outer_x[::-1])
			scar_y = np.append(scar_inner_y, scar_outer_y[::-1])
			# Re-shift based on center of slice
			scar_x += slice_center[0]
			scar_y += slice_center[1]
			# Stack the x, y, and z values into a slice of the full contour
			full_scar_contour.append(np.column_stack((scar_x, scar_y, [slice_vals[i]]*scar_x.size)))
		
		# Get equal number of points per slice
		for i in range(len(full_scar_contour)):
			scar_layer = full_scar_contour[i]
			if scar_layer.size == 0: continue
			scar_xy_pts = scar_layer[:, :2]
			scar_range_pts = np.linspace(0, 1, scar_xy_pts.shape[0])
			interp_func = sp.interpolate.interp1d(scar_range_pts, scar_xy_pts, kind='cubic', axis=0)
			new_scar_numpts = np.linspace(0, 1, 80)
			new_xy_pts = interp_func(new_scar_numpts)
			new_xy_pts = np.column_stack((new_xy_pts, [scar_layer[0, 2]] * 80))
			full_scar_contour[i] = new_xy_pts
		
		# Interpolate additional scar slices
		interp_scar_contour = []
		for i in range(len(full_scar_contour) - 1):
			# Get scar slices to use for interpolation
			scar_layer = full_scar_contour[i]
			scar_adj_layer = full_scar_contour[i+1]
			# If the layers are edge layers, or non-scar, don't bother
			if scar_layer.size == 0 or scar_adj_layer.size == 0: continue
			# Use the two slices as the edge values for the interpolation function
			interp_arr = [0, 1]
			x_interp_func = sp.interpolate.interp1d(interp_arr, np.column_stack((scar_layer[:, 0], scar_adj_layer[:, 0])))
			y_interp_func = sp.interpolate.interp1d(interp_arr, np.column_stack((scar_layer[:, 1], scar_adj_layer[:, 1])))
			z_interp_func = sp.interpolate.interp1d(interp_arr, np.column_stack((scar_layer[:, 2], scar_adj_layer[:, 2])))
			# Interpolate a number of intermediate slices
			new_interp_arr = np.linspace(0, 1, 5)
			new_x_vals = x_interp_func(new_interp_arr)
			new_y_vals = y_interp_func(new_interp_arr)
			new_z_vals = z_interp_func(new_interp_arr)
			# Reconstruct each slice back together, removing duplicate slices
			for j in range(new_x_vals.shape[1]):
				cur_slice = np.column_stack((new_x_vals[:, j], new_y_vals[:, j], new_z_vals[:, j]))
				slice_stored = any((np.around(cur_slice, 5) == np.around(slice_arr, 5)).all() for slice_arr in interp_scar_contour)
				if not slice_stored: interp_scar_contour.append(np.column_stack((new_x_vals[:, j], new_y_vals[:, j], new_z_vals[:, j])))
		full_scar_contour = interp_scar_contour
		# Append the contour to the total aligned scar data (essentially tracks timepoints)
		if len(self.aligned_scar) == 0:
			self.aligned_scar = [None] * len(self.cine_endo)
		self.aligned_scar[timepoint] = full_scar_contour
		return(full_scar_contour)

	def alignDense(self, cine_timepoint=0):
		"""Align DENSE data to a cine slice by selected timepoint.
		"""
		# Get data at specified timepoints.
		try:
			cine_endo_timepoint = self.cine_endo[cine_timepoint]
			cine_epi_timepoint = self.cine_epi[cine_timepoint]
		except:
			print("Invalid timepoint selected. Try again.")
			return(False)

		slice_array = [self.endo_slices[i - 1] for i in cine_endo_timepoint[:, 3].astype(int)]
		dense_aligned_pts = [False]*len(self.dense_pts)
		scaled_strain = [False]*len(self.dense_displacement)
		
		# Iterate through DENSE slices
		for slice_num in range(len(self.dense_endo)):
			# Extract slice-based values
			dense_slice_endo = self.dense_endo[slice_num]
			dense_slice_epi = self.dense_epi[slice_num]
			dense_slice_pts = self.dense_pts[slice_num]
			cur_slice = self.dense_slices[slice_num]
			slice_in_cine = np.where(round(cur_slice, 1) == np.round(slice_array, 1))[0]
			
			# Get cine contours matching DENSE slice
			cine_endo_slice = cine_endo_timepoint[slice_in_cine, :] - np.mean(cine_epi_timepoint[slice_in_cine, :], axis=0)
			cine_epi_slice = cine_epi_timepoint[slice_in_cine, :] - np.mean(cine_epi_timepoint[slice_in_cine, :], axis=0)
			
			# Convert both sets of slices to polar
			dense_endo_theta, dense_endo_rho = mathhelper.cart2pol(dense_slice_endo[:, 0], dense_slice_endo[:, 1])
			dense_epi_theta, dense_epi_rho = mathhelper.cart2pol(dense_slice_epi[:, 0], dense_slice_epi[:, 1])
			cine_endo_theta, cine_endo_rho = mathhelper.cart2pol(cine_endo_slice[:, 0], cine_endo_slice[:, 1])
			cine_epi_theta, cine_epi_rho = mathhelper.cart2pol(cine_epi_slice[:, 0], cine_epi_slice[:, 1])
			
			# Generate interpolation equations to ensure order and number of points is the same
			theta_interp_pts = np.linspace(0, 2*math.pi, 100)[:-1]
			
			dense_endo_eq = sp.interpolate.interp1d(dense_endo_theta, dense_endo_rho, fill_value="extrapolate")
			dense_epi_eq = sp.interpolate.interp1d(dense_epi_theta, dense_epi_rho, fill_value="extrapolate")
			cine_endo_eq = sp.interpolate.interp1d(cine_endo_theta, cine_endo_rho, fill_value="extrapolate")
			cine_epi_eq = sp.interpolate.interp1d(cine_epi_theta, cine_epi_rho, fill_value="extrapolate")
			
			# Get interpolated rho values at new theta points
			dense_endo_interp_rho = dense_endo_eq(theta_interp_pts)
			dense_epi_interp_rho = dense_epi_eq(theta_interp_pts)
			cine_endo_interp_rho = cine_endo_eq(theta_interp_pts)
			cine_epi_interp_rho = cine_epi_eq(theta_interp_pts)
			
			# Get endo and epi values together, in order, since transform must be universal
			dense_interp_vals = np.append(np.column_stack((theta_interp_pts, dense_endo_interp_rho)), np.column_stack((theta_interp_pts, dense_epi_interp_rho)), axis=0)
			cine_interp_vals = np.append(np.column_stack((theta_interp_pts, cine_endo_interp_rho)), np.column_stack((theta_interp_pts, cine_epi_interp_rho)), axis=0)
			
			# Convert back to x and y to get a 2d transform field
			dense_interp_x, dense_interp_y = mathhelper.pol2cart(dense_interp_vals[:, 0], dense_interp_vals[:, 1])
			cine_interp_x, cine_interp_y = mathhelper.pol2cart(cine_interp_vals[:, 0], cine_interp_vals[:, 1])
			
			# Calculate the linear distance in x and y between 
			x_dist = cine_interp_x - dense_interp_x
			y_dist = cine_interp_y - dense_interp_y
			
			# Pull slice of points
			dense_pts = self.dense_pts[slice_num]
			
			# Interpolate new grid points
			dense_pts_new_x = dense_pts[:, 0] + sp.interpolate.griddata(np.column_stack((dense_interp_x, dense_interp_y)), x_dist, dense_pts[:, :2], method='cubic')
			dense_pts_new_y = dense_pts[:, 1] + sp.interpolate.griddata(np.column_stack((dense_interp_x, dense_interp_y)), y_dist, dense_pts[:, :2], method='cubic')
			
			# Get points outside of the hull and use nearest-neighbor to calculate positional change
			isnan_x = np.where(np.isnan(dense_pts_new_x))[0]
			isnan_y = np.where(np.isnan(dense_pts_new_y))[0]
			
			dense_pts_new_x[isnan_x] = dense_pts[isnan_x, 0] + sp.interpolate.griddata(np.column_stack((dense_interp_x, dense_interp_y)), x_dist, dense_pts[isnan_x, :2], method='nearest')
			dense_pts_new_y[isnan_y] = dense_pts[isnan_y, 1] + sp.interpolate.griddata(np.column_stack((dense_interp_x, dense_interp_y)), y_dist, dense_pts[isnan_y, :2], method='nearest')
			
			# Get the difference in scale between the old and new positions to scale strain
			scale_diff_x = (np.max(dense_pts_new_x) - np.min(dense_pts_new_x)) / (np.max(dense_pts[:, 0]) - np.min(dense_pts[:, 0]))
			scale_diff_y = (np.max(dense_pts_new_y) - np.min(dense_pts_new_y)) / (np.max(dense_pts[:, 1]) - np.min(dense_pts[:, 1]))
			
			# Scale strain
			for time in range(len(self.dense_displacement)):
				dense_time_disp = self.dense_displacement[time] if not isinstance(self.dense_displacement[time], list) else self.dense_displacement[time][slice_num]
				dense_time_x_scale = dense_time_disp[:, 0] * scale_diff_x
				dense_time_y_scale = dense_time_disp[:, 1] * scale_diff_y
				if not scaled_strain[time]:
					scaled_strain[time] = [np.column_stack((dense_time_x_scale, dense_time_y_scale))]
				else:
					scaled_strain[time].append(np.column_stack((dense_time_x_scale, dense_time_y_scale)))
			
			# Store loop variables
			dense_aligned_pts[slice_num] = np.column_stack((dense_pts_new_x, dense_pts_new_y))
			
		# Set globals upon loop completion
		self.dense_aligned_displacement = scaled_strain
		self.dense_aligned_pts = dense_aligned_pts
			
		return(True)