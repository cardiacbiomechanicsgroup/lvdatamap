# -*- coding: utf-8 -*-
"""
Contains information necessary for import and alignment of confocal microscopy images.

Created on Mon Jan 27 12:22:15 2018

@author: cdw2be

Planar Image Brightness Correction based on: https://imagej.nih.gov/ij/plugins/plane-brightness/2010_Michalek_Biosignal.pdf
"""

# Imports
import math
import tkinter as tk
from tkinter import filedialog
import scipy as sp
import numpy as np
import matplotlib.pyplot as mplt
import glob
from PIL import Image
import os
from natsort import natsorted, ns
from cardiachelpers import importhelper
from cardiachelpers import confocalhelper

class ConfocalModel():
	"""Model class to hold confocal microscopy images and format them to generate a mesh to align with MRI data.
	"""

	def __init__(self, top_dir, slice_gap = 10):
		"""Initialize the model made to import confocal microscopy data.
		
		args:
			confocal_dir (string): The path to the directory containing tif image files
		"""
		
		# Get all TIFF files in the directory
		dirs = natsorted([(top_dir + '/' + sub_dir) for sub_dir in os.listdir(top_dir) if os.path.isdir(top_dir + '/' + sub_dir)], alg=ns.IC)
		self.top_dir = top_dir
		self.slices = [ConfocalSlice(im_dir) for im_dir in dirs]
		self.slice_names = [confocal_slice.slice_name for confocal_slice in self.slices]
		self.model_image = None
		self.slice_image_list = None
		self.endo_contours = None
		self.epi_contours = None
	
	def generateStitchedImages(self, slices, sub_slices, overlap=0.1, compress_ratio=0.25, force_file=False):
		"""Adjust image intensity based on edge intensity of adacent images.
		"""
		# Set subslices list to mid-slice if none is indicated.
		for slice_ind, slice_num in enumerate(slices):
			# Determine if the sub_slices list is per-slice, or a general list.
			cur_sub_slices = sub_slices[slice_ind] if isinstance(sub_slices[slice_ind], list) else sub_slices
			for sub_slice in cur_sub_slices:
				file_name = self.top_dir + '/' + self.slice_names[slice_num] + 'Frame' + str(sub_slice) + 'Stitched.tif'
				if (not os.path.isfile(file_name)) or force_file:
					self.slices[slice_num].createStitchedImage(overlap=overlap, compress_ratio=compress_ratio, frame=sub_slice, stitched_file=file_name, force_file=force_file)
				else:	
					print('Image already exists! No need to overwrite.')
		
	def generateImageGridFiles(self, slices):
		for slice_ind, slice_num in enumerate(slices):
			self.slices[slice_num].createImageGridFile()
		
	def getSubsliceList(self, top_slice):
		"""Gather information about which sub-slices are present within each image.
		"""
		return(list(range(self.slices[top_slice].num_slices)))
		
	def getChannelList(self, top_slice):
		"""Gather information about which channels are present within each image.
		"""
		return(list(self.slices[top_slice].channels))
	
	def importModelImage(self, model_image_file):
		self.model_image = model_image_file
		self.slice_image_list = confocalhelper.openModelImage(model_image_file)
	
	def generateContours(self):
		self.endo_contours = [None]*len(self.slice_image_list)
		self.epi_contours = [None]*len(self.slice_image_list)
		for im_num, im_slice in enumerate(self.slice_image_list):
			edge_image = confocalhelper.contourMaskImage(im_slice)
			endo_path, epi_path, labelled_arr = confocalhelper.splitImageObjects(edge_image)
			self.endo_contours[im_num] = confocalhelper.orderPathTrace(endo_path)
			self.epi_contours[im_num] = confocalhelper.orderPathTrace(epi_path)
	
class ConfocalSlice():
	"""Class to hold information for a single biological slice of tissue.
	
	Biological slices are stored in individual directories.
	"""
	def __init__(self, confocal_dir):
		# Store the directory of images used for this model and parse out the folder name.
		self.confocal_dir = confocal_dir
		self.slice_name = os.path.split(confocal_dir)[1]
		
		# Record filenames for all tiff image files in the directory.
		self.tif_files = natsorted(glob.glob(os.path.join(confocal_dir, '*.tif')).copy(), alg=ns.IC)
		self.raw_images = [None]*len(self.tif_files)
		
		# Iterate through and create image objects for each tiff file
		for file_num, image_file in enumerate(self.tif_files):
			self.raw_images[file_num] = Image.open(image_file)
		
		# Define how many sub-slices (confocal slices) are contained within the overall slice
		self.num_slices = self.raw_images[0].n_frames
		
		# Define the channels within the image
		self.channels = self.raw_images[0].getbands()
		
		# Set up a list to hold compressed versions of the images, formatted as slices at top level
		#		This is due to compression only working on a single slice
		self.compressed_images = [None]*self.num_slices
		
		# Assign a file to contain positional information for the images for stitching purposes
		#		This file may or may not exist already in the directory, but is either referenced or created during stitching
		self.image_grid_file = self.confocal_dir + '/stitch_grid.txt'
	
	def createStitchedImage(self, overlap=0.1, compress_ratio=0.25, frame=0, stitched_file=False, force_file=False):
		"""Stitch images together in a grid.
		"""
		# Either read or create the image grid file
		if os.path.isfile(self.image_grid_file):
			self.im_grid = confocalhelper.readImageGrid(self.image_grid_file)
		else:
			# If the image grid file does not exist, create it for future use.
			self.im_locs, im_locs_dict = confocalhelper.getImagePositions(self.tif_files)
			self.im_grid = confocalhelper.getImageGrid(self.tif_files, self.im_locs, im_locs_dict)
			confocalhelper.writeImageGrid(self.im_grid, self.image_grid_file)
		
		# Define a name for the file to use to store the stitched image
		if not stitched_file:
			stitched_file = self.confocal_dir + '/StitchedImages/' + self.slice_name + 'slice' + str(frame) + 'chan' + str(channel) + '.tif'
		if os.path.isfile(stitched_file) and not force_file:
			# If the file exists and overwriting isn't forced, don't bother stitching
			print('File exists! No need to overwrite.')
			return(True)
		
		# If there is nothing in the compressed images (i.e. it hasn't already been compressed), instantiate it
		if not self.compressed_images[frame]:
			self.compressed_images[frame] = [None]*len(self.raw_images)
		
		# Iterate through the images to generate compressed versions
		for image_num, raw_image in enumerate(self.raw_images):
			# Pull the desired frame
			image_frame = confocalhelper.splitImageFrames(raw_image)[frame] if frame < raw_image.n_frames else confocalhelper.splitImageFrames(raw_image)[0]
			# If being told to compress the images, compress to desired ratio
			compressed_frame = confocalhelper.compressImages(image_frame, image_scale=compress_ratio)
			self.compressed_images[frame][image_num] = compressed_frame

		# Pass the compressed images, image grid information, and stitched file to save to the image stitching function
		stitched_success = confocalhelper.stitchImages(self.compressed_images[frame], self.im_grid[:, 0], self.im_grid[:, 1], save_pos=stitched_file, stitched_type='F')
		
	def createImageGridFile(self):
		im_locs, im_locs_dict = confocalhelper.getImagePositions(self.tif_files)
		im_grid = confocalhelper.getImageGrid(self.tif_files, im_locs, im_locs_dict)
		confocalhelper.writeImageGrid(im_grid, self.image_grid_file)