# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 12:48:42 2017

@author: cdw2be
"""
import warnings
warnings.simplefilter('ignore', UserWarning)
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import font
from tkinter import messagebox
import mrimodel
import confocalmodel
import mesh
import numpy as np
from cardiachelpers import displayhelper
import math
import winreg, glob, os
warnings.simplefilter('default', UserWarning)

class modelGUI(tk.Frame):
	"""Generates a GUI to control the Python-based cardiac modeling toolbox.
	"""
	
	def __init__(self, master=None):
		tk.Frame.__init__(self, master)
		self.grid()
		self.identifyPostView()
		self.createWidgets()
		self.scar_assign = False
		self.dense_assign = False
		self.mri_model = False
		self.mri_mesh = False
		
		# Set row and column paddings
		master.rowconfigure(0, pad=5)
		master.rowconfigure(1, pad=5)
		master.rowconfigure(2, pad=5)
		master.rowconfigure(3, pad=5)
		master.rowconfigure(4, pad=5)
		master.rowconfigure(5, pad=5)
		master.rowconfigure(6, pad=5)
		master.rowconfigure(7, pad=5)
		master.columnconfigure(8, pad=5)
		master.columnconfigure(9, pad=5)
		
		# On window close by user
		master.protocol('WM_DELETE_WINDOW', self.destroy())
		
		self.master = master

	def identifyPostView(self):
		try:
			postview_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\University of Utah\\PostView")
			postview_folder = winreg.QueryValueEx(postview_key, "Location")[0]
			os.chdir(postview_folder)
			postview_exe_file = glob.glob('P*.exe')[0]
			self.postview_exe = os.path.join(postview_folder, postview_exe_file)
		except FileNotFoundError:
			self.postview_exe = ''
		
	def createWidgets(self):
		"""Place widgets throughout GUI frame and assign functionality.
		"""
		# Establish tk variables
		sa_filename = tk.StringVar()
		la_filename = tk.StringVar()
		lge_filename = tk.StringVar()
		la_lge_filenames = tk.StringVar()
		dense_filenames = tk.StringVar()
		premade_mesh_filename = tk.StringVar()
		confocal_dir = tk.StringVar()
		self.scar_plot_bool = tk.IntVar(value=0)
		self.dense_plot_bool = tk.IntVar(value=0)
		self.nodes_plot_bool = tk.IntVar(value=0)

		# Import Settings
		#	Place labels
		import_label = ttk.Label(text='Model Import Options and Settings')
		import_label.grid(row=0, column=0, columnspan=9)
		ttk.Label(text='Short-Axis File:').grid(row=1, sticky='W')
		ttk.Label(text='Long-Axis File:').grid(row=2, sticky='W')
		ttk.Label(text='SA LGE File:').grid(row=3, sticky='W')
		ttk.Label(text='LA LGE Files:').grid(row=4, sticky='W')
		ttk.Label(text='DENSE Files:').grid(row=5, sticky='W')
		ttk.Label(text='Premade Mesh File:').grid(row=6, sticky='W')
		ttk.Label(text='Confocal Directory:').grid(row=7, sticky='W')
		
		# 	Create entry objects
		sa_file_entry = ttk.Entry(width=80, textvariable=sa_filename)
		la_file_entry = ttk.Entry(width=80, textvariable=la_filename)
		lge_file_entry = ttk.Entry(width=80, textvariable=lge_filename)
		la_lge_file_entry = ttk.Entry(width=80, textvariable=la_lge_filenames)
		dense_file_entry = ttk.Entry(width=80, textvariable=dense_filenames)
		confocal_dir_entry = ttk.Entry(width=80, textvariable=confocal_dir)
		premade_mesh_entry = ttk.Entry(width=80, textvariable=premade_mesh_filename)
		# 	Place entry object
		sa_file_entry.grid(row=1, column=1, columnspan=5)
		la_file_entry.grid(row=2, column=1, columnspan=5)
		lge_file_entry.grid(row=3, column=1, columnspan=5)
		la_lge_file_entry.grid(row=4, column=1, columnspan=5)
		dense_file_entry.grid(row=5, column=1, columnspan=5)
		premade_mesh_entry.grid(row=6, column=1, columnspan=5)
		confocal_dir_entry.grid(row=7, column=1, columnspan=5)
		
		#	Place "Browse" buttons
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(sa_file_entry)).grid(row=1, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(la_file_entry)).grid(row=2, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(lge_file_entry)).grid(row=3, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(la_lge_file_entry, multi='True')).grid(row=4, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(dense_file_entry, multi='True')).grid(row=5, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(premade_mesh_entry)).grid(row=6, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(confocal_dir_entry, multi='Dir')).grid(row=7, column=6)
		
		# Model Options
		#	Place labels
		ttk.Label(text='Primary Cine Timepoint:').grid(row=1, column=7, sticky='W')
		#	Create options comboboxes
		self.cine_timepoint_cbox = ttk.Combobox(state='disabled', width=5)
		self.cine_timepoint_cbox.bind('<<ComboboxSelected>>', lambda _ : self.cineTimeChanged())
		self.cine_timepoint_cbox.grid(row=1, column=8)
		#	Buttons to generate models
		ttk.Button(text='Generate MRI Model', command= lambda: self.createMRIModel(sa_filename, la_filename, lge_filename, la_lge_filenames, dense_filenames)).grid(row=2, column=7, columnspan=2)
		
		
		# Confocal Model Options
		#	Place labels
		#	Create options entries
		self.confocal_slice_button = ttk.Menubutton(text='Select confocal slices', state='disabled')
		self.confocal_slice_menu = tk.Menu(self.confocal_slice_button, tearoff=False)
		self.confocal_slice_button.configure(menu=self.confocal_slice_menu)
		self.confocal_slice_button.grid(row=4, column=7, columnspan=2)
		#	Buttons to generate models
		ttk.Button(text='Generate Confocal Model', command= lambda: self.createConfocalModel(confocal_dir_entry)).grid(row=3, column=7, columnspan=2)
		ttk.Button(text='Stitch Selected Slices', command= lambda: self.startStitching()).grid(row=5, column=7, columnspan=2)
		
		# Mesh Options / Creation
		#	Place labels
		mesh_label = ttk.Label(text='Mesh Settings')
		mesh_label.grid(row=0, column=10, columnspan=2)
		ttk.Label(text='Number of Rings:').grid(row=1, column=10, sticky='W')
		ttk.Label(text='Elements per Ring:').grid(row=2, column=10, sticky='W')
		ttk.Label(text='Elements through Wall:').grid(row=3, column=10, sticky='W')
		ttk.Label(text='Mesh Type:').grid(row=4, column=10, sticky='W')
		ttk.Label(text='Select conn matrix:').grid(row=5, column=10, sticky='W')
		#	Create mesh option entry boxes
		num_rings_entry = ttk.Entry(width=10)
		elem_per_ring_entry = ttk.Entry(width=10)
		elem_thru_wall_entry = ttk.Entry(width=10)
		mesh_type_cbox = ttk.Combobox(values=['4x2', '4x4', '4x8'], state='readonly', width=10)
		self.conn_mat_cbox = ttk.Combobox(state='disabled', values=['hex', 'pent'], width=10)
		#	Place mesh option entry boxes
		num_rings_entry.grid(row=1, column=11, sticky='W')
		elem_per_ring_entry.grid(row=2, column=11, sticky='W')
		elem_thru_wall_entry.grid(row=3, column=11, sticky='W')
		mesh_type_cbox.grid(row=4, column=11, sticky='W')
		self.conn_mat_cbox.grid(row=5, column=11, sticky='W')
		#	Mesh option entry boxes default text and input validation
		num_rings_entry.insert(0, '28')
		elem_per_ring_entry.insert(0, '48')
		elem_thru_wall_entry.insert(0, '5')
		num_rings_entry.configure(validate='key', validatecommand=(num_rings_entry.register(self.intValidate), '%P'))
		elem_per_ring_entry.configure(validate='key', validatecommand=(num_rings_entry.register(self.intValidate), '%P'))
		elem_thru_wall_entry.configure(validate='key', validatecommand=(num_rings_entry.register(self.intValidate), '%P'))
		mesh_type_cbox.current(2)
		self.conn_mat_cbox.current(0)
		#	Create mesh option buttons
		self.premadeMeshButton = ttk.Button(text='Use Premade Mesh', state='enabled', command= lambda: self.createMRIMesh(num_rings_entry, elem_per_ring_entry, elem_thru_wall_entry, premade_mesh_file=premade_mesh_filename))
		self.meshButton = ttk.Button(text='Generate Model First', state='disabled', command= lambda: self.createMRIMesh(num_rings_entry, elem_per_ring_entry, elem_thru_wall_entry, mesh_type_cbox=mesh_type_cbox))
		self.scar_fe_button = ttk.Button(text='Identify scar nodes', state='disabled', command= lambda: self.scarElem())
		self.dense_fe_button = ttk.Button(text='Assign element displacements', state='disabled', command= lambda: self.denseElem())
		self.scar_dense_button = ttk.Button(text='Get scar region DENSE average', state='disabled', command= lambda: self.scarDense())
		#	Place mesh option buttons
		self.premadeMeshButton.grid(row=8, column=10, columnspan=2)
		self.meshButton.grid(row=10, column=10, columnspan=2)
		self.scar_fe_button.grid(row=11, column=10, columnspan=2)
		self.dense_fe_button.grid(row=12, column=10, columnspan=2)
		self.scar_dense_button.grid(row=13, column=10, columnspan=2)
		
		# FEBio File Creation
		#	Place labels
		postview_label = ttk.Label(text='Postview Options')
		postview_label.grid(row=10, column=2, columnspan=7)
		ttk.Label(text='Postview filename:').grid(row=11, column=2, sticky='W')
		ttk.Label(text='Postview installation:').grid(row=12, column=2, sticky='W')
		#	Create entry objects
		self.postview_file_entry = ttk.Entry()
		self.postview_exe_entry = ttk.Entry()
		self.postview_file_entry.grid(row=11, column=3, columnspan=5, sticky='WE')
		self.postview_exe_entry.grid(row=12, column=3, columnspan=5, sticky='WE')
		self.postview_exe_entry.insert(0, self.postview_exe)
		#	Buttons to create and open files
		self.feb_file_button = ttk.Button(text='Generate FEBio File', state='disabled', command= lambda: self.genFebFile())
		if self.postview_exe == '':
			self.postview_open_button = ttk.Button(text='Launch PostView', state='disabled', command= lambda: self.openPostview())
		else:
			self.postview_open_button = ttk.Button(text='Launch PostView', state='enabled', command= lambda: self.openPostview())
		self.feb_file_button.grid(row=13, column=3)
		self.postview_open_button.grid(row=13, column=4)
		#	Create "Browse" button
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(self.postview_file_entry, multi='Feb')).grid(row=11, column=8, sticky='W')
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(self.postview_exe_entry, multi='Exe')).grid(row=12, column=8, sticky='W')
		
		# Plot Options
		#	Place labels
		ttk.Label(text='Plot nodes in mesh?').grid(row=10, column=0, sticky='W')
		ttk.Label(text='Plot Scar?').grid(row=11, column=0, sticky='W')
		ttk.Label(text='Plot DENSE?').grid(row=12, column=0, sticky='W')
		ttk.Label(text='DENSE Timepoint:').grid(row=13, column=0, sticky='W')
		#	DENSE Timepoint combobox
		self.dense_timepoint_cbox = ttk.Combobox(state='disabled', width=5)
		self.dense_timepoint_cbox.grid(row=13, column=1, sticky='W')
		#	Options checkboxes
		self.scar_cbutton = ttk.Checkbutton(variable = self.scar_plot_bool, state='disabled')
		self.dense_cbutton = ttk.Checkbutton(variable = self.dense_plot_bool, state='disabled')
		self.nodes_cbutton = ttk.Checkbutton(variable = self.nodes_plot_bool, state='disabled')
		self.scar_cbutton.grid(row=11, column=1, sticky='W')
		self.dense_cbutton.grid(row=12, column=1, sticky='W')
		self.nodes_cbutton.grid(row=10, column=1, sticky='W')
		#	Buttons to plot MRI Models or Meshes
		self.plot_mri_button = ttk.Button(text='Plot MRI Model', command= lambda: self.plotMRIModel(), state='disabled')
		self.plot_mesh_button = ttk.Button(text='Plot MRI Mesh', command= lambda: self.plotMRIMesh(), state='disabled')
		self.plot_mri_button.grid(row=14, column=0, sticky='W')
		self.plot_mesh_button.grid(row=14, column=1, sticky='W')
		
		# Separators
		ttk.Separator(orient='vertical').grid(column=9, row=0, rowspan=15, sticky='NS')
		ttk.Separator(orient='horizontal').grid(column=0, row=9, columnspan=9, sticky='EW')
		ttk.Separator(orient='vertical').grid(row=9, column=1, rowspan=6, sticky='NSE')
		
		# Set specific label fonts
		f = font.Font(mesh_label, mesh_label.cget('font'))
		f.configure(underline=True)
		mesh_label.configure(font=f)
		import_label.configure(font=f)
		postview_label.configure(font=f)

	def openFileBrowser(self, entry_box, multi='False'):
		"""Open a file browser window and assign the file name to the passed entry box.
		Allows various options for type of file browser to be launched.
		"""
		if multi == 'True':
			file_name = filedialog.askopenfilenames(title='Select Files')
		elif multi == 'Dir':
			file_name = filedialog.askdirectory(title='Select Folder')
		elif multi == 'Feb':
			file_name = filedialog.asksaveasfilename(title='Save File', filetypes=(('FEBio files','*.feb'), ('All files', '*')))
			if not (file_name.split('.')[-1] == 'feb'):
				file_name += '.feb'
		else:
			file_name = filedialog.askopenfilename(title='Select File')
		if not file_name == '':
			entry_box.delete(0, 'end')
			entry_box.insert(0, file_name)
		return(file_name)
		
	def createMRIModel(self, sa_filename, la_filename, lge_filename, la_lge_filenames, dense_filenames):
		"""Function run to instantiate MRI model based on input files.
		"""
		# Check that required files are present
		if sa_filename.get() == '' or la_filename.get() == '':
			if sa_filename.get() == '':
				messagebox.showinfo('File Error', 'Need Short-Axis file.')
			elif la_filename.get() == '':
				messagebox.showinfo('File Error', 'Need Long-Axis file.')
			return(False)
		
		# Parse DENSE Filenames
		if not(dense_filenames.get() == ''):
			dense_filenames_replaced = list(self.master.tk.splitlist(dense_filenames.get()))
		else:
			dense_filenames_replaced = dense_filenames.get()
			
		if not(la_lge_filenames.get() == ''):
			la_lge_filenames_replaced = list(self.master.tk.splitlist(la_lge_filenames.get()))
		else:
			la_lge_filenames_replaced = la_lge_filenames.get()
		
		# Instantiate MRI model object and import cine stack (at default timepoint)
		self.mri_model = mrimodel.MRIModel(sa_filename.get(), la_filename.get(), sa_scar_file=lge_filename.get(), la_scar_files=la_lge_filenames_replaced, dense_file=dense_filenames_replaced)
		self.mri_model.importCine(timepoint=0)
		
		# Import LGE, if included, and generate full alignment array
		if self.mri_model.scar:
			self.scar_cbutton.configure(state='normal')
			self.mri_model.importLGE()
			if not(la_lge_filenames_replaced == ''):
				self.mri_model.importScarLA()
			self.mri_model.convertDataProlate()
			interp_scar = self.mri_model.alignScar()
		else:
			# Only occurs if scar is removed from MRI model on later instantiation
			self.scar_cbutton.configure(state='disabled')
			self.scar_fe_button.configure(state='disabled')
		
		# Import DENSE, if included
		if self.mri_model.dense:
			self.dense_cbutton.configure(state='normal')
			self.mri_model.importDense()
		else:
			# Only occurs if DENSE is removed from MRI model on later instantiation
			self.dense_cbutton.configure(state='disabled')
			self.dense_timepoint_cbox.configure(values=[], state='disabled')
			self.dense_fe_button.configure(state='disabled')
		self.mri_model.convertDataProlate()
		if self.mri_model.dense:
			self.mri_model.alignDense(cine_timepoint=0)
			self.dense_timepoint_cbox.configure(values=list(range(len(self.mri_model.circumferential_strain))), state='readonly')
			self.dense_timepoint_cbox.current(0)
		
		# Update GUI elements
		self.cine_timepoint_cbox.configure(values=list(range(len(self.mri_model.cine_endo))), state='readonly')
		self.cine_timepoint_cbox.current(0)
		if not self.mri_mesh:
			self.meshButton.configure(state='normal', text='Generate MRI Mesh')
		else:
			self.mri_mesh.assignInsertionPts(self.mri_model.cine_apex_pt, self.mri_model.cine_basal_pt, self.mri_model.cine_septal_pts)
			if self.mri_model.scar:
				self.scar_fe_button.configure(state='enabled')
			if self.mri_model.dense:
				self.dense_fe_button.configure(state='enabled')
		
	def createMRIMesh(self, num_rings_entry, elem_per_ring_entry, elem_thru_wall_entry, mesh_type_cbox=False, premade_mesh_file=False):
		"""Function to generate base-level mesh from MRI model object
		"""
		# Pull variables from GUI entry fields
		if not (num_rings_entry.get() == '' or elem_per_ring_entry.get() == '' or elem_thru_wall_entry.get() == ''):
			num_rings = int(num_rings_entry.get())
			elem_per_ring = int(elem_per_ring_entry.get())
			elem_in_wall = int(elem_thru_wall_entry.get())
		else:
			messagebox.showinfo('Mesh Settings', 'Mesh option left blank. Correct and try again.')
			return(False)
			
		if mesh_type_cbox:
			time_point = int(self.cine_timepoint_cbox.get())
		
			# Create base mesh
			self.mri_mesh = mesh.Mesh(num_rings, elem_per_ring, elem_in_wall)
		
			# Fit mesh to MRI model data
			#temp_view = displayhelper.visPlot3d(self.mri_model.cine_endo_rotate[time_point][:, :3])
			#displayhelper.visPlot3d(self.mri_model.cine_epi_rotate[time_point][:, :3], view=temp_view)
			self.mri_mesh.fitContours(self.mri_model.cine_endo_rotate[time_point][:, :3], self.mri_model.cine_epi_rotate[time_point][:, :3], self.mri_model.cine_apex_pt, self.mri_model.cine_basal_pt, self.mri_model.cine_septal_pts, mesh_type_cbox.get())
			self.mri_mesh.feMeshRender()
			self.mri_mesh.nodeNum(self.mri_mesh.meshCart[0], self.mri_mesh.meshCart[1], self.mri_mesh.meshCart[2])
			self.mri_mesh.getElemConMatrix()
			# Update GUI elements as needed
			self.plot_mri_button.configure(state='normal')
			self.plot_mesh_button.configure(state='normal')
			self.feb_file_button.configure(state='normal')
			self.nodes_cbutton.configure(state='normal')
			self.conn_mat_cbox.configure(state='readonly')
			if self.mri_model.scar:
				self.scar_fe_button.configure(state='normal')
			else:
				self.scar_fe_button.configure(state='disabled')
			if self.mri_model.dense:
				self.dense_fe_button.configure(state='normal')
			else:
				self.dense_fe_button.configure(state='disabled')
		elif premade_mesh_file:
			self.mri_mesh = mesh.Mesh(num_rings, elem_per_ring, elem_in_wall)
			import_success = self.mri_mesh.importPremadeMesh(premade_mesh_file.get())
			# Update GUI elements as needed
			if import_success:
				self.plot_mri_button.configure(state='normal')
				self.plot_mesh_button.configure(state='normal')
				self.feb_file_button.configure(state='normal')
				self.nodes_cbutton.configure(state='normal')
				self.meshButton.configure(state='disabled', text='Using Premade Mesh')
				if self.mri_model:
					self.mri_mesh.assignInsertionPts(self.mri_model.cine_apex_pt, self.mri_model.cine_basal_pt, self.mri_model.cine_septal_pts)
					if self.mri_model.scar:
						self.scar_fe_button.configure(state='normal')
					else:
						self.scar_fe_button.configure(state='disabled')
					if self.mri_model.dense:
						self.dense_fe_button.configure(state='normal')
					else:
						self.dense_fe_button.configure(state='disabled')
			else:
				self.mri_mesh = None
				messagebox.showwarning('No Mesh File', 'Mesh file not found.')
	
	def createConfocalModel(self, confocal_dir_entry):
		"""Set up a new confocalModel object of stitched images.
		
		Models contain ConfocalSlice objects, which are defined within the confocalModel file.
		"""
		# Establish the directory of interest in the model.
		confocal_dir = confocal_dir_entry.get()
		if confocal_dir == '':
			messagebox.showinfo('No Directory', 'Please select a directory for confocal images.')
			return(False)
		# Instantiate a new model object
		self.confocal_model = confocalmodel.ConfocalModel(confocal_dir)
		self.confocal_slice_selections = {}
		# Create a local variable to track slices in the confocalmodel object.
		for slice_name in self.confocal_model.slice_names:
			self.confocal_slice_selections[slice_name] = tk.IntVar(value=1)
			self.confocal_slice_menu.add_checkbutton(label=slice_name, variable=self.confocal_slice_selections[slice_name], onvalue=1, offvalue=0)
		# Enable GUI elements that require an instantiated confocalModel object.
		self.confocal_slice_button.configure(state='enabled')
	
	def cineTimeChanged(self):
		"""Function to respond to timepoint adjustments in the base cine mesh / model
		"""
		# Insure timepoint is an integer (should always succeed, in place for possible errors)
		try:
			new_timepoint = int(self.cine_timepoint_cbox.get())
		except:
			return(False)
		# Import the cine model at the selected timepoint (updates landmarks)
		self.mri_model.importCine(timepoint = new_timepoint)
		# If necessary, align DENSE to new cine timepoint (most important aspect)
		if self.mri_model.dense:
			self.mri_model.alignDense(cine_timepoint = new_timepoint)
	
	def plotMRIModel(self):
		"""Plots MRI Model based on raw data (slice contours, scar traces, etc.)
		"""
		# Pull timepoint from timepoint selection combobox
		time_point = int(self.cine_timepoint_cbox.get())
		# Plot overall cine segmentation data
		mri_axes = displayhelper.segmentRender(self.mri_model.cine_endo_rotate[time_point], self.mri_model.cine_epi_rotate[time_point], self.mri_model.cine_apex_pt, self.mri_model.cine_basal_pt, self.mri_model.cine_septal_pts, self.mri_mesh.origin, self.mri_mesh.transform)
		# If desired, plot scar data
		if self.scar_plot_bool.get() and self.mri_model.scar:
			mri_axes = displayhelper.displayScarTrace(self.mri_model.interp_scar_trace, self.mri_mesh.origin, self.mri_mesh.transform, ax=mri_axes)
			if hasattr(self.mri_model, 'interp_scar_la_trace'):
				mri_axes = displayhelper.displayScarTrace(self.mri_model.interp_scar_la_trace, self.mri_mesh.origin, self.mri_mesh.transform, ax=mri_axes)
		# If desired, plot DENSE data
		if self.dense_plot_bool.get() and self.mri_model.dense:
			mri_axes = displayhelper.displayDensePts(self.mri_model.dense_aligned_pts, self.mri_model.dense_slice_shifted, self.mri_mesh.origin, self.mri_mesh.transform, self.mri_model.dense_aligned_displacement, dense_plot_quiver=1, timepoint=int(self.dense_timepoint_cbox.get()), ax=mri_axes)
	
	def plotMRIMesh(self):
		"""Plots the mesh data as a surface plot, with display options
		"""
		# Plot surface contours of endocardium and epicardium
		mesh_axes = displayhelper.surfaceRender(self.mri_mesh.endo_node_matrix, self.mri_mesh.focus)
		mesh_axes = displayhelper.surfaceRender(self.mri_mesh.epi_node_matrix, self.mri_mesh.focus, ax=mesh_axes)
		# Display node positions, if selected
		if self.nodes_plot_bool.get():
			mesh_axes = displayhelper.nodeRender(self.mri_mesh.nodes, ax=mesh_axes)
		# Display scar locations, if available and desired
		if self.scar_plot_bool.get() and self.mri_mesh.nodes_in_scar.size:
			mesh_axes = displayhelper.nodeRender(self.mri_mesh.nodes[self.mri_mesh.nodes_in_scar, :], ax=mesh_axes)
		elif self.scar_plot_bool.get() and not self.mri_mesh.nodes_in_scar.size:
			# Warn if scar box selected, but elements unidentified.
			messagebox.showinfo('Warning', 'Identify scar nodes before plotting to view.')
	
	def scarElem(self):
		"""Requests mesh to process which elements are in scar
		"""
		time_point = int(self.cine_timepoint_cbox.get())
		self.mri_model.convertDataProlate(self.mri_mesh.focus)
		self.mri_mesh.rotateNodesProlate()
		self.mri_model.alignScar()
		self.mri_mesh.interpScarData(self.mri_model.interp_scar, trans_smooth=1, depth_smooth=0.5)
		if not self.scar_assign:
			self.scar_assign = True
		if self.dense_assign and self.scar_assign:
			self.scar_dense_button.configure(state='normal')
	
	def denseElem(self):
		"""Requests mesh to assign DENSE information to all applicable elements
		"""
		time_point = int(self.cine_timepoint_cbox.get())
		self.mri_mesh.assignDenseElems(self.mri_model.dense_aligned_pts, self.mri_model.dense_slices, self.mri_model.dense_aligned_displacement, self.mri_model.radial_strain, self.mri_model.circumferential_strain)
		if not self.dense_assign:
			self.dense_assign = True
		if self.dense_assign and self.scar_assign:
			self.scar_dense_button.configure(state='normal')
	
	def scarDense(self):
		"""Function designed to calculate regional DENSE data based on model / mesh scar extent.
		
		Calculates DENSE data in the identified scar and non-scar (remote) regions and reports / plots those values.
		"""
		scar_average_dense = [None]*len(self.mri_model.dense_aligned_displacement)
		remote_average_dense = [None]*len(self.mri_model.dense_aligned_displacement)
		for time_point in range(len(self.mri_model.dense_aligned_displacement)):
			scar_average_dense[time_point] = self.mri_mesh.getElemData(self.mri_mesh.elems_in_scar, 'dense', timepoint=time_point).tolist()
			remote_average_dense[time_point] = self.mri_mesh.getElemData(self.mri_mesh.elems_out_scar, 'dense', timepoint=time_point).tolist()
		scar_radial = [scar_dense[0] for scar_dense in scar_average_dense]
		scar_circ = [scar_dense[1] for scar_dense in scar_average_dense]
		remote_radial = [remote_dense[0] for remote_dense in remote_average_dense]
		remote_circ = [remote_dense[1] for remote_dense in remote_average_dense]
		displayhelper.plotListData([scar_radial, remote_radial], ['Scar', 'Remote'])
		displayhelper.plotListData([scar_circ, remote_circ], ['Scar', 'Remote'])
	
	def genFebFile(self):
		"""Generate FEBio file in indicated location
		"""
		# Perform filename checks to ensure proper filename entered.
		feb_file_name = self.postview_file_entry.get()
		if feb_file_name == '':
			messagebox.showinfo('Filename Error', 'File name is not indicated.')
			return(False)
		elif not (feb_file_name.split('.')[-1] == 'feb'):
			messagebox.showinfo('Filename Error', 'File must be an FEBio file (*.feb).')
			return(False)
		# Generate FEBio file through Mesh function
		self.mri_mesh.generateFEFile(feb_file_name, self.conn_mat_cbox.get())
		# Update GUI Elements
		self.postview_open_button.configure(state='normal')
		
	def openPostview(self):
		"""Launch a PostView instance pointed at the FEBio File
		
		Note for this function to operate as expected, the function displayhelper/displayMeshPostview must point to the FEBio PostView executable on your machine.
		"""
		# Pull FEBio file name
		feb_file_name = self.postview_file_entry.get()
		feb_executable = self.postview_exe_entry.get()
		# Check that file is an accessible file
		try:
			open(feb_file_name)
		except:
			messagebox.showinfo('File Warning', 'FEBio File not found. Check file name and try again.')
			return(False)
		# Ensure that file is an FEBio file
		if not (feb_file_name.split('.')[-1] == 'feb'):
			messagebox.showinfo('File Warning', 'File selected is not an FEBio file. Check file name and try again.')
			return(False)
		# Request PostView Launch
		try:
			open(feb_executable)
			displayhelper.displayMeshPostview(feb_file_name, feb_executable)
		except FileNotFoundError:
			messagebox.showinfo('Executable Warning', 'PostView installation not found. Please identify executable location.')
			return(False)
		except OSError:
			messagebox.showinfo('Executable Warning', 'Invalid file selected. Please identify correct executable location.')
			return(False)
	
	def startStitching(self):
		"""Stitch selected slices into a large, combined image and save.
		"""
		stitch_slices = []
		for slice_name, stitch_var in self.confocal_slice_selections.items():
			if stitch_var.get():
				stitch_slices.append(self.confocal_model.slice_names.index(slice_name))
		# Get subslice list	
		sub_slices = [None]*len(stitch_slices)
		for slice_num, cur_slice in enumerate(stitch_slices):
			sub_slices[slice_num] = self.confocal_model.getSubsliceList(cur_slice)
		# Get channel list
		channels = [None]*len(stitch_slices)
		for slice_num, cur_slice in enumerate(stitch_slices):
			channels[slice_num] = self.confocal_model.getChannelList(cur_slice)
		self._createSubsliceWindow(stitch_slices, sub_slices, channels)
		return(True)
	
	def intValidate(self, new_value):
		"""Simple validation function to ensure an entry receives only int-able inputs or null
		"""
		# Accept empty entry box to allow clearing the box
		if new_value == '':
			return(True)
		# Attempt integer conversion. If possible, accept new input
		try:
			int(new_value)
			return(True)
		except:
			return(False)
	
	def _createSubsliceWindow(self, slice_list, subslice_list, channel_list):
		"""Create a window to select subslices and channels.
		"""
		# Calculate window height
		root = tk.Tk()
		screen_height = root.winfo_screenheight()
		root.destroy()
		
		self.slice_menu = tk.Toplevel(self.master)
		self.slice_menu.wm_title('Select Subslices and Channels')
		
		# Set up canvas to allow scrollable window
		slice_canvas = tk.Canvas(self.slice_menu, borderwidth=0)
		slice_frame = tk.Frame(slice_canvas)
		scroll_bar = tk.Scrollbar(self.slice_menu, orient="vertical", command=slice_canvas.yview)
		slice_canvas.configure(yscrollcommand=scroll_bar.set)
		
		scroll_bar.pack(side="right", fill="y")
		slice_canvas.pack(side="left", fill="both", expand=True)
		slice_canvas.create_window((4,4), window=slice_frame, anchor="nw")
		
		def onFrameConfigure(canvas):
			canvas.configure(scrollregion=canvas.bbox("all"))
		
		slice_canvas.bind_all("<MouseWheel>", lambda event: slice_canvas.yview_scroll(int(-1*(event.delta/40)), "units"))
		slice_frame.bind("<Configure>", lambda event, canvas=slice_canvas: onFrameConfigure(slice_canvas))
		
		self.subslice_selections = {}
		
		for slice_num, slice_index in enumerate(slice_list):
			slice_frame.columnconfigure(slice_num, pad=10)
			ttk.Label(slice_frame, text=self.confocal_model.slice_names[slice_index]).grid(row=0, column=slice_num)
			for sub_num, sub_slice in enumerate(subslice_list[slice_num]):
				cur_string = self.confocal_model.slice_names[slice_index] + " Frame " + str(sub_slice)
				self.subslice_selections[cur_string] = tk.IntVar(value=1)
				ttk.Checkbutton(slice_frame, text="Frame " + str(sub_slice), variable=self.subslice_selections[cur_string]).grid(row=sub_num+1, column=slice_num)
		
		self.channel_selections = {}
		farthest_column, lowest_row = slice_frame.grid_size()
		# Place Channel List
		for slice_num, slice_index in enumerate(slice_list):
			ttk.Label(slice_frame, text='Channels').grid(row=lowest_row, column=slice_num)
			for channel_num, channel in enumerate(channel_list[slice_num]):
				cur_string = self.confocal_model.slice_names[slice_index] + ' ' + channel
				self.channel_selections[cur_string] = tk.IntVar(value=1)
				ttk.Checkbutton(slice_frame, text=channel, variable=self.channel_selections[cur_string]).grid(row=lowest_row+channel_num+1, column=slice_num)
				
		farthest_column, lowest_row = slice_frame.grid_size()
		ttk.Button(slice_frame, text='Generate Stitched Image', command= lambda: self._stitchSlices(slice_list)).grid(row=lowest_row, column=math.ceil(farthest_column/2)-1, columnspan=2-(farthest_column % 2))
		ttk.Button(slice_frame, text='Generate Stitch Files', command= lambda: self._generateStitchFiles(slice_list)).grid(row=lowest_row+1, column=math.ceil(farthest_column/2)-1, columnspan=2-(farthest_column % 2))
		
		# Update and resize the canvas to match the frame
		slice_frame.update()
		slice_canvas.update()
		
		frame_height = np.min([int(screen_height * 3 / 4), slice_frame.winfo_height()])
		
		slice_canvas.config(width = slice_frame.winfo_width(), height=frame_height)
		
	def _stitchSlices(self, slice_list):
		"""Actually iterate through and run the stitching process for each item selected by the user.
		"""
		subslice_list, channel_list = self.__getSubsChannels(slice_list)
		self.confocal_model.generateStitchedImages(slice_list, subslice_list, compress_ratio=1)
	
	def _generateStitchFiles(self, slice_list):
		self.confocal_model.generateImageGridFiles(slice_list)
	
	def __getSubsChannels(self, slice_list):
		"""Get the subslices and channels based on the selections made in the slice selection window.
		"""
		sub_slices = [None]*len(slice_list)
		slice_chans = [None]*len(slice_list)
		for slice_num, slice_index in enumerate(slice_list):
			subslice_list = self.confocal_model.getSubsliceList(slice_index)
			channel_list = self.confocal_model.getChannelList(slice_index)
			subslice_selected = [False]*len(subslice_list)
			channel_selected = [False]*len(channel_list)
			for sub_num, sub_slice in enumerate(subslice_list):
				subslice_string = self.confocal_model.slice_names[slice_index] + " Frame " + str(sub_slice)
				subslice_selected[sub_num] = self.subslice_selections[subslice_string].get()
			for channel_num, channel in enumerate(channel_list):
				channel_string = self.confocal_model.slice_names[slice_index] + ' ' + channel
				channel_selected[channel_num] = self.channel_selections[channel_string].get()
			sub_slices[slice_num] = list(np.where(subslice_selected)[0])
			slice_chans[slice_num] = list(np.where(channel_selected)[0])
		return([sub_slices, slice_chans])
	
root = tk.Tk()
gui = modelGUI(master=root)
gui.master.title('Cardiac Modeling Toolbox')
gui.mainloop()