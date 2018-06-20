### display.py
### project 8
### Kebing Li
### 05/5/2018
### CS251

import tkinter as tk
from tkinter import *
import math
import random
import numpy as np
import view
import data
import sys
from tkinter import filedialog
from tkinter import messagebox
import analysis as al
import scipy.stats
from PIL import ImageGrab
import csv
import naive_bayes_process as nbp
import knn_process as knnp

# create a class to build and manage the display
class DisplayApp:

	def __init__(self, width, height):

		# create a tk object, which is the root window
		self.root = tk.Tk()

		# width and height of the window
		self.initDx = width
		self.initDy = height

		# set up the geometry for the window
		self.root.geometry("%dx%d+50+30" % (self.initDx, self.initDy))

		# set the title of the window
		self.root.title("Multifunctional Integrated Clustering Analysis (MICA)")

		# set the maximum size of the window for resizing
		self.root.maxsize(1024, 768)

		# bring the window to the front
		self.root.lift()

		# setup the menus
		self.buildMenus()

		# build the controls
		self.buildControls()

		# build the objects on the Canvas
		self.buildCanvas()

		# set up the key bindings
		self.setBindings()

		# Create a View object and set up the default parameters
		self.view = view.View()
		self.view.reset()

		# Create the axes fields and build the axes
		self.axes = np.matrix([[0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 1],
		                       [0, 0, 0, 1], [0, 0, 1, 1]])

		# set up the application state
		self.line = []
		self.objects = []
		self.new_headers = []
		self.shape = 'circle'

		self.label = []
		# #self.data = data.Data("data.csv")
		# self.file = 'AustraliaCoast.csv'
		# self.file = 'african_hiv_econ.csv'
		self.file = None
		self.data = data.Data(self.file)
		self.translation_speed = 1.0
		self.scaling_speed = 1.0
		self.rotation_speed = 1.0

		self.linear = []
		self.endPoints = None
		self.regText = []
		self.axisLabel = []
		self.previousAnalysis = []
		self.count = 1

		self.pca_analysis = []
		self.num_analysis = 0

		self.clustering_analysis = []
		self.num_clustering_analysis = 0
		self.header2clustering_analysis = {}
		self.header2Analysis = {}
		self.des_leng = []
		self.pca_on = False

	# build the menu object
	def buildMenus(self):

		# create a new menu
		self.menu = tk.Menu(self.root)

		# set the root menu to our new menu
		self.root.config(menu=self.menu)

		# create a variable to hold the individual menus
		self.menulist = []

		# create a file menu
		filemenu = tk.Menu(self.menu)
		self.menu.add_cascade(label="File", menu=filemenu)
		self.menulist.append(filemenu)

		cmdmenu = tk.Menu(self.menu)
		self.menu.add_cascade(label="Command", menu=cmdmenu)
		self.menulist.append(cmdmenu)

		pcamenu = tk.Menu(self.menu)
		self.menu.add_cascade(label="PCA", menu=pcamenu)
		self.menulist.append(pcamenu)

		clustering_menu = tk.Menu(self.menu)
		self.menu.add_cascade(label="Clustering", menu=clustering_menu)
		self.menulist.append(clustering_menu)

		pcamenu = tk.Menu(self.menu)
		self.menu.add_cascade(label="Machine Learning", menu=pcamenu)
		self.menulist.append(pcamenu)

		# menu text for the elements
		menutext = [['Open...  \xE2\x8C\x98-O', 'New', 'Quit  \xE2\x8C\x98-Q'],
		            ['Linear Regression', 'Get Previous Results', 'Save Analysis To txt File', "Read A File and Replot",
		             "Save To Plot"],
		            ['PCA Analysis', 'Project To the view', 'Browse Results', 'Write to a csv File'],
		            ['Clustering Analysis', 'Project to the view', 'Browse Cluster Means', 'Calculate Quality', 'See All Qualities', 'Cluster on PCA'],
		            ['Classifier']]

		# menu callback functions
		menucmd = [[self.handleOpen, self.clearData, self.handleQuit],
		           [self.handleLinearRegression, self.handleGetPrevious, self.writeToFile, self.readAndReplot,
		            self.saveImage],
		           [self.handle_pca, self.project_view, self.browse_results, self.write_pca],
		           [self.handle_cluster, self.project_cluster, self.cluster_data_display, self.description_length, self.show_all_lengths, self.handle_cluster_on_pca],
		           [self.handle_classifier]]

		# build the menu elements and callbacks
		for i in range(len(self.menulist)):
			for j in range(len(menutext[i])):
				if menutext[i][j] != '-':
					self.menulist[i].add_command(label=menutext[i][j], command=menucmd[i][j])
				else:
					self.menulist[i].add_separator()

	# create the canvas object
	def buildCanvas(self):
		self.canvas = tk.Canvas(self.root, width=self.initDx, height=self.initDy)
		self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
		return

	# build a frame and put controls in it
	def buildControls(self):

		# make a control frame
		self.cntlframe = tk.Frame(self.root, bd=10)
		self.cntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

		sep = tk.Frame(self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN)
		sep.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

		# make a cmd 1 button in the frame
		self.buttons = []

		self.buttons.append(('Ultra reset',
		                     tk.Button(self.cntlframe, text="Ultra Reset", command=self.handleUltraResetButton,
		                               width=10)))
		self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

		self.buttons.append(
			('reset', tk.Button(self.cntlframe, text="Reset", command=self.handleResetButton, width=10)))
		self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

		self.buttons.append(
			('plot data', tk.Button(self.cntlframe, text="Plot Data", command=self.handlePlotData, width=10)))
		self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

		self.buttons.append(('resetYZ', tk.Button(self.cntlframe, text="ResetYZ", command=self.resetYZ, width=10)))
		self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

		self.buttons.append(('resetXZ', tk.Button(self.cntlframe, text="ResetXZ", command=self.resetXZ, width=10)))
		self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

		self.buttons.append(
			('calculate mean', tk.Button(self.cntlframe, text="mean", command=self.compute_mean, width=10)))
		self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

		self.buttons.append(('standard deviation',
		                     tk.Button(self.cntlframe, text="standard deviation", command=self.compute_std, width=15)))
		self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

		self.buttons.append(('range', tk.Button(self.cntlframe, text="range", command=self.compute_range, width=10)))
		self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

		# build a listbox with two options (oval, rectangle)
		# the default setting is Circle
		self.listbox2 = tk.Listbox(self.cntlframe, exportselection=0, height=2)
		self.listbox2.insert(tk.END, 'Circle', 'Square')
		self.listbox2.select_set(0)
		self.listbox2.pack(side=tk.TOP)

		# listbox for pca analysis
		self.listbox3 = tk.Listbox(self.cntlframe, exportselection=0)
		self.listbox3.pack(side=tk.TOP)
		b = Button(self.cntlframe, text="Delete", command=lambda: self.listbox3.delete(ANCHOR))
		b.pack(side=tk.TOP)

		# listbox for cluster analysis
		self.listbox4 = tk.Listbox(self.cntlframe, exportselection=0)
		self.listbox4.pack(side=tk.TOP)
		b = Button(self.cntlframe, text="Delete", command=lambda: self.listbox4.delete(ANCHOR))
		b.pack(side=tk.TOP)

		return

	# returns the selected shape
	def getShape(self):
		if self.listbox2.get(self.listbox2.curselection()) == "Square":
			self.shape = 'square'
			return 'square'
		else:
			self.shape = 'circle'
			return 'circle'

	# build the data points based on the data input
	def buildPoints(self, headers, together=False, cluster = False):

		if not cluster:
			self.size_not_select = False
			self.color_not_select = False
			for object in self.objects:
				self.canvas.delete(object)
			for lines in self.linear:
				self.canvas.delete(lines)

			self.linear = []

			self.objects = []
			# get the normalized size matrix if selected
			if len(headers) >= 5 and headers[4] != None:
				size_matrix_normalized = al.normalize_columns_separately(headers[4:5], self.data)
			else:
				self.size_not_select = True
			# get the normalized color matrix if selected
			if len(headers) >= 4 and headers[3] != None:
				color_matrix_normalized = al.normalize_columns_separately(headers[3:4], self.data)
			else:
				self.color_not_select = True
			if not together:
				self.matrix = al.normalize_columns_separately(headers[:3], self.data)
			else:
				self.matrix = al.normalize_columns_together(headers[:3], self.data)
			matrix = self.matrix.copy()
			matrix = matrix.T
			homo = np.matrix([1.0])
			for i in range(self.data.num_points() - 1):
				homo = np.hstack((homo, np.matrix([1.0])))
			self.matrix = np.vstack((matrix, homo))
			self.matrix = self.matrix.T
			matrix = self.view.build() * self.matrix.T

			if self.getShape() == 'circle':
				for i in range(len(matrix.T)):
					if self.size_not_select:
						dx = 5
					else:
						dx = size_matrix_normalized.item(i, 0) * 10
					if self.color_not_select:
						RGB = 'red'
					else:
						RGB = "#%02x%02x%02x" % (
						int(255 * color_matrix_normalized.item(i, 0)), int(255 * color_matrix_normalized.item(i, 0)),
						int(255 - 255 * color_matrix_normalized.item(i, 0), ))
					point = self.canvas.create_oval(matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] + dx,
					                                matrix[1, i] + dx, fill=RGB, outline='')
					self.objects.append(point)

			else:
				for i in range(len(matrix.T)):
					if self.size_not_select:
						dx = 5
					else:
						dx = size_matrix_normalized.item(i, 0) * 10
					if self.color_not_select:
						RGB = 'red'
					else:
						RGB = "#%02x%02x%02x" % (
						int(255 * color_matrix_normalized.item(i, 0)), int(255 * color_matrix_normalized.item(i, 0)),
						int(255 - 255 * color_matrix_normalized.item(i, 0), ))
					point = self.canvas.create_polygon(matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] - dx,
					                                   matrix[1, i] + dx, matrix[0, i] + dx, matrix[1, i] + dx,
					                                   matrix[0, i] + dx, matrix[1, i] - dx, fill=RGB, outline='')
					self.objects.append(point)
		else:
			self.size_not_select = False
			for object in self.objects:
				self.canvas.delete(object)
			for lines in self.linear:
				self.canvas.delete(lines)

			self.linear = []

			color_list = ['red', 'black', 'navy', 'yellow', 'green', 'salmon', 'gray', 'pink', 'purple', 'brown']
			selected = int(self.listbox4.curselection()[0])
			clusterdata = self.clustering_analysis[selected]
			ids = clusterdata.get_ids()
			num_clusters = clusterdata.get_num_clusters()
			color_selected = []
			if num_clusters<=10:
				index_list = np.arange(num_clusters)
				np.random.shuffle(index_list)
				list = index_list[:num_clusters]
				for element in list:
					color_selected.append(color_list[element])
			else:
				for i in range(num_clusters):
					de = ("%02x" % random.randint(0, 255))
					re = ("%02x" % random.randint(0, 255))
					we = ("%02x" % random.randint(0, 255))
					ge = "#"
					color = ge + de + re + we
					color_selected.append(color)


			self.objects = []
			# get the normalized size matrix if selected
			if len(headers) >= 5 and headers[4] != None:
				size_matrix_normalized = al.normalize_columns_separately(headers[4:5], self.data)
			else:
				self.size_not_select = True
			if not together:
				print(self.data)
				self.matrix = al.normalize_columns_separately(headers[:3], self.data)
			else:
				self.matrix = al.normalize_columns_together(headers[:3], self.data)
			matrix = self.matrix.copy()
			matrix = matrix.T
			homo = np.matrix([1.0])
			for i in range(self.data.num_points() - 1):
				homo = np.hstack((homo, np.matrix([1.0])))
			self.matrix = np.vstack((matrix, homo))
			self.matrix = self.matrix.T
			matrix = self.view.build() * self.matrix.T

			if self.getShape() == 'circle':
				for i in range(len(matrix.T)):
					if self.size_not_select:
						dx = 5
					else:
						dx = size_matrix_normalized.item(i, 0) * 10
					RGB = color_selected[int(ids[i])]
					point = self.canvas.create_oval(matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] + dx,
					                                matrix[1, i] + dx, fill=RGB, outline='')
					self.objects.append(point)

			else:
				for i in range(len(matrix.T)):
					if self.size_not_select:
						dx = 5
					else:
						dx = size_matrix_normalized.item(i, 0) * 10
					RGB = color_selected[int(ids[i])]
					point = self.canvas.create_polygon(matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] - dx,
					                                   matrix[1, i] + dx, matrix[0, i] + dx, matrix[1, i] + dx,
					                                   matrix[0, i] + dx, matrix[1, i] - dx, fill=RGB, outline='')
					self.objects.append(point)

	# update the data points after the user interactions
	def updatePoints(self):
		if len(self.objects) == 0:
			return
		else:
			if self.getShape() == 'circle':
				matrix = self.view.build() * self.matrix.T
				if not self.size_not_select:
					size_matrix_normalized = al.normalize_columns_separately(self.new_headers[4:5], self.data)
					for i in range(len(self.objects)):
						dx = size_matrix_normalized.item(i, 0) * 10
						self.canvas.coords(self.objects[i], matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] + dx,
						                   matrix[1, i] + dx)

				else:
					for i in range(len(self.objects)):
						dx = 5
						self.canvas.coords(self.objects[i], matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] + dx,
						                   matrix[1, i] + dx)
			else:
				matrix = self.view.build() * self.matrix.T
				if not self.size_not_select:
					size_matrix_normalized = al.normalize_columns_separately(self.new_headers[4:5], self.data)
					for i in range(len(self.objects)):
						dx = size_matrix_normalized.item(i, 0) * 10
						self.canvas.coords(self.objects[i], matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] - dx,
						                   matrix[1, i] + dx, matrix[0, i] + dx, matrix[1, i] + dx,
						                   matrix[0, i] + dx, matrix[1, i] - dx)

				else:
					for i in range(len(self.objects)):
						dx = 5
						self.canvas.coords(self.objects[i], matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] - dx,
						                   matrix[1, i] + dx, matrix[0, i] + dx, matrix[1, i] + dx,
						                   matrix[0, i] + dx, matrix[1, i] - dx)

	# build the axes at their default locations
	def buildAxes(self):
		if len(self.line) != 0:
			self.canvas.delete(self.line[0])
			self.canvas.delete(self.line[1])
			self.canvas.delete(self.line[2])
			for label in self.label:
				self.canvas.delete(label)

		self.line = []
		vtm = self.view.build()
		pts = (vtm * self.axes.T).T

		# create three new line objects, one for each axis.
		self.line.append(self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1]))
		self.line.append(self.canvas.create_line(pts[2, 0], pts[2, 1], pts[3, 0], pts[3, 1]))
		self.line.append(self.canvas.create_line(pts[4, 0], pts[4, 1], pts[5, 0], pts[5, 1]))

		# create three labels for the corresponding axis
		### extension 3 ###
		# print(self.data)
		# self.max1 = al.data_range([self.new_headers[0]], self.data)[0][0]
		# self.max2 = al.data_range([self.new_headers[1]], self.data)[0][0]
		#
		# if len(self.new_headers) >= 3:
		# 	self.max3 = al.data_range([self.new_headers[2]], self.data)[0][0]
		#
		# self.label = []
		# self.label.append(self.canvas.create_text(pts[1, 0], pts[1, 1],
		#                                           text="X-axis " + self.new_headers[0] + " max: " + str(self.max1),
		#                                           anchor=tk.SE))
		# self.label.append(self.canvas.create_text(pts[3, 0], pts[3, 1],
		#                                           text="Y-axis " + self.new_headers[1] + " max: " + str(self.max2),
		#                                           anchor=tk.SW))

		# if len(self.new_headers) >= 3:
		# 	self.label.append(self.canvas.create_text(pts[5, 0], pts[5, 1],
		# 	                                          text="Z-axis " + self.new_headers[2] + " max: " + str(self.max3),
		# 	                                          anchor=tk.NE))

	# modify the endpoints of the axes to their new location
	def updateAxes(self):

		# delete the original stuff
		if len(self.line) != 0:
			self.canvas.delete(self.line[0])
			self.canvas.delete(self.line[1])
			self.canvas.delete(self.line[2])
		else:
			return

		if len(self.label) != 0:
			for label in self.label:
				self.canvas.delete(label)

		# update axes and labels
		vtm = self.view.build()
		pts = (vtm * self.axes.T).T
		self.line[0] = self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1])
		self.line[1] = self.canvas.create_line(pts[2, 0], pts[2, 1], pts[3, 0], pts[3, 1])
		self.line[2] = self.canvas.create_line(pts[4, 0], pts[4, 1], pts[5, 0], pts[5, 1])

		# self.label.append(self.canvas.create_text(pts[1, 0], pts[1, 1],
		#                                           text="X-axis " + self.new_headers[0] + " max: " + str(self.max1),
		#                                           anchor=tk.SE))
		# self.label.append(self.canvas.create_text(pts[3, 0], pts[3, 1],
		#                                           text="Y-axis " + self.new_headers[1] + " max: " + str(self.max2),
		#                                           anchor=tk.SW))
		# self.label.append(self.canvas.create_text(pts[5, 0], pts[5, 1],
		#                                           text="Z-axis " + self.new_headers[2] + " max: " + str(self.max3),
		#                                           anchor=tk.NE))

	# update the best fit line
	def updateFits(self):
		if not self.linear:
			return
		for object in self.linear:
			self.canvas.delete(object)
		vtm = self.view.build()
		pts = (vtm * self.endPoints.T).T
		self.linear = []
		self.linear.append(self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1], fill="red"))

	# create the bindings
	def setBindings(self):
		self.root.bind('<Button-1>', self.handleButton1)
		self.root.bind('<Button-2>', self.handleButton2)
		self.root.bind('<Control-Button-1>', self.handleButton3)
		self.root.bind('<Button-3>', self.handleButton3)
		self.root.bind('<B1-Motion>', self.handleButton1Motion)
		self.root.bind('<B2-Motion>', self.handleButton2Motion)
		self.root.bind('<Control-B1-Motion>', self.handleButton3Motion)
		self.root.bind('<B3-Motion>', self.handleButton3Motion)
		self.root.bind('<Control-q>', self.handleQuit)
		self.root.bind('<Control-o>', self.handleModO)
		self.root.bind('<Control-O>', self.handleOpen)
		self.canvas.bind('<Configure>', self.handleResize)
		return

	# computes the mean of each selected variable
	def compute_mean(self):
		head = []
		for headers in self.new_headers:
			if headers != None:
				head.append(headers)
		list = al.mean(head, self.data)
		string = ''
		for i in range(len(head)):
			string += head[i] + ": " + str(list[i]) + '\n'
		messagebox.showinfo(title="selected column means are", message=string)

	# computes the standard deviation of each selected variable
	def compute_std(self):
		head = []
		for headers in self.new_headers:
			if headers != None:
				head.append(headers)
		list = al.stdev(head, self.data)
		string = ''
		for i in range(len(head)):
			string += head[i] + ": " + str(list[i]) + '\n'
		messagebox.showinfo(title="selected column standard deviations are", message=string)

	# computes the range of each selected variable
	def compute_range(self):
		head = []
		for headers in self.new_headers:
			if headers != None:
				head.append(headers)
		list = al.data_range(head, self.data)
		new_list = []
		for element in list:
			new_list.append([element[0].item(0, 0), element[1].item(0, 0)])
		string = ''
		for i in range(len(head)):
			string += head[i] + ": " + str(new_list[i]) + '\n'
		messagebox.showinfo(title="selected column ranges are", message=string)

	# completely reset the screen
	# delete all things include the data
	# the user has to import data
	def handleUltraResetButton(self):
		self.view.reset()
		self.degree = (0, 0)
		self.extent = [1, 1]
		self.rotation_speed = 1.0
		self.scaling_speed = 1.0
		self.translation_speed = 1.0
		for line in self.line:
			self.canvas.delete(line)
		self.line = []
		for label in self.label:
			self.canvas.delete(label)
		self.label = []
		# self.updateAxes()
		self.data = None
		for object in self.objects:
			self.canvas.delete(object)
		self.objects = []
		for line in self.linear:
			self.canvas.delete(line)
		self.linear = []
		# self.updatePoints()
		print('handling reset button')

	# click the reset button will move everything to the default location and the translation/scaling/rotation speed all equal to 1
	def handleResetButton(self):
		self.view.reset()
		self.degree = (0, 0)
		self.extent = [1, 1]
		self.rotation_speed = 1.0
		self.scaling_speed = 1.0
		self.translation_speed = 1.0
		self.updateAxes()
		self.updatePoints()
		self.updateFits()
		print('handling reset button')

	# view from yz plane
	def resetYZ(self, event=None):
		self.view.vrp = np.matrix([1, 0.5, 0.5])
		self.view.vpn = np.matrix([1, 0, 0])
		self.view.vup = np.matrix([0, 0, 1])
		self.view.u = np.matrix([[0, 1, 0]])
		self.degree = (0, 0)
		self.extent = [1, 1]
		self.updateAxes()
		self.updatePoints()
		self.updateFits()

	# view from xz plane
	def resetXZ(self, event=None):
		self.view.vrp = np.matrix([0.5, 1, 0.5])
		self.view.vpn = np.matrix([0, 1, 0])
		self.view.vup = np.matrix([1, 0, 0])
		self.view.u = np.matrix([[0, 0, 1]])
		self.degree = (0, 0)
		self.extent = [1, 1]
		self.updateAxes()
		self.updatePoints()
		self.updateFits()

	# let the screen adjust the streching of canvas
	def handleResize(self, event=None):

		wscale = event.width / self.initDx
		hscale = event.height / self.initDy
		self.initDx = event.width
		self.initDy = event.height
		self.canvas.config(width=self.initDx, height=self.initDy)
		self.canvas.scale("all", 0, 0, wscale, hscale)
		pass

	# open a file and assign the data object to self.data
	def handleOpen(self):
		fn = filedialog.askopenfilename(parent=self.root, title="Choose a data file", initialdir='.')
		self.file = fn
		self.data = data.Data(fn)
		print('handleOpen')

	# let the user selects the dimensions and then plot the data
	def handlePlotData(self):
		if self.data != None:
			headers = self.handleChooseAxes()
			if headers != False:
				print(headers)
				self.buildAxes()
				self.buildPoints(headers)
			else:
				return
		### extension 7 ###
		else:
			messagebox.showwarning("No Data", 'Need to import data')

	# handle the user's selection of axis.
	def handleChooseAxes(self):
		md1 = MyDialog(self.root, title="select_inputs", data=self.data)
		if not md1.userCancelled():
			x_header = md1.Xhead
			y_header = md1.Yhead
			z_header = md1.Zhead
			color_header = md1.colorhead
			size_header = md1.sizehead
			new_headers = [x_header, y_header, z_header, color_header, size_header]
			self.new_headers = new_headers
			return new_headers
		else:
			print("No value selected")
			return False

	# handele the user's selection of simple linear regression (1 independent variable and 1 dependent variable)
	def handleLinear(self):
		Dialog = LinearDialog(self.root, title="Choose Axis", data=self.data)
		return Dialog.result, Dialog.userCancelled()

	def handleModO(self, event):
		self.handleOpen()

	def handleQuit(self, event=None):
		print('Terminating')
		self.root.destroy()

	# clear all things on canvas and data
	def clearData(self, event=None):
		for item in self.linear:
			self.canvas.delete(item)
		self.endPoints = None
		for item in self.regText:
			self.canvas.delete(item)
		for item in self.objects:
			self.canvas.delete(item)
		self.objects = []
		self.linear = []

	# implement linear regression
	def handleLinearRegression(self):
		if self.data == None:
			print("please enter data")
			return
		headers = self.data.get_headers()
		self.LinearResult = self.handleLinear()

		if self.LinearResult[1] == True:
			print("selection cancelled")
			return

		self.clearData()

		self.linear = []
		self.endPoints = None

		self.handleResetButton()

		self.buildAxes()
		self.updateAxes()

		self.buildLinearRegression(self.LinearResult[0])

	# build the actual regression line
	def buildLinearRegression(self, headers):

		self.matrix = al.normalize_columns_separately(headers, self.file)
		matrix = self.matrix.copy()
		homo = np.matrix([0.0])
		for i in range(self.matrix.shape[0] - 1):
			homo = np.vstack((homo, np.matrix([0.0])))
		matrix = np.hstack((matrix, homo))

		homo = np.matrix([1.0])
		for i in range(self.matrix.shape[0] - 1):
			homo = np.vstack((homo, np.matrix([1.0])))
		self.matrix = np.hstack((matrix, homo))
		matrix = self.view.build() * self.matrix.T

		if self.getShape() == 'circle':
			for i in range(len(matrix.T)):
				dx = 5
				RGB = 'red'
				point = self.canvas.create_oval(matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] + dx,
				                                matrix[1, i] + dx, fill=RGB, outline='')
				self.objects.append(point)

		else:
			for i in range(len(matrix.T)):
				dx = 5
				RGB = 'red'
				point = self.canvas.create_polygon(matrix[0, i] - dx, matrix[1, i] - dx, matrix[0, i] - dx,
				                                   matrix[1, i] + dx, matrix[0, i] + dx, matrix[1, i] + dx,
				                                   matrix[0, i] + dx, matrix[1, i] - dx, fill=RGB, outline='')
				self.objects.append(point)

		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
			self.data.all_rows_specified_columns(headers).A)
		t_stat = al.linear_regression(self.data, [headers[0]], headers[1])[3]
		p_value = al.linear_regression(self.data, [headers[0]], headers[1])[4]

		drange = al.data_range(headers, self.file)
		xmin = drange[0][1]
		xmax = drange[0][0]
		ymin = drange[1][1]
		ymax = drange[1][0]

		self.endPoints = np.matrix([[0.0, ((xmin * slope + intercept) - ymin) / (ymax - ymin), 0.0, 1],
		                            [1.0, ((xmax * slope + intercept) - ymin) / (ymax - ymin), 0.0, 1]])
		pts = (self.view.build() * self.endPoints.T).T

		self.linear = []
		self.linear.append(self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1], fill="red"))

		####### extension 1 ########
		self.regText = []
		self.regText.append(self.canvas.create_text(200, 427, text="slope: " + str(slope), font=('Helvetica', '12')))
		self.regText.append(
			self.canvas.create_text(200, 442, text="intercept: " + str(intercept), font=('Helvetica', '12')))
		self.regText.append(
			self.canvas.create_text(200, 457, text="R^2: " + str(r_value * r_value), font=('Helvetica', '12')))
		self.regText.append(self.canvas.create_text(200, 472, text="t-stat: " + str(t_stat), font=('Helvetica', '12')))
		self.regText.append(
			self.canvas.create_text(200, 487, text="p-value: " + str(p_value), font=('Helvetica', '12')))

		self.previousAnalysis.append(
			[self.count, headers[0], headers[1], slope, intercept, r_value * r_value, p_value, std_err])
		self.count += 1

	# implement the pca analysis
	def handle_pca(self):
		dialog = PCADialog(self.root, title="Choose columns for PCA analysis", data=data.Data(self.file))
		if not dialog.userCancelled():
			headers = dialog.result
			self.pcaAnalysis(headers, True)
		else:
			print("Haven't chosen any variables")
			return

	# implement the clustering analysis
	def handle_cluster(self):
		dialog = ClusteringDialog(self.root, title="Choose columns for Clustering analysis", data=data.Data(self.file))
		if not dialog.userCancelled():
			headers = dialog.result
			num_cluster = dialog.num_cluster
			self.cluster_analysis(headers, num_cluster)
		else:
			print("Haven't chosen any variables")
			return

	# implement classifier application
	def handle_classifier(self):
		# let the user select the type of classifier
		dialog = BichoiceDialog(self.root, title="select classifier", subtitle='select classifier', choices=['Naive Bayes', 'KNN'])
		classifier = ''
		if dialog.result:
			classifier = 1
		else:
			classifier = 2

		# let the user choose the files (traindata, traincat, testdata, testcat)
		smallDialog = SmallDialog(self.root, title="select files", num=4)
		files = smallDialog.result
		if classifier == 1:
			self.nbc(files[0], files[1], files[2], files[3])
		else:
			self.knn(files[0], files[1], files[2], files[3])

	# implement Naive bayes process
	def nbc(self, traind, trainc, testd, testc):
		self.data = nbp.process(traind, trainc, testd, testc)
		print(self.data)
		dialog = MyDialog(self.root, title="Choose projected axis", data=self.data)
		if dialog.userCancelled():
			headers = self.data.headers[:]
			self.new_headers = headers
		else:
			headers = dialog.selectedheaders
			self.new_headers = headers
		self.buildAxes()
		self.buildPoints(headers, together=False)

	# implement kNN process
	def knn(self, traind, trainc, testd, testc, K =3):
		num = K
		self.data = knnp.process(traind, trainc, testd, testc, write = False, K = num)
		dialog = MyDialog(self.root, title="Choose projected axis", data=self.data)
		if dialog.userCancelled():
			headers = self.data.headers[:]
			self.new_headers = headers
		else:
			headers = dialog.selectedheaders
			self.new_headers = headers
		self.buildAxes()
		self.buildPoints(headers, together=False)

	# implement the combined analysis of pca and clustering
	def handle_cluster_on_pca(self):
		self.pca_on = True
		dialog = ClusteringDialog(self.root, title="Choose columns for Clustering analysis", data=self.data)
		if not dialog.userCancelled():
			headers = dialog.result
			num_cluster = dialog.num_cluster
			self.cluster_analysis(headers, num_cluster)
		else:
			print("Haven't chosen any variables")
			return

	# actual pca analysis
	def pcaAnalysis(self, headers, normalize):
		self.header2Analysis = {}
		self.num_analysis = self.num_analysis + 1

		pcaData = al.pca(data.Data(self.file), headers, normalize)
		self.pca_analysis.append(pcaData)

		# allow the user to enter name
		dialog = SmallDialog(self.root, title="Enter Name of Analysis")
		if not dialog.userCancelled():
			string = dialog.result[0]
		else:
			string = "Analysis" + str(self.num_analysis)
		self.listbox3.insert(tk.END, string)

	# acutal clustering analysis
	def cluster_analysis(self, headers, K):
		self.header2clustering_analysis = {}
		self.num_clustering_analysis = self.num_clustering_analysis + 1

		# let the user select the measurement
		#### extension4&5 #####
		choice = 0
		biDialog = BichoiceDialog(self.root, title="Measurement of Distance", subtitle= "Which measurement to calculate distance", choices = ['L1', 'L2'])
		if not biDialog.result:
			choice = 2
		else:
			choice = 1

		if not self.pca_on:
			codebook, codes, errors = al.kmeans(data.Data(self.file), headers, K, whiten=True, measurement = choice)
			clusterdata = data.ClusterData(codebook, K, codes, data.Data(self.file), errors, headers)
		else:
			codebook, codes, errors = al.kmeans(self.data, headers, K, whiten=True, measurement = choice)
			clusterdata = data.ClusterData(codebook, K, codes, self.data, errors, headers)
		self.clustering_analysis.append(clusterdata)
		print(clusterdata)
		quality = al.kmeans_quality(clusterdata.get_errors(), clusterdata.get_num_clusters())

		# allow the user to enter name
		##### extension 6 #####
		dialog = SmallDialog(self.root, title="Enter Name of Analysis")
		if not dialog.userCancelled():
			string = dialog.result[0]
		else:
			string = "Clustering Analysis" + str(self.num_clustering_analysis)
		self.listbox4.insert(tk.END, string)

		new_string = string + "'s description length: " + str(quality) + '\n'
		self.des_leng.append(new_string)
		self.pca_on = False

	# project the data to the axes
	def project_view(self):
		selected = int(self.listbox3.curselection()[0])
		pcaData = self.pca_analysis[selected]
		self.data = pcaData

		# allow the user to decide whether using mixing data or not
		##### extension 4 #####
		choice = BichoiceDialog(self.root, title="Whether using mixing data", subtitle= "Whether using mixing data", choices=["Yes", "No"])
		if not choice.result:
			# let the user to select up to five columns to plot
			#### extension 3 ####
			dialog = MyDialog(self.root, title="Choose projected axis", data=self.data)
			if dialog.userCancelled():
				if len(pcaData.headers) == 1:
					headers = [pcaData.headers[0], pcaData.headers[0], pcaData.headers[0]]
					self.new_headers = headers
				elif len(pcaData.headers) == 2:
					headers = [pcaData.headers[0], pcaData.headers[1], pcaData.headers[0]]
					self.new_headers = headers
				else:
					headers = pcaData.headers[:]
					self.new_headers = pcaData.headers
			else:
				headers = dialog.selectedheaders
				self.new_headers = headers
			self.buildAxes()
			self.buildPoints(headers, together=True)
		else:
			new_data = data.NewData(pcaData, data.Data(self.file))
			self.data = new_data
			print("Using new data")
			dialog = MyDialog(self.root, title="Choose projected axis", data=self.data)
			if dialog.userCancelled():
				if len(pcaData.headers) == 1:
					headers = [pcaData.headers[0], pcaData.headers[0], pcaData.headers[0]]
					self.new_headers = headers
				elif len(pcaData.headers) == 2:
					headers = [pcaData.headers[0], pcaData.headers[1], pcaData.headers[0]]
					self.new_headers = headers
				else:
					headers = pcaData.headers[:]
					self.new_headers = pcaData.headers
			else:
				headers = dialog.selectedheaders
				self.new_headers = headers
			self.buildAxes()
			self.buildPoints(headers, together=False)

	# project the clusterdata
	def project_cluster(self):
		selected = int(self.listbox4.curselection()[0])
		clusterdata = self.clustering_analysis[selected]
		print(clusterdata)
		self.data = clusterdata
		dialog = MyDialog(self.root, title="Choose projected axis", data=self.data)
		if dialog.userCancelled():
			if len(clusterdata.headers) == 1:
				headers = [clusterdata.headers[0], clusterdata.headers[0], clusterdata.headers[0]]
				self.new_headers = headers
			elif len(clusterdata.headers) == 2:
				headers = [clusterdata.headers[0], clusterdata.headers[1], clusterdata.headers[0]]
				self.new_headers = headers
			else:
				headers = clusterdata.headers[:]
				self.new_headers = clusterdata.headers
		else:
			headers = dialog.selectedheaders
			self.new_headers = headers
		self.buildAxes()
		print(self.data)
		self.buildPoints(self.new_headers, together=False, cluster=True)

	# create a table for important information
	def browse_results(self):
		selected = int(self.listbox3.curselection()[0])
		pcaData = self.pca_analysis[selected]
		eigval = pcaData.get_eigenvalues()
		eigvec = pcaData.get_eigenvectors()
		sum = 0
		energy = []
		for item in eigval:
			sum = sum + item
		thing2 = 0
		for item in eigval:
			thing1 = item / sum
			thing2 = thing1 + thing2
			energy.append(thing2)

		height = len(pcaData.headers) + 1
		width = len(pcaData.headers) + 3

		list = [["eigenvector", "eigenvalue", "energy"]]
		list[0] = list[0] + pcaData.originalHeaders

		for i in range(height - 1):
			sublist = []
			sublist.append(pcaData.headers[i])
			sublist.append(eigval[i])
			sublist.append(energy[i])
			for j in range(width - 3):
				sublist.append(eigvec.item(i, j))
			list.append(sublist)

		print(list)
		# BrowseDialog(self.root, title="data browse", data=list, height=height, width=width)

		myFile = open('pca_analysis' + str(selected) + '.csv', 'w')
		myFile.write(','.join(list[0]) + '\n')
		for row in list[1:]:
			for i in range(len(row)):
				if i>0:
					row[i] = str(np.round(row[i], 3))
			string = ','.join(row)
			myFile.write(string + '\n')

	# display the cluster means
	def cluster_data_display(self):
		selected = int(self.listbox4.curselection()[0])
		clusterdata = self.clustering_analysis[selected]
		codebook = clusterdata.get_cluster_means()

		headers = clusterdata.get_original_headers()
		width = len(headers)
		height = clusterdata.get_num_clusters()+1

		list = []
		list.append(headers)

		for i in range(height-1):
			list.append(codebook[i,:].tolist())

		print(list)
		BrowseDialog(self.root, title="cluster data browse", data=list, height=height, width=width)

	# show the description length of selected analysis
	def description_length(self):
		selected = int(self.listbox4.curselection()[0])
		clusterdata = self.clustering_analysis[selected]
		errors = clusterdata.get_errors()
		K = clusterdata.get_num_clusters()
		des_len = al.kmeans_quality(errors, K)
		messagebox.showinfo("Quality", "Description Length is: " + str(des_len))

	# show all descriotion lengths all together
	##### extension 6 #####
	def show_all_lengths(self):
		string = ''
		for element in self.des_leng:
			string = string + element
		messagebox.showinfo("Overall Qualities", string)

	# get the previous regression result
	def handleGetPrevious(self):
		string = ''
		for analysis in self.previousAnalysis:
			string = string + "Analysis " + str(analysis[0]) + ":" + "\n" + "Ind: " + str(analysis[1]) + " Dep: " + str(
				analysis[2]) + " Slope: " + str(analysis[3]) + " Intercept: " + str(analysis[4]) + " R-squared: " + str(
				analysis[5]) + " P-value: " + str(analysis[6]) + '\n'
		messagebox.showinfo(title="Previous Analysis", message=string)

	# save to a human readble file
	def writeToFile(self):
		if len(self.previousAnalysis) == 0:
			return
		file = open("saved_analysis.txt", "w")
		string = ''
		analysis = self.previousAnalysis[-1]
		string = "Analysis " + str(analysis[0]) + ":" + "\n" + "Ind: " + str(analysis[1]) + "\n" + "Dep: " + str(
			analysis[2]) + "\n" + "Slope: " + str(analysis[3]) + "\n" + "Intercept: " + str(
			analysis[4]) + "\n" + "R-squared: " + str(
			analysis[5]) + "\n" + "P-value: " + str(analysis[6])
		file.write(string)
		file.close()

	# write the pca analysis result to a csv file
	def write_pca(self):
		selected = int(self.listbox3.curselection()[0])
		pcaData = self.pca_analysis[selected]
		eigval = pcaData.get_eigenvalues()
		eigvec = pcaData.get_eigenvectors()
		firstrow = ['eigenvector', 'eigenvalue'] + pcaData.originalHeaders
		mean = pcaData.get_original_means()

		myData = [firstrow]
		for i in range(len(pcaData.headers)):
			list = [pcaData.headers[i]]
			list.append(eigval.tolist()[i])
			print(eigvec[i, :].tolist())
			for j in range(len(eigvec[i, :].tolist()[0])):
				list.append(eigvec[i, :].tolist()[0][j])
			myData.append(list)

		print(myData)

		myFile = open('pca_analysis.csv', 'w')
		with myFile:
			writer = csv.writer(myFile)
			writer.writerow(pcaData.originalHeaders)
			writer.writerow(['original mean:'])
			writer.writerows(mean.tolist())
			writer.writerow(['projected data:'])
			writer.writerows(pcaData.data.tolist())
			writer.writerows(myData)

	# let the user read in a specific_type_of file and replot the regression
	def readAndReplot(self):
		fn = filedialog.askopenfilename(parent=self.root, title="Choose a data file", initialdir='.')
		file = open(fn, "r")
		list = file.readlines()
		for element in list:
			print(element)
			if "Ind:" in element:
				header1 = element[5:-1]
				str1 = ''.join(str(e) for e in header1)

			elif "Dep:" in element:
				header2 = element[5:-1]
				str2 = ''.join(str(e) for e in header2)

		headers = [str1, str2]
		if str1 in self.data.get_headers() and str2 in self.data.get_headers():
			self.clearData()
			self.linear = []
			self.endPoints = None
			self.handleResetButton()
			self.buildAxes()
			self.updateAxes()
			self.buildLinearRegression(headers)
			file.close()
		else:
			print("Not Correct Data")
			messagebox.showwarning(title="Not Correct Data", message="Please import the correct data!")
			return

	# let the user save the regression to a plot
	def saveImage(self):
		x = self.root.winfo_rootx()
		y = self.root.winfo_rooty()
		x1 = x + self.initDx
		y1 = y + self.initDy
		ImageGrab.grab().crop((x, y, x1, y1)).save("image.png")

	# translation
	def handleButton1(self, event):
		self.baseClick = (event.x, event.y)
		print('handle button 1: %d %d' % (event.x, event.y))

	# rotation
	def handleButton2(self, event):
		self.baseClick2 = (event.x, event.y)
		self.original_view = self.view.clone()
		print('handle button 2: %d %d' % (event.x, event.y))

	# scaling
	def handleButton3(self, event):
		self.baseClick = (event.x, event.y)
		clone = self.view.clone()
		self.base_extent = []
		self.base_extent.append(clone.extent[0])
		self.base_extent.append(clone.extent[1])
		self.base_extent.append(clone.extent[2])
		print('handle button 3: %d %d' % (event.x, event.y))

	# translation
	def handleButton1Motion(self, event):
		diff = (event.x - self.baseClick[0], event.y - self.baseClick[1])
		dx = diff[0] / self.view.screen[0]
		dy = diff[1] / self.view.screen[1]
		delta0 = dx * self.view.extent[0] * self.translation_speed
		delta1 = dy * self.view.extent[1] * self.translation_speed
		self.view.vrp = self.view.vrp + delta0 * self.view.u + delta1 * self.view.vup
		self.updateAxes()
		self.updatePoints()
		self.updateFits()
		self.baseClick = (event.x, event.y)
		print('handle button 1 motion: %d %d' % (event.x, event.y))

	# rotation
	def handleButton2Motion(self, event):
		dx = self.baseClick2[0] - event.x
		dy = self.baseClick2[1] - event.y
		delta0 = dx / 500 * math.pi * self.rotation_speed
		delta1 = dy / 500 * math.pi * self.rotation_speed
		self.degree = (delta0, delta1)
		self.view = self.original_view.clone()
		self.view.rotateVRC(-delta0, delta1)
		self.updateAxes()
		self.updatePoints()
		self.updateFits()
		print('handle button 2 motion: %d %d' % (event.x, event.y))

	# scaling
	def handleButton3Motion(self, event):
		dy = float(event.y - self.baseClick[1]) / self.initDy
		scale = self.scaling_speed
		scale_factor = 1 + scale * dy
		scale_factor = np.max([scale_factor, 0.1])
		scale_factor = np.min([scale_factor, 3.0])
		self.view.extent[0] = scale_factor * self.base_extent[0]
		self.view.extent[1] = scale_factor * self.base_extent[1]
		self.updatePoints()
		self.updateAxes()
		self.updateFits()
		print('handle button 3 motion: %d %d' % (event.x, event.y))

	def main(self):
		print('Entering main loop')
		self.root.mainloop()


# a class that creates a Dialog Box object
class MyDialog(tk.Toplevel):

	def __init__(self, parent, title=None, index=[0, 0, 0, 0, 0, 0], data=None):

		self.e1 = index[0]
		self.e2 = index[1]
		self.e3 = index[2]
		self.e4 = index[3]
		self.e5 = index[4]

		self.data = data
		self.header = self.data.get_headers()

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent
		self.result = None

		self.applied = False
		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5)

		self.buttonbox()
		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
		                          parent.winfo_rooty() + 50))

		self.initial_focus.focus_set()
		self.wait_window(self)

		self.column = None

	def body(self, master):

		b1 = tk.Label(master, text="---Matching Data to X-Axis---")
		b1.grid(row=0, column=0)
		self.l1 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		self.l1.grid(row=1, column=0)
		for header in self.header:
			self.l1.insert(tk.END, header)

		b2 = tk.Label(master, text="---Matching Data to Y-Axis---")
		b2.grid(row=0, column=1)
		self.l2 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		self.l2.grid(row=1, column=1)
		for header in self.header:
			self.l2.insert(tk.END, header)

		b3 = tk.Label(master, text="---Matching Data to Z-Axis---")
		b3.grid(row=0, column=2)
		self.l3 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		self.l3.grid(row=1, column=2)
		for header in self.header:
			self.l3.insert(tk.END, header)

		b4 = tk.Label(master, text="---Color---")
		b4.grid(row=0, column=3)
		self.l4 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		self.l4.grid(row=1, column=3)
		for header in self.header:
			self.l4.insert(tk.END, header)

		b5 = tk.Label(master, text="---Size---")
		b5.grid(row=0, column=4)
		self.l5 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		self.l5.grid(row=1, column=4)
		for header in self.header:
			self.l5.insert(tk.END, header)

		self.l1.select_set(self.e1)
		self.l2.select_set(self.e2)
		self.l3.select_set(self.e3)
		# self.l4.select_set(self.e4)
		# self.l5.select_set(self.e5)

		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	# This function confirms user's selection.
	def ok(self, event=None):

		if not self.validate():
			self.initial_focus.focus_set()  # put focus back
			return

		self.withdraw()
		self.update_idletasks()

		self.apply()

		self.cancel()

	# This function cancel's user's selection.
	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()

	def validate(self):

		return 1

	# This function take in user's selection and sort selected data for the six dimensions.
	def apply(self):

		self.selectedheaders = []

		self.Xhead = self.l1.get(self.l1.curselection())
		self.Yhead = self.l2.get(self.l2.curselection())
		self.Zhead = self.l3.get(self.l3.curselection())
		if len(self.l4.curselection()) != 0:
			self.colorhead = self.l4.get(self.l4.curselection())
		else:
			self.colorhead = None
		if len(self.l5.curselection()) != 0:
			self.sizehead = self.l5.get(self.l5.curselection())
		else:
			self.sizehead = None
		self.selectedheaders.append(self.Xhead)
		self.selectedheaders.append(self.Yhead)
		self.selectedheaders.append(self.Zhead)
		self.selectedheaders.append(self.colorhead)
		self.selectedheaders.append(self.sizehead)

		self.applied = True

		pass

	# returns true if user clicks "cancel"
	def userCancelled(self):
		if self.applied is True:
			print("ok")
			return False
		else:
			print("cancel")
			return True


# a class that creates a Dialog Box for linear regression
class LinearDialog(tk.Toplevel):

	def __init__(self, parent, title=None, index=[0, 0], data=None):

		self.e1 = index[0]
		self.e2 = index[1]

		self.data = data
		self.header = self.data.get_headers()

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent
		self.result = []

		self.applied = False
		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5)

		self.buttonbox()
		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
		                          parent.winfo_rooty() + 50))

		self.initial_focus.focus_set()
		self.wait_window(self)

		self.column = None

	def body(self, master):

		b1 = tk.Label(master, text="---Matching Data to X-Axis---")
		b1.grid(row=0, column=0)
		self.l1 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		self.l1.grid(row=1, column=0)
		for header in self.header:
			self.l1.insert(tk.END, header)

		b2 = tk.Label(master, text="---Matching Data to Y-Axis---")
		b2.grid(row=0, column=1)
		self.l2 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		self.l2.grid(row=1, column=1)
		for header in self.header:
			self.l2.insert(tk.END, header)

		self.l1.select_set(self.e1)
		self.l2.select_set(self.e2)

		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	# This function confirms user's selection.
	def ok(self, event=None):

		if not self.validate():
			self.initial_focus.focus_set()  # put focus back
			return

		self.withdraw()
		self.update_idletasks()

		self.apply()

		self.cancel()

	# This function cancel's user's selection.
	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()

	def validate(self):

		return 1

	# This function take in user's selection and sort selected data for the six dimensions.
	def apply(self):

		self.Xhead = self.l1.get(self.l1.curselection())
		self.Yhead = self.l2.get(self.l2.curselection())
		# if len(self.l4.curselection())!=0:
		# 	self.colorhead = self.l4.get(self.l4.curselection())
		# else:
		# 	self.colorhead = None
		# if len(self.l5.curselection())!=0:
		# 	self.sizehead = self.l5.get(self.l5.curselection())
		# else:
		# 	self.sizehead = None
		self.result.append(self.Xhead)
		self.result.append(self.Yhead)
		# self.selectedheaders.append(self.Zhead)
		# self.selectedheaders.append(self.colorhead)
		# self.selectedheaders.append(self.sizehead)

		self.applied = True

		pass

	# returns true if user clicks "cancel"
	def userCancelled(self):
		if self.applied is True:
			print("ok")
			return False
		else:
			print("cancel")
			return True


# a class that creates a Dialog box for choosing columns in PCA analysis
class PCADialog(tk.Toplevel):

	def __init__(self, parent, title=None, data=None):

		self.data = data
		self.header = self.data.get_headers()

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent
		self.result = []

		self.applied = False
		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5)

		self.buttonbox()
		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
		                          parent.winfo_rooty() + 50))

		self.initial_focus.focus_set()
		self.wait_window(self)

		self.column = None

	def body(self, master):

		b1 = tk.Label(master, text="---Variables for PCA Analysis---")
		b1.grid(row=0, column=0)
		self.l1 = tk.Listbox(master, selectmode=tk.MULTIPLE, exportselection=0)
		self.l1.grid(row=1, column=0)
		for header in self.header:
			self.l1.insert(tk.END, header)

		self.l1.select_set(0)

		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	# This function confirms user's selection.
	def ok(self, event=None):

		if not self.validate():
			self.initial_focus.focus_set()  # put focus back
			return

		self.withdraw()
		self.update_idletasks()

		self.apply()

		self.cancel()

	# This function cancel's user's selection.
	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()

	def validate(self):

		return 1

	# This function take in user's selection and sort selected data for the six dimensions.
	def apply(self):

		self.Xhead = self.l1.curselection()
		for selection in self.Xhead:
			self.result.append(self.l1.get(selection))

		self.applied = True

		pass

	# returns true if user clicks "cancel"
	def userCancelled(self):
		if self.applied is True:
			print("ok")
			return False
		else:
			print("cancel")
			return True


# dialog box as a table
class BrowseDialog(tk.Toplevel):

	def __init__(self, parent, title=None, data=None, height=0, width=0):

		self.data = data
		self.height = height
		self.width = width

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent
		self.result = []

		self.applied = False
		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5)

		self.buttonbox()
		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
		                          parent.winfo_rooty() + 50))

		self.initial_focus.focus_set()
		self.wait_window(self)

		self.column = None

	def body(self, master):

		for i in range(self.height):
			for j in range(self.width):
				tk.Label(master, text=self.data[i][j], borderwidth=1).grid(row=i, column=j)

		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	# This function confirms user's selection.
	def ok(self, event=None):

		self.withdraw()
		self.update_idletasks()
		self.cancel()

	# This function cancel's user's selection.
	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()


# One entry dialog box
class SmallDialog(tk.Toplevel):

	def __init__(self, parent, title=None, num = 1):

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent
		self.result = []
		self.num = num

		self.applied = False
		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5)

		self.buttonbox()
		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
		                          parent.winfo_rooty() + 50))

		self.initial_focus.focus_set()
		self.wait_window(self)

		self.column = None

	def body(self, master):

		b1 = tk.Label(master, text="---Name of Analysis---")
		b1.grid(row=0, column=0)
		if self.num == 1:
			self.e1 = tk.Entry(master, exportselection=0)
			self.e1.grid(row=1, column=0)
			pass
		else:
			self.e1 = tk.Entry(master, exportselection=0)
			self.e1.grid(row=1, column=0)
			self.e2 = tk.Entry(master, exportselection=0)
			self.e2.grid(row=2, column=0)
			self.e3 = tk.Entry(master, exportselection=0)
			self.e3.grid(row=3, column=0)
			self.e4 = tk.Entry(master, exportselection=0)
			self.e4.grid(row=4, column=0)

		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	# This function confirms user's selection.
	def ok(self, event=None):

		if not self.validate():
			self.initial_focus.focus_set()  # put focus back
			return

		self.withdraw()
		self.update_idletasks()

		self.apply()

		self.cancel()

	# This function cancel's user's selection.
	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()

	def validate(self):

		return 1

	# This function take in user's selection and sort selected data for the six dimensions.
	def apply(self):

		if self.num == 1:
			self.result.append(self.e1.get())
		else:
			self.result.append(self.e1.get())
			self.result.append(self.e2.get())
			self.result.append(self.e3.get())
			self.result.append(self.e4.get())

		self.applied = True

		pass

	# returns true if user clicks "cancel"
	def userCancelled(self):
		if self.applied is True:
			print("ok")
			return False
		else:
			print("cancel")
			return True


# two-choice dialog box
class BichoiceDialog(tk.Toplevel):

	def __init__(self, parent, title=None, subtitle = None, choices=[]):

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		self.c1 = choices[0]
		self.c2 = choices[1]

		if title:
			self.title(title)

		self.parent = parent
		self.result = False

		self.applied = False
		body = tk.Frame(self)
		self.initial_focus = self.body(body, subtitle)
		body.pack(padx=5, pady=5)

		self.buttonbox()
		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
		                          parent.winfo_rooty() + 50))

		self.initial_focus.focus_set()
		self.wait_window(self)

		self.column = None

	def body(self, master, subtitle):

		b1 = tk.Label(master, text= subtitle)
		b1.grid(row=0, column=0)

		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text=self.c1, width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text=self.c2, width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	# This function confirms user's selection.
	def ok(self, event=None):

		if not self.validate():
			self.initial_focus.focus_set()  # put focus back
			return

		self.withdraw()
		self.update_idletasks()

		self.apply()
		self.result = True

		self.cancel()

	# This function cancel's user's selection.
	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()

	def validate(self):

		return 1

	# This function take in user's selection and sort selected data for the six dimensions.
	def apply(self):

		self.applied = True

		pass

	# returns true if user clicks "cancel"
	def userCancelled(self):
		if self.applied is True:
			print("ok")
			return False
		else:
			print("cancel")
			return True

# a class that creates a Dialog box for choosing columns in PCA analysis
class ClusteringDialog(tk.Toplevel):

	def __init__(self, parent, title=None, data=None):

		self.data = data
		self.header = self.data.get_headers()

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent
		self.result = []
		self.num_cluster = 0

		self.applied = False
		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5)

		self.buttonbox()
		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
		                          parent.winfo_rooty() + 50))

		self.initial_focus.focus_set()
		self.wait_window(self)

		self.column = None

	def body(self, master):

		b1 = tk.Label(master, text="Choose Columns for Clustering Analysis---")
		b1.grid(row=0, column=0)
		self.l1 = tk.Listbox(master, selectmode=tk.MULTIPLE, exportselection=0)
		self.l1.grid(row=1, column=0)
		for header in self.header:
			self.l1.insert(tk.END, header)

		self.l1.select_set(0)

		b2 = tk.Label(master, text="---Number of Clusters---")
		b2.grid(row=2, column=0)
		self.e1 = tk.Entry(master, exportselection=0)
		self.e1.grid(row=3, column=0)

		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	# This function confirms user's selection.
	def ok(self, event=None):

		if not self.validate():
			self.initial_focus.focus_set()  # put focus back
			return

		self.withdraw()
		self.update_idletasks()

		self.apply()

		self.cancel()

	# This function cancel's user's selection.
	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()

	def validate(self):

		return 1

	# This function take in user's selection and sort selected data for the six dimensions.
	def apply(self):

		self.Xhead = self.l1.curselection()
		for selection in self.Xhead:
			self.result.append(self.l1.get(selection))

		self.num_cluster = int(self.e1.get())
		self.applied = True

		pass

	# returns true if user clicks "cancel"
	def userCancelled(self):
		if self.applied is True:
			print("ok")
			return False
		else:
			print("cancel")
			return True

if __name__ == "__main__":
	dapp = DisplayApp(1000, 700)
	dapp.main()

