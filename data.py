## Data.py
## Kebing Li
## CS251 Project2
## Feb.20.2018

import csv
import numpy
import analysis as an
import copy
import datetime
#import xlrd


# create a class to store and manage data
class Data():

	def __init__(self, filename=None):

		# field elements
		self.rawHeaders = []
		self.headers = []
		self.rawTypes = []
		self.types = []
		self.rawData = []
		self.data = []
		self.header2col = {}
		self.enum_dictionary = {}
		self.filename = filename

		self.numeric = []
		self.non_numeric = []
		self.string_column = []
		self.enum_column = []
		self.date_column = []

		if filename != None:
			self.read(filename)

	# read in the file
	def read(self, filename):

		# when dealing with csv file
		if "csv" in filename:
			with open(filename, mode='rU') as csvFile:

				csv_reader = csv.reader(csvFile)
				file = []
				for lines in csv_reader:
					file.append(lines)

				self.rawHeaders = copy.copy(file[0])
				self.headers = copy.copy(file[0])
				self.rawTypes = copy.copy(file[1])
				self.types = copy.copy(file[1])
				self.rawData = copy.deepcopy(file[2:])
				self.data = copy.deepcopy(file[2:])

		# when dealing with other types of file such as excel *** extension 4
		if "xls" in filename:
			workbook = xlrd.open_workbook(filename)
			sheet = workbook.sheet_by_index(0)
			file = []
			for rows in range(sheet.nrows):
				sublist = []
				for cols in range(sheet.ncols):
					value = str(sheet.cell_value(rows, cols))
					sublist.append(value)
				file.append(sublist)
			print(file)

			self.rawHeaders = copy.copy(file[0])
			self.headers = copy.copy(file[0])
			self.rawTypes = copy.copy(file[1])
			self.types = copy.copy(file[1])
			self.rawData = copy.deepcopy(file[2:])
			self.data = copy.deepcopy(file[2:])

		# differentiate different types of columns
		for i in range(len(self.types)):
			if "numeric" not in self.types[i]:
				self.non_numeric.append(i)
			if "numeric" in self.types[i]:
				self.numeric.append(i)
			if "enum" in self.types[i]:
				self.enum_column.append(i)
			if "date" in self.types[i]:
				self.date_column.append(i)
			if "string" in self.types[i]:
				self.string_column.append(i)

		# remove the non-numeric columns
		# make a new headers and a new types such that they discard the non-numeric values
		self.non_numeric.reverse()
		for element in self.non_numeric:
			index = int(element)
			self.headers.pop(index)
			self.types.pop(index)
		for lines in self.data:
			for element in self.non_numeric:
				index = int(element)
				lines.pop(index)

		# create a matrix object using numpy and store it to self.data
		self.data = numpy.matrix(self.data).astype(numpy.float)
		for i in range(len(self.headers)):
			self.header2col[self.headers[i]] = i

		# print(self.rawHeaders)
		# print(self.rawTypes)
		# print(self.rawData)
		# print(self.headers)
		# print(self.types)
		# print(self.data)
		self.convert_enum_to_numeric()
		self.convert_date_to_numeric("csv")

	# this function converts the enum columns into numeric values *** extension 1
	def convert_enum_to_numeric(self):
		enum = []
		for enum_element in self.rawData:
			for i in self.enum_column:
				enum.append(enum_element[i])
		j = 0
		for i in range(len(enum)):
			enum[i] = str(enum[i])
			if enum[i] not in self.enum_dictionary:
				self.enum_dictionary[enum[i]] = j
				j = j + 1
		for enum_element in self.rawData:
			for i in self.enum_column:
				enum_element[i] = int(self.enum_dictionary[enum_element[i]])

	# this function converts the date columns into ordinal numbers *** extension 2
	def convert_date_to_numeric(self, filetype):

		if filetype == "csv":
			datelist = []
			for date_element in self.rawData:
				for i in self.date_column:
					datelist.append(date_element[i])
			for i in range(len(datelist)):
				thing = datelist[i]

				# deal with multiple date formats *** extension 3
				if '/' in thing:
					date = thing.split('/')
				if '.' in thing:
					date = thing.split('.')
				if '-' in thing:
					date = thing.split('-')
				if ' ' in thing:
					date = thing.split(' ')
				if ',' in thing:
					date = thing.split(',')
				if ':' in thing:
					date = thing.split(':')

				month = int(date[0])
				day = int(date[1])
				year = int(date[2])

				date_object = datetime.date(year, month, day)
				date_num = date_object.toordinal()
				pos1 = int(i/len(self.date_column))
				pos2 = i%len(self.date_column)
				self.rawData[pos1][self.date_column[pos2]] = date_num

		elif filetype == "excel":
			datelist = []
			for date_element in self.rawData:
				for i in self.date_column:
					datelist.append(date_element[i])
			for i in range(len(datelist)):
				pos1 = int(i / len(self.date_column))
				pos2 = i % len(self.date_column)
				self.rawData[pos1][self.date_column[pos2]] = datelist[i]

	# gets the enum_dictionary for user to implement with the enum-type data
	def get_enum_dictionary(self):
		return self.enum_dictionary

	# get the top line
	def get_headers(self):
		return self.headers

	# get the second line
	def get_types(self):
		return self.types

	# get the number of columns
	def get_num_dimensions(self):
		return self.get_data().shape[1]

	# get the number of rows
	def num_points(self):
		return self.get_data().shape[0]

	# get the specific row
	def get_row(self, rowIndex):
		return self.get_data()[rowIndex - 1]

	# get the specific value
	def get_value(self, header, rowIndex):
		for i in range(len(self.headers)):
			if header == self.headers[i]:
				return self.get_data()[(rowIndex - 1), i]

	# get the data matrix
	def get_data(self):
		return self.data

	# takes in a list of columns headers
	# returns a Numpy matrix with the data for all rows but just the specified columns.
	def all_rows_specified_columns(self, headers):
		col = []
		print(self.header2col)
		print(headers)
		if len(headers) == 1:
			new_matrix = self.get_data()[:, [self.header2col[headers[0]]]]
			return new_matrix
		else:
			for header in headers:
				for i in range(len(self.headers)):
					if header == self.headers[i]:
						col.append(i)
			new_matrix = self.get_data()[:, col]
			return new_matrix



	# this function adds a column to the original data
	# if the added data happens to be numeric values, then add a column to the matrix
	# *** extension 5
	def addColumn(self, header, type, input):

		self.rawHeaders.append(header)
		self.rawTypes.append(type)

		for i in range(len(self.rawData)):
			self.rawData[i].append(input[i])

		new_data = None
		if 'numeric' in type:
			self.headers.append(header)
			self.types.append(type)
			input = numpy.matrix(input).astype(numpy.float).transpose()
			print(len(self.headers))
			self.header2col[header] = len(self.headers)-1

			new_data = numpy.hstack((self.data, input))
			self.data = new_data

		return


	# write function
	def write(self, filename, headers=None):

		fp = open(filename, 'w')
		if headers is None:
			headers = self.get_headers()
		data = self.all_rows_specified_columns(headers)

		for i in headers:
			fp.write(i + ",")
		fp.write("\n")

		for i in range(len(headers)):
			fp.write("numeric,")
		fp.write("\n")

		for k in range(self.all_rows_specified_columns(headers).shape[0]):
			for i in range(len(headers)):
				fp.write(str(data[k, i]) + ",")
			fp.write("\n")

		fp.close()



	# describe the file
	def __str__(self):
		string1 = "The data is as following:" + str(self.get_headers()) + str(self.get_types())
		string2 = numpy.array2string(self.data)
		string3 = "The dimension is: " + str(self.num_points()) + " x " + str(self.get_num_dimensions())
		return string1 + "\n" + string2 + "\n" + string3

# PCA Data inherits from Data object and contains more information for PCA Analysis
class PCAData(Data):

	def __init__(self, projected_data, eigenvectors, eigenvalues, data_means, original_headers):
		Data.__init__(self)
		self.eigenvalues = eigenvalues
		self.eigenvectors = eigenvectors
		self.meanValues = data_means
		self.originalHeaders = original_headers
		self.data = projected_data
		num_new_headers = projected_data.shape[1]
		for i in range(num_new_headers):
			self.headers.append("PCA" + str(i))
			self.types.append("numeric")
			self.header2col[self.headers[i]] = i

	def get_eigenvalues(self):
		return copy.deepcopy(self.eigenvalues)

	def get_eigenvectors(self):
		return  copy.deepcopy(self.eigenvectors)

	def get_original_means(self):
		return copy.deepcopy(self.meanValues)

	def get_original_headers(self):
		return copy.deepcopy(self.originalHeaders)

# NewData inherits from Data to merge to data objects together
class NewData(Data):
	def __init__(self, data1, data2):
		Data.__init__(self)
		self.data1 = data1
		self.data2 = data2
		self.headers = data1.headers + data2.headers
		self.types = data1.types + data2.types
		self.data = numpy.hstack((data1.data, data2.data))
		for i in range(len(self.headers)):
			self.header2col[self.headers[i]] = i
		print(self.headers)
		print(self.types)
		print(self.header2col)

# Cluster Data inherits from Data object and contains more information for clustering Analysis
class ClusterData(Data):

	def __init__(self, codebook, num_clusters, codes, original_data, errors, original_headers):
		Data.__init__(self)

		self.original_data = original_data
		self.original_headers = original_headers
		data = copy.deepcopy(original_data)
		self.data = data.addColumn('ID', 'numeric', codes)
		self.headers = data.get_headers()
		self.types = data.get_types()
		self.codebook = codebook
		self.num_clusters = num_clusters
		self.codes = codes
		self.errors = errors
		
		for i in range(len(self.headers)):
			self.header2col[self.headers[i]] = i

	def get_original_data(self):
		return copy.deepcopy(self.original_data)

	def get_original_headers(self):
		return copy.deepcopy(self.original_headers)

	def get_cluster_means(self):
		return copy.deepcopy(self.codebook)

	def get_num_clusters(self):
		return  copy.deepcopy(self.num_clusters)

	def get_ids(self):
		return copy.deepcopy(self.codes)

	def get_errors(self):
		return copy.deepcopy(self.errors)

def test(filename):
	data = Data(filename)
	data.addColumn('enumstuff3', 'enum', ['a','a','a','a','a','a','a','a','a','aa','aaa','a','a','a','aa'])
	data.addColumn('numberstuff3', 'numeric', [1,1,1,1,1,2,2,2,2,2,4,3,3,4,5])
	print(data.get_data())
	data.__str__()
	print(an.data_range([data.get_headers()[0], data.get_headers()[1]], filename))
	print(an.mean([data.get_headers()[0], data.get_headers()[1]], filename))
	print(an.stdev([data.get_headers()[0], data.get_headers()[1]], filename))
	print(an.normalize_columns_seperately([data.get_headers()[0], data.get_headers()[1]], filename))
	print(an.normalize_columns_together([data.get_headers()[0], data.get_headers()[1]], filename))


if __name__ == "__main__":
	data = Data('data-simple.csv')
	print(data.all_rows_specified_columns(['X0']))
#	test("testdata4.csv")
#	test("testdata_combined.csv")
	# testcode for data1
	# data = Data("testdata1.csv")
	# data.__str__()
	# print(data.get_row(3))
	# print(data.get_value('thing3', 2))
	# print(data.all_rows_specified_columns(["thing1", "thing2"]))
	# print(data.get_data())

	# testcode for data3
	# data = Data("testdata3.csv")
	# data.__str__()
	# print(data.all_rows_specified_columns(['headers', 'places']))
	# print(data.all_rows_specified_columns((["bad"])))

# testcode for data4
#	data = Data("testdata4.csv")
#	print(data.all_rows_specified_columns(['numberstuff']))
