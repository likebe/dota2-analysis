# Template by Bruce Maxwell
# Spring 2015
# CS 251 Project 8
# Modified by Kebing Li
# Classifier class and child definitions

import sys
import data
import analysis as an
import numpy as np
import math
from scipy import spatial

# class to implement two classifiers
class Classifier:

	def __init__(self, type):
		'''The parent Classifier class stores only a single field: the type of
		the classifier.  A string makes the most sense.

		'''
		self._type = type

	def type(self, newtype=None):
		'''Set or get the type with this function'''
		if newtype != None:
			self._type = newtype
		return self._type

	def confusion_matrix(self, truecats, classcats):
		'''Takes in two Nx1 matrices of zero-index numeric categories and
		computes the confusion matrix. The rows represent true
		categories, and the columns represent the classifier output.
		To get the number of classes, you can use the np.unique
		function to identify the number of unique categories in the
		truecats matrix.

		'''
		unique = np.unique(truecats.A)
		confusion_matrix = np.asmatrix(np.zeros((unique.shape[0], unique.shape[0])))
		for i in range(truecats.shape[0]):
			confusion_matrix[int(classcats[i, 0]), int(truecats[i, 0])] += 1
		return confusion_matrix

		return

	def confusion_matrix_str(self, cmtx):
		'''Takes in a confusion matrix and returns a string suitable for printing.'''
		s = 'Confusion Matrix->' + '\n'
		s += '            Classified As' + '\n'
		s += 'Truth'
		for i in range(cmtx.shape[1]):
			s += '%10d' % (i)
		s += "\n"
		for i in range(cmtx.shape[0]):
			s += '%d' % (i)
			s += '    '
			for k in range(cmtx.shape[1]):
				s += "%10d" % (cmtx[k,i])
			s += "\n"

		return s

	def __str__(self):
		'''Converts a classifier object to a string.  Prints out the type.'''
		return str(self._type)


class NaiveBayes(Classifier):
	'''NaiveBayes implements a simple NaiveBayes classifier using a
	Gaussian distribution as the pdf.

	'''

	def __init__(self, data=None, headers=[], categories=None):
		'''Takes in a Matrix of data with N points, a set of F headers, and a
		matrix of categories, one category label for each data point.'''

		# call the parent init with the type
		Classifier.__init__(self, 'Naive Bayes Classifier')

		# store the headers used for classification
		# number of classes and number of features
		# original class labels
		# unique data for the Naive Bayes: means, variances, scales, priors
		# if given data,
		# call the build function
		self.headers = headers
		self.num_class = None
		self.num_feature = None
		self.class_labels = None
		self.class_means = None
		self.class_vars = None
		self.class_scales = None
		self.class_priors = None

		if data != None:
			self.build(data.all_rows_specified_columns(self.headers), categories)

	def build(self, A, categories):
		'''Builds the classifier give the data points in A and the categories'''

		# figure out how many categories there are and get the mapping (np.unique)
		# create the matrices for the means, vars, and scales
		# the output matrices will be categories x features
		# compute the means/vars/scales/priors for each class
		# the prior for class i will be the number of examples in class i divided by the total number of examples
		# store any other necessary information: # of classes, # of features, original labels

		print(categories)
		unique, mapping, counts = np.unique(np.array(categories.T), return_inverse=True, return_counts=True)
		self.num_class = len(unique)
		self.num_feature = A.shape[1]
		self.class_labels = unique

		self.class_means = np.asmatrix(np.zeros((self.num_class, self.num_feature)))
		self.class_vars = np.asmatrix(np.zeros((self.num_class, self.num_feature)))
		self.class_scales = np.asmatrix(np.zeros((self.num_class, self.num_feature)))

		self.class_priors = []
		for i in range(self.num_class):
			self.class_priors.append(counts[i]/A.shape[0])

		for i in range(self.num_class):
			for j in range(self.num_feature):
				self.class_means[i, j] = np.mean(A[(mapping == i), j])
				self.class_vars[i, j] = np.var(A[(mapping == i), j], ddof=self.num_feature-1)
				self.class_scales[i, j] = 1.0 / math.sqrt(2 * math.pi * self.class_vars[i, j])

		return

	def classify(self, A, return_likelihoods=False):
		'''Classify each row of A into one category. Return a matrix of
		category IDs in the range [0..C-1], and an array of class
		labels using the original label values. If return_likelihoods
		is True, it also returns the probability value for each class, which
		is product of the probability of the data given the class P(Data | Class)
		and the prior P(Class).

		'''

		# error check to see if A has the same number of columns as the class means

		# make a matrix that is N x C to store the probability of each class for each data point
		P = ''  # a matrix of zeros that is N (rows of A) x C (number of classes)

		# Calcuate P(D | C) by looping over the classes
		#  with numpy-fu you can do this in one line inside a for
		#  loop, calculating a column of P in each loop.
		#
		#  To compute the likelihood, use the formula for the Gaussian
		#  pdf for each feature, then multiply the likelihood for all
		#  the features together The result should be an N x 1 column
		#  matrix that gets assigned to a column of P

		# Multiply the likelihood for each class by its corresponding prior

		if (A.shape[1] != self.class_means.shape[1]):
			print("Error with input!")
			return

		row = A.shape[0]

		P = np.asmatrix(np.zeros((row, self.num_class)))

		for i in range(self.num_class):
			for j in range(A.shape[0]):
				a = self.class_vars[i, :] * 2
				b = np.square(A[j, :] - self.class_means[i, :])
				P[j, i] = np.prod(np.multiply(self.class_scales[i, :], np.exp(-b / a)))*self.class_priors[i]

		# calculate the most likely class for each data point
		cats = np.argmax(P, axis=1)  # take the argmax of P along axis 1

		# use the class ID as a lookup to generate the original labels
		labels = self.class_labels[cats]

		if return_likelihoods:
			return cats, labels, P

		return cats, labels

	def __str__(self):
		'''Make a pretty string that prints out the classifier information.'''
		s = "\nNaive Bayes Classifier\n"
		for i in range(self.num_class):
			s += 'Class %d --------------------\n' % (i)
			s += 'Mean  : ' + str(self.class_means[i, :]) + "\n"
			s += 'Var   : ' + str(self.class_vars[i, :]) + "\n"
			s += 'Scales: ' + str(self.class_scales[i, :]) + "\n"
			s += 'Prior: ' + str(self.class_priors[i]) + "\n"

		s += "\n"
		return s

	def write(self, filename):
		'''Writes the Bayes classifier to a file.'''
		fp = open(filename, 'w')

		for i in data.get_headers():
			fp.write(i + ",")
		fp.write("\n")

		for i in range(len(self.headers)):
			fp.write("numeric,")
		fp.write("\n")

		for k in range(data.self.headers.shape[0]):
			for i in range(len(self.headers)):
				fp.write(str(data[k, i]) + ",")
			fp.write("\n")
		return

	def read(self, filename):
		'''Reads in the Bayes classifier from the file'''
		# extension
		return


class KNN(Classifier):

	def __init__(self, data=None, headers=[], categories=None, K=None):
		'''Take in a Matrix of data with N points, a set of F headers, and a
		matrix of categories, with one category label for each data point.'''

		# call the parent init with the type
		Classifier.__init__(self, 'KNN Classifier')

		# store the headers used for classification
		# number of classes and number of features
		# original class labels
		# unique data for the KNN classifier: list of exemplars (matrices)
		# if given data,
		# call the build function
		self.headers = headers
		self.num_class = None
		self.num_feature = None
		self.class_labels = None
		self.exemplar = []
		if data is not None:
			self.build(data.all_rows_specified_columns(self.headers),categories, K)

	def build(self, A, categories, K=None):
		'''Builds the classifier give the data points in A and the categories'''

		# figure out how many categories there are and get the mapping (np.unique)
		# for each category i, build the set of exemplars
		# if K is None
		# append to exemplars a matrix with all of the rows of A where the category/mapping is i
		# else
		# run K-means on the rows of A where the category/mapping is i
		# append the codebook to the exemplars

		# store any other necessary information: # of classes, # of features, original labels

		unique, mapping = np.unique(np.array(categories.T), return_inverse=True)
		self.num_class = len(unique)
		self.num_feature = A.shape[1]

		self.class_labels = unique
		for i in range(self.num_class):
			if K is None:
				self.exemplar.append(A[(mapping == i), :])
			else:
				codebook, codes, errors = an.kmeans(A[(mapping == i), :], self.headers, K, whiten=False)
				self.exemplar.append(codebook)

		return

	def classify(self, A, return_distances=False, K=3):
		'''Classify each row of A into one category. Return a matrix of
		category IDs in the range [0..C-1], and an array of class
		labels using the original label values. If return_distances is
		True, it also returns the NxC distance matrix. The distance is
		calculated using the nearest K neighbors.'''

		# error check to see if A has the same number of columns as the class means
		if (A.shape[1] != self.num_feature):
			print("error with input")
			return

		# make a matrix that is N x C to store the distance to each class for each data point
		D = np.asmatrix(np.zeros((A.shape[0],self.num_class))) # a matrix of zeros that is N (rows of A) x C (number of classes)


		# for each class i
		# make a temporary matrix that is N x M where M is the number of examplars (rows in exemplars[i])
		# calculate the distance from each point in A to each point in exemplar matrix i (for loop)
		# sort the distances by row
		# sum the first K columns
		# this is the distance to the first class
		for i in range(self.num_class):
			matrix = np.asmatrix(np.zeros((A.shape[0], self.exemplar[i].shape[0])))
			for j in range(A.shape[0]):
				for k in range(self.exemplar[i].shape[0]):
					matrix[j, k] = spatial.distance.euclidean(A[j], self.exemplar[i][k])


			matrix.sort(axis=1)
			D[:, i] = np.sum(matrix[:, :K], axis=1)

		# calculate the most likely class for each data point
		cats = np.argmin(D, axis=1)  # take the argmin of D along axis 1

		# use the class ID as a lookup to generate the original labels
		labels = self.class_labels[cats]

		if return_distances:
			return cats, labels, D

		return cats, labels

	def __str__(self):
		'''Make a pretty string that prints out the classifier information.'''
		s = "\nKNN Classifier\n"
		for i in range(self.num_class):
			s += 'Class %d --------------------\n' % (i)
			s += 'Number of Exemplars: %d\n' % (self.exemplar[i].shape[0])
			s += 'Mean of Exemplars  :' + str(np.mean(self.exemplar[i], axis=0)) + "\n"

		s += "\n"
		return s

	def write(self, filename):
		'''Writes the KNN classifier to a file.'''
		# extension
		return

	def read(self, filename):
		'''Reads in the KNN classifier from the file'''
		# extension
		return


# test function
def main(argv):
	# test function here
	if len(argv) < 3:
		print(
			'Usage: python %s <training data file> <test data file> <optional training categories file> <optional test categories file>' % (
			argv[0]))
		print(
			'    If categories are not provided as separate files, then the last column is assumed to be the category.')
		exit(-1)

	train_file = argv[1]
	test_file = argv[2]
	dtrain = data.Data(train_file)
	dtest = data.Data(test_file)

	if len(argv) >= 5:
		train_headers = dtrain.get_headers()
		test_headers = dtrain.get_headers()

		traincat_file = argv[3]
		testcat_file = argv[4]

		traincats = data.Data(traincat_file)
		traincatdata = traincats.all_rows_specified_columns(traincats.get_headers())

		testcats = data.Data(testcat_file)
		testcatdata = testcats.all_rows_specified_columns(testcats.get_headers())

	else:
		train_headers = dtrain.get_headers()[:-1]
		test_headers = dtrain.get_headers()[:-1]

		traincatdata = dtrain.all_rows_specified_columns([dtrain.get_headers()[-1]])
		testcatdata = dtest.all_rows_specified_columns([dtest.get_headers()[-1]])

	nbc = NaiveBayes(dtrain, train_headers, traincatdata)

	print('Naive Bayes Training Set Results')
	A = dtrain.all_rows_specified_columns(train_headers)

	newcats, newlabels = nbc.classify(A)

	uniquelabels, correctedtraincats = np.unique(traincatdata.T.tolist()[0], return_inverse=True)
	correctedtraincats = np.matrix([correctedtraincats]).T

	confmtx = nbc.confusion_matrix(correctedtraincats, newcats)
	print("stop!")
	print(nbc.confusion_matrix_str(confmtx))

	print('Naive Bayes Test Set Results')
	A = dtest.all_rows_specified_columns(test_headers)

	newcats, newlabels = nbc.classify(A)

	uniquelabels, correctedtestcats = np.unique(testcatdata.T.tolist()[0], return_inverse=True)
	correctedtestcats = np.matrix([correctedtestcats]).T

	confmtx = nbc.confusion_matrix(correctedtestcats, newcats)
	print(nbc.confusion_matrix_str(confmtx))

	# dtest.addColumn("Category", "numeric", newcats.T.A[0])
	#
	# dtest.write("own_data_nb_result.csv", headers=dtest.get_headers())

	print('-----------------')
	print('Building KNN Classifier')

	knnc = KNN(dtrain, train_headers, traincatdata,5)

	print('KNN Training Set Results')
	A = dtrain.all_rows_specified_columns(train_headers)

	newcats, newlabels = knnc.classify(A)

	confmtx = knnc.confusion_matrix(correctedtraincats, newcats)
	print(knnc.confusion_matrix_str(confmtx))

	print('KNN Test Set Results')
	A = dtest.all_rows_specified_columns(test_headers)

	newcats, newlabels = knnc.classify(A)

	# print the confusion matrix
	confmtx = knnc.confusion_matrix(correctedtestcats, newcats)
	print(knnc.confusion_matrix_str(confmtx))

	# dtest.addColumn("Category", "numeric", newcats.T.A[0])
	#
	# dtest.write("own_data_knn_result.csv", headers=dtest.get_headers())

	return


if __name__ == "__main__":
	main(sys.argv)