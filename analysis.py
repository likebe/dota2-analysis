## analysis.py
## Kebing Li
## CS251 Project2
## Feb.25.2018

import data as dt
import numpy
import math
import scipy
from scipy import stats
import scipy.cluster.vq as vq
import random


# Takes in a list of column headers and the Data object and
# returns a list of 2-element lists with the minimum and maximum values for each column.
# The function is required to work only on numeric data types.
# def data_range(headers, file):
# 	thing = []
# 	data_object = dt.Data(file)
# 	for i in range(len(headers)):
# 		column = data_object.all_rows_specified_columns([headers[i]])
# 		minimum = column.min(0)
# 		maximum = column.max(0)
# 		sublist = [maximum,minimum]
# 		thing.append(sublist)
# 	#
# 	# if len(headers) == 1:
# 	# 	matrix = thing[0]
# 	#
# 	# else:
# 	# 	for i in range(len(headers) - 1):
# 	# 		matrix = numpy.hstack((thing[i], thing[i + 1])).transpose()
# 	return thing

def data_range(headers, d):
	thing = []
	for i in range(len(headers)):
		column = d.all_rows_specified_columns([headers[i]])
		minimum = column.min(0)
		maximum = column.max(0)
		sublist = [maximum,minimum]
		thing.append(sublist)
	#
	# if len(headers) == 1:
	# 	matrix = thing[0]
	#
	# else:
	# 	for i in range(len(headers) - 1):
	# 		matrix = numpy.hstack((thing[i], thing[i + 1])).transpose()
	return thing



# takes in a list of column headers and the Data object and
# returns a list of the mean values for each column.
# def mean(headers, file):
#
# 	thing = []
# 	data_object = dt.Data(file)
# 	for header in headers:
# 		column = data_object.all_rows_specified_columns([header])
# 		mean = column.mean(0)
# 		thing.append(float(mean))
#
# 	return thing

def mean(headers, d):

	thing = []
	for header in headers:
		column = d.all_rows_specified_columns([header])
		mean = column.mean(0)
		thing.append(float(mean))

	return thing


# takes in a list of column headers and the Data object and
# returns a list of the standard deviation for each specified column.
# def stdev(headers, file):
# 	thing = []
# 	data_object = dt.Data(file)
# 	for header in headers:
# 		column = data_object.all_rows_specified_columns([header])
# 		std = column.std(0)
# 		thing.append(float(std))
#
# 	return thing
def stdev(headers, d):
	thing = []
	for header in headers:
		column = d.all_rows_specified_columns([header])
		std = column.std(0)
		thing.append(float(std))

	return thing


# Takes in a list of column headers and the Data object and
# returns a matrix with each column normalized so its minimum value is mapped to zero and
# its maximum value is mapped to 1.
# def normalize_columns_separately(headers, file):
#
# 	normalized_matrix = []
# 	data_object = dt.Data(file)
# 	for header in headers:
# 		column = data_object.all_rows_specified_columns([header])
# 		column.flatten()
# 		maximum = column.max(0)
# 		minimum = column.min(0)
# 		for i in range(len(column)):
# 			numpy.put(column, [i], [float ((column[i]-minimum)/(maximum-minimum))])
# 		normalized_matrix.append(column)
#
# 	if len(headers) == 1:
# 		matrix = numpy.matrix(normalized_matrix[0])
#
# 	else:
# 		for i in range(len(headers) - 1):
# 			normalized_matrix[i+1] = numpy.hstack((normalized_matrix[i], normalized_matrix[i + 1]))
# 		matrix = normalized_matrix[-1]
#
# 	return matrix

def normalize_columns_separately(headers, d):

	normalized_matrix = []
	for header in headers:
		column = d.all_rows_specified_columns([header])
		column.flatten()
		maximum = column.max(0)
		minimum = column.min(0)
		for i in range(len(column)):
			numpy.put(column, [i], [float ((column[i]-minimum)/(maximum-minimum))])
		normalized_matrix.append(column)

	if len(headers) == 1:
		matrix = numpy.matrix(normalized_matrix[0])

	else:
		for i in range(len(headers) - 1):
			normalized_matrix[i+1] = numpy.hstack((normalized_matrix[i], normalized_matrix[i + 1]))
		matrix = normalized_matrix[-1]

	return matrix

# Takes in a list of column headers and the Data object and
# returns a matrix with each entry normalized so that the minimum value
# (of all the data in this set of columns) is mapped to zero and its maximum value is mapped to 1.
# def normalize_columns_together(headers, file):
#
# 	normalized_matrix = []
# 	data_object = dt.Data(file)
# 	columns = data_object.all_rows_specified_columns(headers)
# 	maximum = columns.max()
# 	minimum = columns.min()
# 	for header in headers:
# 		column = data_object.all_rows_specified_columns([header])
# 		column.flatten()
# 		for i in range(len(column)):
# 			numpy.put(column, [i], [float ((column[i]-minimum)/(maximum-minimum))])
# 		normalized_matrix.append(column)
#
#
# 	if len(headers) == 1:
# 		matrix = normalized_matrix[0]
#
# 	else:
# 		for i in range(len(headers) - 1):
# 			matrix = numpy.hstack((normalized_matrix[i], normalized_matrix[i + 1]))
#
# 	return matrix

def normalize_columns_together(headers, d):
	normalized_matrix = []
	columns = d.all_rows_specified_columns(headers)
	maximum = columns.max()
	minimum = columns.min()
	for header in headers:
		column = d.all_rows_specified_columns([header])
		column.flatten()
		for i in range(len(column)):
			numpy.put(column, [i], [float ((column[i]-minimum)/(maximum-minimum))])
		normalized_matrix.append(column)

	if len(headers) == 1:
		matrix = normalized_matrix[0]


	else:
		for i in range(len(headers) - 1):
			normalized_matrix[i + 1] = numpy.hstack((normalized_matrix[i], normalized_matrix[i + 1]))
		matrix = normalized_matrix[-1]

	return matrix


# Takes in a list of column headers and the Data object and returns a list of the sum for each specified column.
def sum(headers, d):

	thing = []
	for header in headers:
		column = d.all_rows_specified_columns([header])
		sum = column.sum(0)
		thing.append(float(sum))

	return thing


def covariance(header1, header2, d):
	column1 = d.all_rows_specified_columns([header1])
	column2 = d.all_rows_specified_columns([header2])
	n = d.num_points()
	mean1 = mean([header1], d)[0]
	mean2 = mean([header2], d)[0]
	cov = 0
	for i in range(len(column1)):
		col_ele = float(column1[i])
		row_ele = float(column2[i])
		cov = cov +(col_ele-mean1)*(row_ele-mean2)
	cov = cov/n
	return cov

def corrlation(header1, header2, d):
	stdev_list = stdev([header1, header2], d)
	variance1 = stdev_list[0]*stdev_list[0]
	variance2 = stdev_list[1]*stdev_list[1]
	cov = covariance(header1,header2,d)
	corr = cov/math.sqrt(variance1*variance2)
	return corr

# do the linear regression and computes the statistics
def linear_regression(d, ind, dep):
	# assign to y the column of data for the dependent variable
	y = d.all_rows_specified_columns([dep])
	# assign to A the columns of data for the independent variables
	A = d.all_rows_specified_columns(ind)
	# add a column of 1's to A to represent the constant term in the
	#    regression equation.
	homo = []
	for i in range(A.shape[0]):
		homo.append(1.0)
	old = numpy.array(A.T)
	A = numpy.vstack((old, homo))
	A = A.T
	# assign to AAinv the result of calling numpy.linalg.inv( np.dot(A.T, A))
	AAinv = numpy.linalg.inv(numpy.dot(A.T, A))
	# assign to x the result of calling numpy.linalg.lstsq( A, y )
	x = numpy.linalg.lstsq(A, y, rcond=None)
	# assign to b the first element of x.
	b = x[0]
	# assign to N the number of data points (rows in y)
	N = y.shape[0]
	# assign to C the number of coefficients (rows in b)
	C = b.shape[0]
	# assign to df_e the value N-C,
	df_e = N - C
	# assign to df_r the value C-1
	df_r = C - 1
	# assign to error, the error of the model prediction.  Do this by
	#    taking the difference between the value to be predicted and
	#    the prediction. These are the vertical differences between the
	#    regression line and the data.
	#    y - numpy.dot(A, b)
	error = y - numpy.dot(A, b)
	# assign to sse, the sum squared error, which is the sum of the
	#    squares of the errors computed in the prior step, divided by the
	#    number of degrees of freedom of the error.  The result is a 1x1 matrix.
	#    numpy.dot(error.T, error) / df_e
	sse = numpy.dot(error.T, error) / df_e
	# assign to stderr, the standard error, which is the square root
	#    of the diagonals of the sum-squared error multiplied by the
	#    inverse covariance matrix of the data. This will be a Cx1 vector.
	#    numpy.sqrt( numpy.diagonal( sse[0, 0] * AAinv ) )
	stderr = numpy.sqrt(numpy.diagonal(sse[0, 0] * AAinv))
	# assign to t, the t-statistic for each independent variable by dividing
	#    each coefficient of the fit by the standard error.
	#    t = b.T / stderr
	t = b.T / stderr
	# assign to p, the probability of the coefficient indicating a
	#    random relationship (slope = 0). To do this we use the
	#    cumulative distribution function of the student-t distribution.
	#    Multiply by 2 to get the 2-sided tail.
	#    2*(1 - scipy.stats.t.cdf(abs(t), df_e))
	p = 2 * (1 - scipy.stats.t.cdf(abs(t), df_e))
	# assign to r2, the r^2 coefficient indicating the quality of the fit.
	#    1 - error.var() / y.var()
	r2 = 1 - error.var() / y.var()
	# Return the values of the fit (b), the sum-squared error, the
	#     R^2 fit quality, the t-statistic, and the probability of a
	#     random relationship.
	return b.A, sse.A[0][0], r2, t.A[0], p[0]


# This version uses SVD
def pca(d, headers, normalize=True):

	print(headers)
	if normalize:
		A = normalize_columns_separately(headers, d)
	else:
		A = d.all_rows_specified_columns(headers)

	m = numpy.mean(A, axis=0)
	D = A - m
	U,S,V = numpy.linalg.svd(D, full_matrices = False)
	N = D.shape[0]
	eigenvalues = S**2/(N-1)
	pmat = (V*D.T).T

	return dt.PCAData(pmat, V, eigenvalues, m, headers)

# numpy's built-in k-means function
def kmeans_numpy(d, headers, K, whiten=True):
	'''Takes in a Data object, a set of headers, and the number of clusters to create
	Computes and returns the codebook, codes, and representation error.
	'''

	A = d.all_rows_specified_columns(headers)
	W = vq.whiten(A)

	codebook, bookerror = vq.kmeans(W, K)
	codes, error = vq.vq(W, codebook)

	return codebook, codes, error

# Selects K random rows from the data matrix A and returns them as a matrix
# Enables initial means as inputs
##### extension 1 #####
def kmeans_init(A, K, initial_means = None):

	if initial_means == None:
		num_rows = A.shape[0]
		index_list = numpy.arange(num_rows)
		numpy.random.shuffle(index_list)
		first_k = index_list[:K]
		newrow = A
		for i in range(len(first_k)):
			row = A[first_k[i], :]
			if i == 0:
				newrow = row
			else:
				newrow = numpy.vstack((newrow, row))
		return newrow
	else:
		if type(initial_means) == list:
			return numpy.matrix(initial_means)
		else:
			return initial_means

# Given a data matrix A and a set of means in the codebook
# Returns a matrix of the id of the closest mean to each point
# Returns a matrix of the sum-squared distance between the closest mean and each point
def kmeans_classify(A, codebook, measurement = 2):
	num_rows = A.shape[0]
	num_means = codebook.shape[0]
	ID = []
	SSD = []
	id = 0
	nearest = 10000000000
	for i in range(num_rows):
		diff = codebook - A[i,:]
		for j in range(num_means):
			if measurement == 2:
				square = numpy.square(diff[j,:])
				sum = numpy.sum(square)
				ssd = math.sqrt(sum)
				if ssd < nearest:
					nearest = ssd
					id = j
			else:
				abs = numpy.abs(diff[j,:])
				sum = numpy.sum(abs)
				ssd = sum
				if ssd < nearest:
					nearest = ssd
					id = j
		ID.append(id)
		SSD.append(nearest)
		nearest = 10000000000000

	return ID, SSD

# Given a data matrix A and a set of K initial means, compute the optimal
# cluster means for the data and an ID and an error for each data point
def kmeans_algorithm(A, means, measurement = 2):
	# set up some useful constants
	MIN_CHANGE = 1e-7     # might want to make this an optional argument
	MAX_ITERATIONS = 100  # might want to make this an optional argument
	D = means.shape[1]    # number of dimensions
	K = means.shape[0]    # number of clusters
	N = A.shape[0]        # number of data points

	# iterate no more than MAX_ITERATIONS
	for i in range(MAX_ITERATIONS):
		# calculate the codes by calling kemans_classify
		# codes[j,0] is the id of the closest mean to point j
		codes = kmeans_classify(A, means)[0]

		# initialize newmeans to a zero matrix identical in size to means
		# Hint: look up the numpy function zeros_like
		newmeans = numpy.zeros_like(means)

		# initialize a K x 1 matrix counts to zeros (use the numpy zeros function)
		zeros = numpy.zeros(K)
		# for the number of data points
		for j in range(N):
			newmeans[codes[j],:] = newmeans[codes[j],:]+A[j,:]
			zeros[codes[j]] = zeros[codes[j]] + 1
			# add to the closest mean (row codes[j,0] of newmeans) the jth row of A
			# add one to the corresponding count for the closest mean

		# finish calculating the means, taking into account possible zero counts
		#for the number of clusters K
		for k in range(K):
			if zeros[k] != 0:
				newmeans[k,:] = newmeans[k,:]/zeros[k]
			else:
				num = random.randint(0,A.shape[0]-1)
				newmeans[k,:] = A[num,:]
			# if counts is not zero, divide the mean by its count
			# else pick a random data point to be the new cluster mean

		# test if the change is small enough and exit if it is
		diff = numpy.sum(numpy.square(means - newmeans))
		means = newmeans
		if diff < MIN_CHANGE:
			break

	# call kmeans_classify one more time with the final means
	codes, errors = kmeans_classify( A, means, measurement )

	# return the means, codes, and errors
	return (means, codes, errors)

# own version of kmeans clustering
def kmeans(d, headers, K, whiten=True, measurement = 2):
	'''Takes in a Data object, a set of headers, and the number of clusters to create
	Computes and returns the codebook, codes and representation errors.
	'''
	if isinstance(d, numpy.ndarray):
		A=d
	else:
		A = d.all_rows_specified_columns(headers)

	if whiten:
		W = vq.whiten(A)
	else:
		W = A

	best_codebook = None
	best_codes = []
	best_errors = []

	quality = 1000000000
	for i in range(10):
		codebook = kmeans_init(W, K)
		codebook, codes, errors = kmeans_algorithm(W, codebook, measurement)
		new_quality = kmeans_quality(errors, K)
		if new_quality < quality:
			quality = new_quality
			best_codebook = codebook
			best_codes = codes
			best_errors = errors

	return codebook, codes, errors

def kmeans_quality(errors, K):

	num_points = len(errors)
	numpy_errors = numpy.matrix(errors)
	errors_square = numpy.square(numpy_errors)
	sum = numpy.sum(errors_square)
	length = sum + math.log2(num_points)*K/2

	return length


def test():
	files = ['data-clean.csv', 'data-good.csv', 'data-noisy.csv']
	for file in files:
		data_obj = dt.Data(file)
		print(linear_regression(data_obj, ["X0", "X1"], "Y"))

def test2():
	data_obj = dt.Data("african_hiv_econ.csv")
	headers = data_obj.get_headers()
	print(data_obj.get_headers())
	print(linear_regression(data_obj, [headers[1], headers[6], headers[9], headers[3]], headers[0])[0])
	print(linear_regression(data_obj, [headers[1], headers[6], headers[9], headers[3]], headers[0])[1])
	print(linear_regression(data_obj, [headers[1], headers[6], headers[9], headers[3]], headers[0])[2])
	print(linear_regression(data_obj, [headers[1], headers[6], headers[9], headers[3]], headers[0])[3])
	print(linear_regression(data_obj, [headers[1], headers[6], headers[9], headers[3]], headers[0])[4])







if __name__ == "__main__":
	# test2()
	test()