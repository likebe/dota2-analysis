# knn_process.py
# Kebing Li
# Fall 2018
# CS251

import data
import classifiers
import sys
import numpy

# For task2, write a file to process KNN classifier
# Similar to what the test function is in classifiers.py
def process(traind, trainc, testd, testc, write = True, K=10):
	filename = "results_knn.csv"
	print ("Reading data")

	train_file = traind
	test_file = testd
	dtrain = data.Data(train_file)
	dtest = data.Data(test_file)

	train_headers = dtrain.get_headers()
	test_headers = dtrain.get_headers()

	traincat_file = trainc
	testcat_file = testc

	traincats = data.Data(traincat_file)
	traincatdata = traincats.all_rows_specified_columns(traincats.get_headers())

	testcats = data.Data(testcat_file)
	testcatdata = testcats.all_rows_specified_columns(testcats.get_headers())

	uniquelabels, correctedtraincats = numpy.unique(traincatdata.T.tolist()[0], return_inverse=True)
	correctedtraincats = numpy.matrix([correctedtraincats]).T

	uniquelabels, correctedtestcats = numpy.unique(testcatdata.T.tolist()[0], return_inverse=True)
	correctedtestcats = numpy.matrix([correctedtestcats]).T

	print('Building KNN Classifier')
	knnc = classifiers.KNN(dtrain, train_headers, traincatdata, K)

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

	dtest.addColumn("Category", "numeric", newcats.T.A[0])

	# if you want to write the test results in a csv file
	if write:
		dtest.write(filename, headers = dtest.get_headers())
	
	return dtest


if __name__ == '__main__':
	process(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])