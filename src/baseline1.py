import numpy as np 
import sklearn
import cv 
import sys 
sys.path.insert(0, './util')

from data_util import *
from feature_extractors import *
from sklearn import svm
from sklearn import metrics


def baseline(setting='color'):
	
	#get features and labels 
	img_names = get_filename_list('../data/groupdataset_release/file_names.txt')

	print "Extracting features..."
	if setting == 'color': 
		X = color_histogram('../data/groupdataset_release/images', img_names)
	elif setting == 'pixel':
		X = pixel_extractor('../data/groupdataset_release/resize_images', img_names)
	elif setting == 'bb':
		X = bb_extractor('../data/groupdataset_release/images', img_names)
	else:
		pass
	Y = get_label_matrix('../data/groupdataset_release/image_annotations.csv')
	y_interact = Y[:,0]
	y_focus = Y[:,1]
	y_happ = Y[:,2]

	#split into train and test 
	print "Splitting into train and test set..."
	X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.2)

	#initialize svm
	for i in xrange(Y_train.shape[1]):
		print "Fitting svm...."
		svm_model = svm.SVC(kernel="linear", decision_function_shape='ovr')
		svm_model.fit(X_train, Y_train[:,i])
		print "Predicting..."
		y_predict_train = svm_model.predict(X_train)
		y_predict = svm_model.predict(X_test)

		_, train_error = output_error(y_predict_train, Y_train[:,i])
		_, test_error = output_error(y_predict, Y_test[:,i])

		print "Training Error:", train_error 
		print "Testing Error:", test_error 


def output_error(y_predict, y_true): 
	"""
	Outputs several performance metrics of a given model, including precision, 
	recall, f1score, and error.

	Args:
		y_predict: an array of the predicted labels of the examples 
		y_true: an array of the true labels of the examples

	Returns
		(precision, recall, fscore, _), error 
	"""
	return metrics.precision_recall_fscore_support(y_true, y_predict), np.sum(y_predict != y_true) / float(y_predict.shape[0])
	

if __name__ == '__main__':
	baseline()

