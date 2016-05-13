import numpy as np 
import sklearn
import cv 
import sys
import analysis 
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
		X = bb_extractor('../data/groupdataset_release/annotations/all', img_names)
	else:
		pass
	Y = get_label_matrix('../data/groupdataset_release/image_annotations.csv')

	#split into train and test 
	print "Splitting into train and test set..."
	X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.2)

	#initialize svm
	class_names = {'none': 1, 'low': 2, 'moderate': 3, 'high': 4}
	sentiments = ['interaction', 'focus', 'happiness']
	for i in xrange(Y_train.shape[1]):
		print "Fitting svm...."
		svm_model = svm.SVC(kernel="linear", decision_function_shape='ovr', max_iter=10000)
		svm_model.fit(X_train, Y_train[:,i])
		print "Predicting..."
		y_predict_train = svm_model.predict(X_train)
		y_predict = svm_model.predict(X_test)

		analysis.run_analyses(y_predict_train, Y_train[:,i], y_predict, Y_test[:,i], class_names, sentiments[i])
		# _, train_error = analysis.output_error(y_predict_train, Y_train[:,i])
		# _, test_error = analysis.output_error(y_predict, Y_test[:,i])

		# print "Training Error:", train_error 
		# print "Testing Error:", test_error 


if __name__ == '__main__':
	baseline(setting='bb')
	
