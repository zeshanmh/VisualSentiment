import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

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


def plot_confusion_matrix(cm, class_names, sentiment): 
	"""
	Plots a confusion matrix. 

	Args:
		cm: confusion matrix
		class_names: a list of class names where the index of the class
		name in the list corresponds to how it was labeled in the models

	Returns: 
		None
	"""
	##plotting unnormalized confusion matrix
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('confusion matrix of ' + sentiment + ' classification')

	#uncomment for actual labels 
	# tick_marks = np.arange(len(class_names))
	# plt.xticks(tick_marks, class_names, rotation=90, fontsize=5)
	# plt.yticks(tick_marks, class_names, fontsize=5)

	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.show()


def run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, sentiment, ablation=False): 
	"""
	Runs analyses, including finding error, precision, recall, f1score, plotting
	a confusion matrix, on the results of a particular model. Prints out the numeric
	metrics and plots the graphical ones.

	Args:
		y_predict_train: 
			the predicted labels on the training examples
		y_train: 
			true labels on training examples
		y_predict: 
			predicted labels on testing examples
		y_test: 
			true labels on testing examples
		class_names: 
			dictionary that contains the class name that corresponds
			with the class index 

	Returns: 
		None
	"""
	# calculate metrics
	_, training_error = output_error(y_predict_train, y_train)
	(precision, recall, f1, _), testing_error = output_error(y_predict, y_test)
	class_names_list = [class_names[index] for index in class_names.keys()]
	if not ablation: 
		cm = metrics.confusion_matrix(y_test, y_predict)
		plot_confusion_matrix(cm, class_names_list, sentiment)

	# print out metrics
	print 'Average Precision:', np.average(precision)
	print 'Average Recall:', np.average(recall)
	print 'Average F1:', np.average(f1)
	print 'Training Error:', training_error
	print 'Testing Error:', testing_error


