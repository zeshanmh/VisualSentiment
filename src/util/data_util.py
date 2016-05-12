import numpy as np 
import pandas as pd 
import sys 

def get_label_matrix(labels=['Interaction', 'Focus', 'Happiness']): 
	path = '../../data/image_annotations.csv'
	pand_arr = pd.read_csv(path) 

	label_matrix = np.vstack((pand_arr['Interaction'].as_matrix(), \
		pand_arr['Focus'].as_matrix(), pand_arr['Happiness'].as_matrix()))
	return label_matrix.T


if __name__ == '__main__':
    get_label_matrix()