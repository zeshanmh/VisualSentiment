import numpy as np 
import pandas as pd 
import sys 

def get_label_matrix(path, labels=['Interaction', 'Focus', 'Happiness']): 
	# path = '../../data/image_annotations.csv'
	pand_arr = pd.read_csv(path) 	
	label_matrix = np.vstack((pand_arr['Interaction'].as_matrix(), \
		pand_arr['Focus'].as_matrix(), pand_arr['Happiness'].as_matrix()))
	return label_matrix.T


def get_filename_list(path): 
	filenames = open(path, 'r')
	# filenames = open('../../data/groupdataset_release/file_names.txt', 'r')
	flist = filenames.readlines()
	return flist[0].split('\r')

def reduce_labels(): 
	orig_img_path = '../../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt'
	orig_labels_path = '../../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt'
	orig_img_file = open(orig_img_path, 'r')
	orig_labels_file = open(orig_labels_path, 'r') 

	img_labels_dict = {}
	orig_imgs = orig_img_file.readlines()
	orig_imgs = [filename.strip() for filename in orig_imgs]

	lines = orig_labels_file.readlines()
	labels = []
	for line in lines: 
		labels.append(int(line.split()[0]))

	for i,label in enumerate(labels): 
		img_labels_dict[orig_imgs[i]] = label

	new_labels_file = open('../../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels_Reduced.txt', 'w')
	new_imgs_file = open('../../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images_Reduced.txt', 'r') 

	new_imgs = new_imgs_file.readlines()
	new_imgs = [filename.strip() for filename in new_imgs]

	for img in new_imgs: 
		img_label = img_labels_dict[img]
		new_labels_file.write(str(img_label) + '\n')


if __name__ == '__main__':
    reduce_labels()
