import numpy as np 
import os
import cv2

from xml.dom import minidom
# from bs4 import BeautifulSoup

# class TUDAnnotationHandler(xml.sax.ContentHandler):
# 	def __init__(self):
# 		self.CurrentData = ""
# 		self.annotation = ""
# 		self.image = ""
# 		self.annorect = ""
# 		self.x1 = ""
# 		self.y1 = ""

# 		self.x2 = ""
# 		self.y2 = ""
# 		self.annopoints = ""
# 		self.point = ""
# 		self.id = ""
# 		self.x = ""
# 		self.y = "" 
def get_labels_pictures(tud_path='../../../data/TUD'):
	# build the train dict
	train_dict = {}
	train_folder = os.path.join(tud_path, 'train')
	train_filenames = [f for f in os.listdir(train_folder) if 'DS_Store' not in f]

	# get corresponding training annotations
	train_annotation_files = [f for f in os.listdir(tud_path) if '_train' in f]
	print train_annotation_files

	tree = []
	for file in train_annotation_files:
		fullpath = os.path.join(tud_path,file)
		# annotation_xml = xml.etree.ElementTree.parse(fullpath).getroot()
		annotation_xml = minidom.parse(fullpath)
		sil_list = annotation_xml.getElementsByTagName('silhouette')
		print len(sil_list)
		for sil in sil_list:
			orientation = sil.getElementsByTagName('id')
			orientation = orientation[0]
			print orientation.firstChild.nodeValue
		pass
		# tree.extend(file.readlines())

	# print tree

	# for filename in train_filenames:
	# 	img_path = os.path.join(train_folder, filename)
	# 	print img_path
	# 	img = cv2.imread(img_path)
	# 	cv2.imshow("Faces found", img)
	# 	cv2.waitKey(0)
	# print train_filenames



	# build the test dict



if __name__ == '__main__':
    get_labels_pictures()