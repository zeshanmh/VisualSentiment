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
def create_filename_orient_dict(tud_path='../../../data/TUD', tud_type = 'train'):
	# build the train dict
	# train_dict = {}
	img_folder = os.path.join(tud_path, tud_type)
	img_filenames = [f for f in os.listdir(img_folder) if 'DS_Store' not in f]
	print len(img_filenames)

	# get corresponding training annotations
	annotation_type = '_' + tud_type
	train_annotation_files = [f for f in os.listdir(tud_path) if annotation_type in f]
	# print train_annotation_files

	all_orientations = []
	for file in train_annotation_files:
		fullpath = os.path.join(tud_path,file)
		# annotation_xml = xml.etree.ElementTree.parse(fullpath).getroot()
		annotation_xml = minidom.parse(fullpath)
		sil_list = annotation_xml.getElementsByTagName('silhouette')
		for sil in sil_list:
			orientation = sil.getElementsByTagName('id')
			orientation = orientation[0]
			orientation = orientation.firstChild.nodeValue
			all_orientations.append(orientation)

	# all_images = []
	# for filename in img_filenames:
	# 	img_path = os.path.join(img_folder, filename)
	# 	img = cv2.imread(img_path)
	# 	all_images.append(img)
		# cv2.imshow("Faces found", img)
		# cv2.waitKey(0)
	return dict(zip(img_filenames, all_orientations))


	# build the test dict



if __name__ == '__main__':
    create_filename_orient_dict()