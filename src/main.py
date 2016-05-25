import sys
import os
import cv2
sys.path.insert(0, './util')

from FaceExtractor import FaceExtractor
from EmotionExtractor import EmotionExtractor

def main():
	# img_path = "../data/groupdataset_release/images/Library3.jpg"

	# face_extractor = FaceExtractor()
	# faces_lists, image = face_extractor.detect_faces(img_path)

	img_path = "../data/GENKI-R2009a/Subsets/GENKI-4K/files/file0001.jpg"
	emotion_extractor = EmotionExtractor()
	face_image = cv2.imread(img_path)
	# for face_list in faces_lists: 
	# 	for (x,y,w,h) in face_list: 
	# face_image = image[x:x+w,y:y+h]
	emotion_extractor.set_face(face_image)
	emotion_extractor.extract_smile()

	# 		break
	# 	break



if __name__ == '__main__':
	main()
