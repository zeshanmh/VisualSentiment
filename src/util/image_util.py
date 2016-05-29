import cv2
import os
import scipy.io
from FaceExtractor import FaceExtractor 

def extract_GENKI_faces(img_path, dest_path):
	face_extractor = FaceExtractor()

	# for each file in this image path
	filenames = os.listdir(img_path)

	# error counters
	none_cntr = 0

	for filename in filenames:
		if filename == '.DS_Store': 
			continue
		full_path = os.path.join(img_path, filename)
		face_lists, image = face_extractor.detect_faces(full_path)

		if sum([len(x) for x in face_lists]) == 0: 
			print "No face found in image:", full_path
			none_cntr += 1
			continue

		elif sum([len(x) for x in face_lists]) > 1:
			print "More than one face in image:", full_path


		for face_list in face_lists:
			# save each face in the list as image
			for i, face in enumerate(face_list):
				x,y,w,h = face
				face_image = image[y:y+h,x:x+w]
				cv2.imwrite(os.path.join(dest_path, filename[:-4] \
					+ "_face" + str(i) + '.jpg'), face_image)

	print "Number of times no face found per image per classifier:", none_cntr 

