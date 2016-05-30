import cv2
import os
import scipy.io
from FaceExtractor import FaceExtractor



def extract_GENKI_faces(img_path, dest_path):
	# save images where no faces where found to a file
	# no_faces_file_path = 'GENKI_faces_looser_bounds/no_faces_found_images'
	# no_faces_file_path = os.path.join(dest_path, no_faces_file_path)
	# no_faces_file = open(no_faces_file_path, 'w')

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
			# no_faces_file.write(full_path + '\n')
			# no_faces_file.close()
			none_cntr += 1

		# elif sum([len(x) for x in face_lists]) > 1:
			# print "More than one face in image:", full_path


		for face_list in face_lists:
			# save each face in the list as image
			for i, face in enumerate(face_list):
				x,y,w,h = face
				face_image = image[y:y+h,x:x+w]
				cv2.imwrite(os.path.join(dest_path, filename[:-4] \
					+ "_face" + str(i) + '.jpg'), face_image)

	print "Number of times no face found per image per classifier:", none_cntr 
	# no_faces_file.close()

# def extract_missed_faces(dest_path):
# 	face_extractor = FaceExtractor()

# 	# get all full paths for images with no faces originally found
# 	fullpaths = []
# 	no_faces_file_path = 'no_faces_found_images'
# 	with open(os.path.join(dest_path, no_faces_file_path)) as f:
# 		fullpaths = f.readlines()

# 	# error counters
# 	none_cntr = 0

# 	# for each fullpath detect faces
# 	for fullpath in fullpaths:
# 		fullpath = os.path.join('../', fullpath)
# 		print fullpath 
# 		face_lists, image = face_extractor.detect_faces(fullpath)

# 		if sum([len(x) for x in face_lists]) == 0: 
# 			print "Still no faces found in image:", full_path
# 			no_faces_file.write(full_path + '\n')
# 			none_cntr += 1
# 			continue

# 	for face_list in face_lists:
# 			# save each face in the list as image
# 			for i, face in enumerate(face_list):
# 				x,y,w,h = face
# 				face_image = image[y:y+h,x:x+w]
# 				cv2.imwrite(os.path.join(dest_path, filename[:-4] \
# 					+ "_face" + str(i) + '.jpg'), face_image)

# 	print "Number of times no face found per image per classifier:", none_cntr 




