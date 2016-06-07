import cv2
import os
import shutil
import scipy.io
import numpy as np
import copy

from FaceExtractor import FaceExtractor 
from TorsoExtractor import TorsoExtractor
from SilhouetteExtractor import SilhouetteExtractor
from poselet_util import *
from feature_extractors import *
from natsort import natsorted

def extract_save_group_faces(img_path, dest_path):
	face_extractor = FaceExtractor()
	filenames = os.listdir(img_path)

	# remove all existing files from dest_path
	remove_files_from_directory(dest_path)

	for j, filename in enumerate(filenames):
		# print filename
		print 'Extracting faces from image number:', j
		if filename == '.DS_Store': 
			continue

		full_path = os.path.join(img_path, filename)
		# print full_path
		img = cv2.imread(full_path)
		face_list = getFacesFromHeadPoselets(full_path)

		# for face in face_list:
		# 	print face
		# 	x, y, w, h = face
		# 	cv2.rectangle(img, (x, y), (int(x+w), int(y+h)), (0, 255, 0), 2)
		# 	cv2.imshow("Faces found", img)
		# 	cv2.waitKey(0)


		# face_lists, img = face_extractor.detect_faces(img)

		# form matrix of all faces from all classifiers
		# face_list = [face for face_list in face_lists for face in face_list]
		# face_matrix = np.array(face_list)

		modified_path = filename[:-4]
		new_fold = os.path.join(dest_path, modified_path)
		if not os.path.exists(new_fold):
			os.makedirs(new_fold)

		bb_path = os.path.join(new_fold, 'face_bbs')

		face_bbs = []
		for i, face in enumerate(face_list):
			face_bbs.append(face)
			x, y, w, h = face
			face_window = img[y:y+h,x:x+w]
			face_name = 'face' + str(i) + '.jpg'
			cv2.imwrite(os.path.join(new_fold, face_name), face_window)

		np.save(bb_path, face_bbs)

		clean_duplicate_faces(new_fold)

# def extract_save_group_faces(img_path, dest_path):
# 	face_extractor = FaceExtractor()
# 	filenames = os.listdir(img_path)

# 	# remove all existing files from dest_path
# 	remove_files_from_directory(dest_path)

# 	for j, filename in enumerate(filenames):
# 		# print filename
# 		if filename == '.DS_Store': 
# 			continue

# 		full_path = os.path.join(img_path, filename)
# 		face_lists, img = face_extractor.detect_faces(full_path)

# 		# form matrix of all faces from all classifiers
# 		face_list = [face for face_list in face_lists for face in face_list]
# 		face_matrix = np.array(face_list)

# 		modified_path = filename[:-4]
# 		new_fold = os.path.join(dest_path, modified_path)
# 		if not os.path.exists(new_fold):
# 			os.makedirs(new_fold)

# 		bb_path = os.path.join(new_fold, 'face_bbs')
# 		np.save(bb_path, face_matrix)

# 		for i, face in enumerate(face_list):
# 			x, y, w, h = face
# 			face_window = img[y:y+h,x:x+w]
# 			face_name = 'face' + str(i) + '.jpg'
# 			cv2.imwrite(os.path.join(new_fold, face_name), face_window)

# # TODO: remove poselets
# def clean_all_faces(faces_path):
# 	dirnames = os.listdir(faces_path)
# 	for i, dirname in enumerate(dirnames):
# 		if dirname == '.DS_Store': 
# 			continue

# 		img_folder = os.path.join(faces_path, dirname)
# 		# poselets = poselet_dict[dirname]
# 		# clean_faces(img_folder, poselets)
# 		clean_duplicate_faces(img_folder)



# def clean_faces(img_folder, poselets):
# 	bb_path = os.path.join(img_folder, 'face_bbs.npy')
# 	bbs = np.load(bb_path)
# 	new_bbs = np.zeros_like(bbs)
# 	counter = 0

# 	for i in xrange(bbs.shape[0]):
# 		face = bbs[i,:]
# 		poselet_found = False
# 		for j in xrange(poselets.shape[0]):
# 			poselet = poselets[j,:]
# 			if contained_in(face, poselet):
# 				new_bbs[counter,:] = face
# 				counter += 1
# 				poselet_found = True
# 				break
# 		if not poselet_found:
# 			face_path = os.path.join(img_folder, 'face' + str(i) + '.jpg')
# 			if os.path.exists(face_path):
# 				os.remove(face_path)

# 	bbs = new_bbs[:counter,:]
# 	np.save(bb_path, bbs)

def clean_duplicate_faces(img_folder):
	bb_path = os.path.join(img_folder, 'face_bbs.npy')
	bbs = np.load(bb_path)
	filenames = [f for f in os.listdir(img_folder) if 'bb' not in f and 'DS_Store' not in f]
	# print img_folder
	# new_faces = []
	# new_bbs = np.zeros_like(bbs)
	# counter = 0
	# print type(filenames)
	filenames = natsorted(filenames)
	# print filenames
	# print len(filenames)


	filtered_filenames = copy.deepcopy(filenames)

	to_remove_indices = set()
	for i, face_bb in enumerate(bbs):
		for k in range(i+1, len(bbs)):
			face_bb2 = bbs[k]
			if center_contained_in(face_bb, face_bb2):
				to_remove = i
				# remove smaller one from the list
				if calc_area(face_bb) > calc_area(face_bb2):
					to_remove = k
				# remove
				to_remove_indices.add(to_remove)
	# print to_remove_indices

	# add any file that should not be removed
	filtered_bbs = []
	filtered_filenames = []
	for i, face_bb in enumerate(bbs):
		if i not in to_remove_indices:
			filtered_bbs.append(face_bb)
			filtered_filenames.append(filenames[i])

	# print len(filtered_bbs)
	# print filtered_filenames

	# return


	# get rid of files that should
	filtered_faces = []
	for i, filename in enumerate(filtered_filenames):
		img_path = os.path.join(img_folder, filename)
		face_img = cv2.imread(img_path)
		filtered_faces.append(face_img)


	# loop over pairs of duplicates and add non repeaters to new_faces
	# for i in xrange(len(filenames)):
	# 	face_bbi = bbs[i,:]
	# 	found_duplicate = False
	# 	for j in xrange(len(filenames)):
	# 		if i == j:
	# 			continue
	# 		face_bbj = bbs[j,:]

	# 		if center_contained_in(face_bbi, face_bbj):
	# 			found_duplicate = True
	# 			break
	# 	if not found_duplicate:
	# 		img_path = os.path.join(img_folder, 'face' + str(i) + '.jpg')
	# 		face_img = cv2.imread(img_path)
	# 		new_faces.append(face_img)
	# 		new_bbs[counter,:] = face_bbi
	# 		counter += 1

	# # print len(new_faces)
	# clear all existing files
	remove_files_from_directory(img_folder)

	# #replace bbs
	# bbs = new_bbs[:counter,:]
	np.save(bb_path, filtered_bbs)

	# replace face files
	for i in xrange(len(filtered_faces)):
		face_name = 'face' + str(i) + '.jpg'
		face_path = os.path.join(img_folder, face_name)
		cv2.imwrite(face_path, filtered_faces[i])


def remove_files_from_directory(path):
	for the_file in os.listdir(path):
		file_path = os.path.join(path, the_file)
		if os.path.isfile(file_path):
			os.unlink(file_path)
		elif os.path.isdir(file_path):
			shutil.rmtree(file_path)

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
		face_lists, image = face_extractor.detect_faces_GENKI(full_path)

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

def get_face_matrix_from_matching_list(matching_list):
	face_matrix = np.zeros((len(matching_list), 4))
	counter = 0

	for match in matching_list:
		if match[1] != None:
			face_matrix[counter,:] = match[1]
			counter += 1

	face_matrix = face_matrix[:counter,:]
	return face_matrix

def clean_all_faces_with_bb_matching(imgs_path, person_bb_path, faces_path, torsos_path):
	img_folders = [folder for folder in os.listdir(faces_path) if 'DS_Store' not in folder]

	for i, folder in enumerate(img_folders):
		print folder
		img_path = os.path.join(imgs_path, folder + '.jpg')
		img = cv2.imread(img_path)
		ppl_bbs = np.array(get_bbs(person_bb_path, folder + '.jpg'))
		faces_folder = os.path.join(faces_path, folder)
		face_bb_path = os.path.join(faces_folder, 'face_bbs.npy')
		face_bbs = np.load(face_bb_path)
		torso_bbs = np.genfromtxt(os.path.join(torsos_path, folder + '_torsos.csv'), delimiter=',')
		torso_bbs = torso_bbs[:,:4]
		matching_list = bb_matching(img, ppl_bbs, face_bbs, torso_bbs)

		# remove all files in folder, replace face_bbs
		remove_files_from_directory(faces_folder)
		face_bbs = get_face_matrix_from_matching_list(matching_list)
		np.save(face_bb_path, face_bbs)

		# save remaining faces
		for i in xrange(face_bbs.shape[0]):
			x, y, w, h = face_bbs[i,:]
			face = img[int(y):int(y+h),int(x):int(x+w)]
			face_path = os.path.join(faces_folder, 'face' + str(i) + '.jpg')
			cv2.imwrite(face_path, face)



# assumes params are num_bbs x 4 np arrays
# returns list of list of tuples of coordinates
def bb_matching(img, ppl_bbs, face_bbs, torso_bbs):
	ppl_bbs = ppl_bbs.astype(int)
	# if face_bbs != None:
	face_bbs = face_bbs.astype(int)
	torso_bbs = torso_bbs.astype(int)

	print 'num people: ', ppl_bbs.shape[0]
	# if face_bbs != None:
	print 'num faces: ', face_bbs.shape[0]
	print 'num torsos: ', torso_bbs.shape[0]


	ppl_bbs = ppl_bbs.tolist()
	# if face_bbs != None:
	face_bbs = face_bbs.tolist()
	# else:
	# 	face_bbs = []
	torso_bbs = torso_bbs.tolist()

	matching_list = []
	bb_to_face = {}
	bb_to_torso = {}

	unmatched_ppl = copy.copy(ppl_bbs)
	unmatched_faces = copy.copy(face_bbs)

	# match people and faces
	for i, face in enumerate(face_bbs):

		best_person = None
		best_dist = float('inf')

		for person in unmatched_ppl:

			px, py, pw, ph = person

			if contained_in(face, person) and upper_half(face, person):
				dist = compute_face_dist(face, person)

				if dist < best_dist:
					best_person = person
					best_dist = dist

		# print unmatched_ppl
		if best_person != None:
			# print best_person
			bb_to_face[tuple(best_person)] = tuple(face)
			unmatched_ppl.remove(best_person)
			unmatched_faces.remove(face)
			px, py, pw, ph = best_person
			fx, fy, fw, fh = face
			
	# sort bbs by size
	bbs_w_sizes = [(bb, bb[2] * bb[3]) for bb in ppl_bbs]
	sorted_bbs_w_sizes = sorted(bbs_w_sizes, key=lambda bb_tup: bb_tup[1])
	sorted_bbs = [bb_tup[0] for bb_tup in sorted_bbs_w_sizes]

	torsos_w_sizes = [(torso, torso[2] * torso[3]) for torso in torso_bbs]
	sorted_torsos_w_sizes = sorted(torsos_w_sizes, key=lambda torso_tup: torso_tup[1])
	sorted_torsos = [torso_tup[0] for torso_tup in sorted_torsos_w_sizes]
	unmatched_torsos = copy.copy(sorted_torsos)

	# match people and torsos
	for person in sorted_bbs:
		best_torso = None

		for torso in unmatched_torsos:
			if contained_in(torso, person):
				best_torso = torso


		if best_torso != None:
			bb_to_torso[tuple(person)] = tuple(best_torso)
			if person in unmatched_ppl:
				unmatched_ppl.remove(person)
			unmatched_torsos.remove(best_torso)

	# combine matches
	keys = list(set(bb_to_face.keys()).union(set(bb_to_torso.keys())))
	for person in keys:
		bb_list = [person, None, None]
		if person in bb_to_face:
			bb_list[1] = bb_to_face[person]
		if person in bb_to_torso:
			bb_list[2] = bb_to_torso[person]
		matching_list.append(bb_list)

	return matching_list

def center_contained_in(bb1, bb2):
	x1, y1, w1, h1 = bb1
	x2, y2, w2, h2 = bb2
	cent1x = x1 + w1 / 2
	cent1y = y1 + h1 / 2
	return (cent1x >= x2) and (cent1x <= x2 + w2) and (cent1y >= y2) and (cent1y <= y2 + h2)

def contained_in(bb_small, bb_big):
	# print bb_small
	sx, sy, sw, sh = bb_small
	bx, by, bw, bh = bb_big
	return (sx >= bx) and (sx+sw <= bx+bw) and (sy >= by) and (sy+sh <= by+bh)

def upper_half(face, bb):
	fx, fy, fw, fh = face
	bx, by, bw, bh = bb
	return (fy+fh <= by+ (bh / 2))

def compute_face_dist(face, bb):
	fx, fy, fw, fh = face
	bx, by, bw, bh = bb
	f_cent = np.array([fy, fx + fw / 2])
	b_cent = np.array([by, bx + bw / 2])
	return np.linalg.norm(f_cent - b_cent)

def compute_torso_dist(torso, bb):
	tx, ty, tw, th = torso
	bx, by, bw, bh = bb
	t_cent = np.array([ty + th / 2, tx + tw / 2])
	b_cent = np.array([by + bh / 2, bx + bw / 2])
	return np.linalg.norm(t_cent - b_cent)

def calc_area(bb1):
	x, y, w, h = bb1
	return w * h

def extract_group_bbs(images_path): 
	img_names = os.listdir(images_path)
	img_names = [img_name.strip() for img_name in img_names]

	face_extractor = FaceExtractor()
	torso_extractor = TorsoExtractor()
	for img_name in img_names: 
		pathname = os.path.join(images_path, img_name)
		faces_lists, _ = face_extractor.detect_faces(pathname)
		img_face_list = [face for face_list in faces_lists for face in face_list]
		img_torso_list = torso_extractor.detect_torso(pathname)
		

if __name__ == '__main__':

	imgs_path = '../../data/groupdataset_release/images'
	person_bb_path = '../../data/groupdataset_release/annotations/all'
	faces_path = '../../data/groupdataset_release/faces'
	torsos_path = '../../data/groupdataset_release/all_torsos_hq'
	clean_all_faces_with_bb_matching(imgs_path, person_bb_path, faces_path, torsos_path)
	# extract_save_group_faces('../../data/groupdataset_release/images', '../../data/groupdataset_release/faces')
	# extract_save_group_faces('../../data/groupdataset_release/images/all', '../../data/groupdataset_release/faces')
	# clean_duplicate_faces('../../data/groupdataset_release/faces/13-03-08-classroom-2')
	# clean_all_faces('../../data/groupdataset_release/faces', poselet_dict)
	# img_path = '../../data/groupdataset_release/images'
	# filename = '01-breeze-outdoor-dining.jpg'
	# full_path = os.path.join(img_path, filename)
	# face_extractor = FaceExtractor()
	# face_lists, img = face_extractor.detect_faces(full_path)
	# face_list = [face for face_list in face_lists for face in face_list]
	# faces = np.array(face_list)
	# # img = cv2.imread('../../data/groupdataset_release/images/01-breeze-outdoor-dining.jpg')
	# torsos = np.genfromtxt('/Users/hardiecate/Downloads/all_torsos/01-breeze-outdoor-dining_torsos.csv', delimiter=',')
	# torsos = torsos[:,:4]
	# bb_path = '../../data/groupdataset_release/annotations/all'
	# people = np.array(get_bbs(bb_path, filename))
	# matched_list = bb_matching(img, people, faces, torsos)
	# # for i, match in enumerate(matched_list):
	# # 	# print match
	# # 	color = (80 * i) % 256
	# # 	if match[1] != None and match[2] != None:
	# # 		cv2.rectangle(img, (match[0][0], match[0][1]), (match[0][0]+match[0][2], match[0][1]+match[0][3]), (0, color, 0), 2)
	# # 		cv2.rectangle(img, (match[1][0], match[1][1]), (match[1][0]+match[1][2], match[1][1]+match[1][3]), (0, color, 0), 2)
	# # 		cv2.rectangle(img, (match[2][0], match[2][1]), (match[2][0]+match[2][2], match[2][1]+match[2][3]), (0, color, 0), 2)
	# # 	elif match[1] == None and match[2] != None:
	# # 		cv2.rectangle(img, (match[0][0], match[0][1]), (match[0][0]+match[0][2], match[0][1]+match[0][3]), (0, color, 0), 2)
	# # 		cv2.rectangle(img, (match[2][0], match[2][1]), (match[2][0]+match[2][2], match[2][1]+match[2][3]), (0, color, 0), 2)
	# # 	elif match[1] != None and match[2] == None:
	# # 		cv2.rectangle(img, (match[0][0], match[0][1]), (match[0][0]+match[0][2], match[0][1]+match[0][3]), (0, color, 0), 2)
	# # 		cv2.rectangle(img, (match[1][0], match[1][1]), (match[1][0]+match[1][2], match[1][1]+match[1][3]), (0, color, 0), 2)

	# # cv2.imshow('People!', img)
	# # cv2.waitKey(0)
	# sil_extractor = SilhouetteExtractor()
	# sils = sil_extractor.get_silhouettes(img, matched_list)
	# for sil in sils:
	# 	cv2.imshow('Silhouette!', sil)
	# 	cv2.waitKey(0)







