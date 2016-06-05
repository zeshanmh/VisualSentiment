import cv2
import numpy as np

from feature_extractors import *

FACE_STRIPE_WIDTH_RATIO = 0.125
FACE_STRIPE_HEIGHT_RATIO = 0.9
TORSO_STRIPE_HEIGHT_RATIO = 0.4
TORSO_STRIPE_WIDTH_RATIO = 0.25
TORSO_SLIVER_HEIGHT_RATIO = 0.2
TORSO_SLIVER_WIDTH_RATIO = 0.1
SLIVER_OVERLAP = 0.2

NUM_GC_ITERATIONS = 5

class SilhouetteExtractor:

	def __init__(self):
		pass

	def get_silhouettes(self, img, bb_matching_list):
		silhouettes = []
		print bb_matching_list
		for match in bb_matching_list: 
			# print match
			# person, face, torso = match
			# mask = self.get_mask(img, person, face, torso).astype('uint8')
			# # mask = np.zeros((img.shape[0], img.shape[1]))
			# bg_model = np.zeros((1, 65), np.float64)
			# fg_model = np.zeros((1, 65), np.float64)
			# print np.sum(mask)
			# # bg_model = np.zeros((1, 65))
			# # fg_model = np.zeros((1, 65))
			# # cv2.imshow('img', img)
			# # cv2.waitKey(0)
			# print img.shape
			# print type(img)
			# print mask.shape
			# print type(mask)
			# img = img.astype('uint8')
			img = cv2.imread('../../data/groupdataset_release/images/01-breeze-outdoor-dining.jpg')
			mask = np.zeros(img.shape[:2],np.uint8)
			bgdModel = np.zeros((1,65), np.float64)
			fgdModel = np.zeros((1,65), np.float64)

			rect = (50,50,450,290)

			mask, _, _ = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
			# mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
			# new_img = img * mask[:,:,np.newaxis]
			# x, y, w, h = person
			# silhouette = new_img[y:y+h,x:x+w,:]
			# silhouettes.append(silhouette)

		return silhouettes
		# basepath1 = '../data/groupdataset_release/annotations/all'
		# basepath2 = '../data/groupdataset_release/images'

		# bbs = get_bbs(basepath1, img_name)
		# img = cv2.imread(os.path.join(basepath2, img_name))

		# for img_bb in bbs: 

		# 	mask = self.get_mask(img, img_bb, )

	def get_mask(self, img, img_bb, face_bb, torso_bb, draw=False):

		def draw_rect(rect, color, draw): 
			if rect != None and draw: 
				x, y, w, h = rect 
				cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

		x, y, w, h = img_bb
		if face_bb != None: x_face, y_face, w_face, h_face = face_bb
		if torso_bb != None: x_torso, y_torso, w_torso, h_torso = torso_bb
		mask = np.zeros((img.shape[0], img.shape[1]))

		# setting probable values
		mask[y:y+h,x:x+w] = cv2.GC_PR_BGD
		mask[y_face:y_face+h_face,x_face:x_face+w_face] = cv2.GC_PR_FGD
		mask[y_torso:y_torso+h_torso,x_torso:x_torso+w_torso] = cv2.GC_PR_FGD

		draw_rect(img_bb, (0,255,0), draw)
		draw_rect(face_bb, (0,0,255), draw)
		draw_rect(torso_bb, (0,0,255), draw)

		if face_bb != None: 
			# get indices of face stripe
			f_stripe_x = int(x_face + 0.5*(1 - FACE_STRIPE_WIDTH_RATIO) * w_face)
			f_stripe_y = int(y_face + 0.5*(1 - FACE_STRIPE_HEIGHT_RATIO) * h_face)
			f_stripe_w = int(FACE_STRIPE_WIDTH_RATIO * w_face)
			f_stripe_h = int(FACE_STRIPE_HEIGHT_RATIO * h_face)

			# set mask at face stripe to foreground
			mask[f_stripe_y:f_stripe_y+f_stripe_h,f_stripe_x:f_stripe_x+f_stripe_w] = cv2.GC_FGD

			f_stripe = (f_stripe_x, f_stripe_y, f_stripe_w, f_stripe_h)
			draw_rect(f_stripe, (255,0,0), draw)
			

		if torso_bb != None: 
			# get indices of torso stripe
			t_stripe_cent_x = int(x_torso + w_torso / 2)
			t_stripe_cent_y = int(y_torso + h_torso / 2)
			t_stripe_w = int(w_torso * TORSO_STRIPE_WIDTH_RATIO)
			t_stripe_h = int(h_torso * TORSO_STRIPE_HEIGHT_RATIO)
			t_stripe_x = int(t_stripe_cent_x - t_stripe_w / 2)
			t_stripe_y = int(t_stripe_cent_y - t_stripe_h / 2)

			mask[t_stripe_y:t_stripe_y+t_stripe_h,t_stripe_x:t_stripe_x+t_stripe_w] = cv2.GC_FGD

			t_stripe = (t_stripe_x, t_stripe_y, t_stripe_w, t_stripe_h)
			draw_rect(t_stripe, (255,0,0), draw)

			if face_bb == None: 
				#extend up 
				t_sliver_w = int(w_torso * TORSO_STRIPE_WIDTH_RATIO)
				t_sliver_h = int(h_torso * TORSO_SLIVER_HEIGHT_RATIO)
				t_sliver_x = int(x_torso + w_torso / 2 - t_sliver_w / 2)
				t_sliver_y = int(y_torso - (1 - SLIVER_OVERLAP) * t_sliver_h)
				mask[t_sliver_y:t_sliver_y+t_sliver_h,t_sliver_x:t_sliver_x+t_sliver_w] = cv2.GC_FGD

				t_sliver = (t_sliver_x, t_sliver_y, t_sliver_w, t_sliver_h)
				draw_rect(t_sliver, (255,0,0), draw)

		else:
			f_stripe_h += x_face
			mask[f_stripe_y:f_stripe_y+f_stripe_h,f_stripe_x:f_stripe_x+f_stripe_w] = cv2.GC_FGD

			f_stripe_alt = (f_stripe_x, f_stripe_y, f_stripe_w, f_stripe_h)
			draw_rect(f_stripe_alt, (255,0,0), draw)

		return mask

	
