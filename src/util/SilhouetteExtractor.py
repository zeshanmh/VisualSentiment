import cv2
import numpy as np

class SilhouetteExtractor:

	def __init__(self):
		pass

	def get_silhouette(self, img):
		pass

	def get_mask(self, img, img_bb, face_bb):
		x, y, w, h = img_bb
		x_face, y_face, w_face, h_face = face_bb
		mask = np.zeros_like(img)
		# face = img[y_face:y_face+h_face,x_face:x_face+w_face]

		# get indices of face stripe
		f_stripe_x = x_face + 3.0 / 8 * w_face
		f_stripe_y = y_face + 0.05 * h_face
		f_stripe_w = 0.125 * w_face
		f_stripe_h = 0.9 * h_face

		# set mask at face stripe to foreground
		mask[f_stripe_y:f_stripe_y+f_stripe_h,f_stripe_x:f_stripe_x+f_stripe_w] = 1

		# get indices of torso stripe
		# t_stripe_x = 
		# t_stripe_y
		# t_stripe_w
		# t_stripe_h
