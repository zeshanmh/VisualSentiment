import sys
import os
sys.path.insert(0, './util')

from FaceExtractor import FaceExtractor

def main():

	img_path = "../data/groupdataset_release/images/4452848253_acf5eb37f2_o.jpg"
	face_extractor = FaceExtractor()
	face_extractor.detect_faces(img_path)


if __name__ == '__main__':
	main()
