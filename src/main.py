import sys
import os
sys.path.insert(0, './util')

import FaceExtractor

def main():

	img_path = "../data/groupdataset_release/images/64015746.jpg"
	face_extractor = FaceExtractor()
	face_extractor.detect_faces(img_path)


if __name__ == '__main__':
	main()