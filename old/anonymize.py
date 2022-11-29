#import opencv
from re import L
import cv2

import matplotlib.pyplot as plt
import cvlib as cv
import os
import glob
# print(cv2__.version__)

class FaceDetection:
	def extract_images(self,filename, outdir):
		os.makedirs(outdir, exist_ok=True)
		vid = cv2.VideoCapture(filename)
		i, ret = 0, True
		while ret:
			ret, frame = vid.read()
			if ret:
				# print("read", i)
				cv2.imwrite(outdir + str(i) + ".jpg", frame)
				i += 1
		vid.release()

	def draw_image_with_boxes(self,outdir):
		i = 0
		for file in glob.glob(outdir + '*.jpg'):
			# print(file)
			im = cv2.imread(file)
			faces, confidences = cv.detect_face(im)
			print(confidences)
			# loop through detected faces and add bounding box
			for face in faces:
				(startX,startY) = face[0],face[1]
				(endX,endY) = face[2],face[3]
				# draw rectangle over face
				cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)
			cv2.imwrite(outdir + 'bboxes/'+str(i)+'.jpg',im)
			i+=1

		img_array = []
		for filename in glob.glob(outdir+'bboxes/'+'*.jpg'):
			img = cv2.imread(filename)
			height, width, layers = img.shape
			size = (width,height)
			img_array.append(img)
		out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
		
		for i in range(len(img_array)):
			out.write(img_array[i])
		out.release()
	
if __name__== "__main__":
	
	filename = '/home/summer/nec/lab_data/video/0.avi'
	outdir = '/home/summer/nec/lab_data/result/'
	obj  = FaceDetection()
	# obj.extract_images(filename,outdir)
	obj.draw_image_with_boxes(outdir)
	
