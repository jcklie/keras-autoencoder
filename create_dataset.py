import os.path

import numpy as np
import scipy.io

import imutils
import cv2

import transformations as tf

greenLower = (50, 137, 21)
greenUpper = (94, 255, 255)

BALL_SIZE = .14 # m

def detect_ball(frame):
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
 
	# only proceed if at least one contour was found
	if len(cnts) == 0:
		return

	# find the largest contour in the mask, then use
	# it to compute the minimum enclosing circle and
	# centroid
	c = max(cnts, key=cv2.contourArea)
	((x, y), radius) = cv2.minEnclosingCircle(c)
	M = cv2.moments(c)
	center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

	if radius < 10:
		print('Too small')
		return

	return center, radius

def draw_circle(img, x, y, radius):
	frame = img.copy()
	if radius > 10:
		# draw the circle and centroid on the frame,
		# then update the list of tracked points
		cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		sys.exit(0)

def compute_transform(rx, rz, tx, ty, tz):
	origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

	Rx = tf.rotation_matrix(rx, xaxis)
	Rz = tf.rotation_matrix(rz, zaxis)
	T = tf.translation_matrix([tx, ty, tz])
	
	return T.dot(Rz).dot(Rx)	

def distance_from_known_size(size_pixels, f):
	return f * BALL_SIZE / size_pixels

def get_camera_matrix():
	C = np.array([
		[ 402.22,    0.  ,  321.5 ],
       	[   0.  ,  402.22,  241.5 ],
       	[   0.  ,    0.  ,    1.  ]])
	return C

def pixel_to_world(Cinv, M, f, px, py, r):
	z = distance_from_known_size(r, f)

	p_pix_hom = np.array([px, py, 1])

	p_cam3d = Cinv.dot(p_pix_hom) * z
	p_cam3d_hom = np.append(p_cam3d, 1)

	p_world_hom = M.dot(p_cam3d_hom)

	return p_world_hom

def create_dataset(path, dest, dist, transpose, rgb, resize_factor):
	cap = cv2.VideoCapture(path)
	length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

	fac = 3 if rgb else 1

	rwidth = int(width * resize_factor * fac)
	rheight = int(height * resize_factor * fac)

	frames = 0
	I = np.zeros([rwidth*rheight * fac, length])
	Phi = np.zeros([3, length])

	while True:
		(grabbed, frame) = cap.read()
		
		if not grabbed: break		

		p, r = detect_ball(frame)

		if p != None:
			Phi[:,frames] = dist(p, r)[0:3]
		else:
			Phi[:,frames] = np.nan

		img = frame

		if not rgb:
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		img = imutils.resize(img, width=rwidth, height=rheight, inter=cv2.INTER_AREA)

		I[:,frames] = img.T.ravel() if transpose else img.ravel()

		frames += 1

	print("Got {0} frames".format(frames))

	cap.release()
	cv2.destroyAllWindows()

	data = {'I' : I, 'h' : rheight, 'w' : rwidth, 'Phi' : Phi}
	scipy.io.savemat(dest, data , do_compression=True)

if __name__ == '__main__':
	path = 'data/ball_kawaii3.avi'
	cap = cv2.VideoCapture(path)

	# First rotate around x, then z
	rx = np.deg2rad(-90)
	rz = np.deg2rad(180)

	# Translation is done after rotation
	tx = 0.9
	ty = 0.84
	tz = 0.3

	C = get_camera_matrix()
	Cinv = np.linalg.inv(C)
	f = C[0,0]
	M = compute_transform(rx, rz, tx, ty, tz)

	dist = lambda p, r: pixel_to_world(Cinv, M, f, p[0], p[1], r)

	datafolder = '/home/jck/Dropbox/tu/3/iprobo1/data/cleaned'

	# False for Numpy, true for Matlab
	transpose = True

	# True for rgb, false for grey
	rgb = True

	# How much resize
	resize_factor = .05

	for i, name in enumerate(['cleaned1', 'cleaned2', 'cleaned3']):
		path =  os.path.join(datafolder, name + '.avi')
		dest = os.path.join(datafolder, '{0}_{1}.mat'.format(name, 'ml' if transpose else 'py'))
		create_dataset(path, dest, dist, transpose, rgb, resize_factor)

# 	while True:
# 		(grabbed, frame) = cap.read()
# 		img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# 		detect_ball(img)