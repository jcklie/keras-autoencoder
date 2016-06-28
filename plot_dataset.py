import os.path

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
	datafolder = '/home/jck/Dropbox/tu/3/iprobo1/data/cleaned'

	for name in ['cleaned1_ml', 'cleaned2_ml', 'cleaned3_ml']:
		path = os.path.join(datafolder, name)
		data = sio.loadmat(path)
		x,y,z = data['Phi']

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(x, y, z, c='r', marker='o')

		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')

		plt.savefig(path + '.pdf')

	