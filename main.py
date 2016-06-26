import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model

from autoencoder import *

if __name__ == '__main__':

    #### LOAD DATA
    data = sio.loadmat('data/video5g_cleaned_np.mat')
    h = np.asscalar(data['h'])
    w = np.asscalar(data['w'])

    X = data['I'].T / 255.
    Xm = np.mean(X,axis=0)
    Xs = np.std(X, axis=0)
    Xn = (X-Xm)
    
    n_samples,data_dim = X.shape
    train_percentage = 0.95 # How many % of data are for training?

    (x_train, x_test) = np.split(X, [int(n_samples * train_percentage)])

    x_train = (x_train.astype('float32')) # Normalize dataset
    x_test = (x_test.astype('float32') )
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  

    #### LEARNING SETUP
    autoenc_type = 'sparse'
    encoding_dim = 20 
    epochs = 100
    batch_size = 50

    #### TRAIN
    if autoenc_type == 'vanilla':
        autoenc = Autoencoder(data_dim, encoding_dim)
    elif autoenc_type == 'sparse':
        autoenc = SparseAutoencoder(data_dim, encoding_dim, 0.001)
    elif autoenc_type == 'deep':
        autoenc = DeepAutoencoder([data_dim, 512, 128, 64, encoding_dim])
    elif autoenc_type == 'conv':
        x_train = np.reshape(x_train, (len(x_train), 1, h, w))
        x_test = np.reshape(x_test, (len(x_test), 1, h, w))
        autoenc = ConvolutionalAutoencoder(h,w , [(16, 3, 3), (8, 3, 3), (8, 3, 3)])
    else:
        raise Exception('Unknown autoencoder type: ' + str(autoenc_type))

    print('Using {0} autoencoder...'.format(autoenc_type))

    autoenc.summary()
    autoenc.train(x_train, x_test, epochs, batch_size)

    #### TEST
    encoded_imgs = autoenc.encode(x_test)
    decoded_imgs = autoenc.decode(encoded_imgs)

    decoded_imgs = decoded_imgs

    n_display = 4 # How many images we will display from the testing set
    plt.figure(figsize=(20, 4))
    for i in range(1,n_display+1):
        # Display original
        ax = plt.subplot(2, n_display, i)
        plt.imshow(x_test[i].reshape(h, w))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n_display, i + n_display)
        plt.imshow(decoded_imgs[i].reshape(h, w))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()