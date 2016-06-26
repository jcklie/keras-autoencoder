from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

from autoencoder_base import AutoencoderBase

class ConvolutionalAutoencoder(AutoencoderBase):

    def __init__(self, h_in, w_in, dims):
        # Each MaxPooling2D (2,2) layer halves the image size
        resize_factor = len(dims)

        # Number of filters on last layer correspond to
        # the number of filters of the last conv layer
        filters_encoded = dims[-1][0]

        input_img = Input(shape=(1,h_in, w_in), name='EncoderIn')
        decoder_input = Input(shape=(filters_encoded, h_in / (2 ** resize_factor), w_in / (2 ** resize_factor)), name='DecoderIn')

        # Construct encoder layers
        encoded = input_img

        for i, (filters, rows, cols) in enumerate(dims):
            name = 'Conv{0}'.format(i)
            encoded = Convolution2D(filters, rows, cols, activation='relu', border_mode='same', name=name)(encoded)
            encoded = MaxPooling2D((2, 2), border_mode='same', name= 'MaxPool{0}'.format(i))(encoded)

        # Construct decoder layers
        # The decoded is connected to the encoders, whereas the decoder is not
        decoded = encoded
        decoder = decoder_input
        for i, dim in enumerate(reversed(dims)):
            convlayer = Convolution2D(filters, rows, cols, activation='relu', border_mode='same', name='Deconv{0}'.format(i))
            decoded = convlayer(decoded)
            decoder = convlayer(decoder)

            upsample = UpSampling2D((2, 2), name='UpSampling{0}'.format(i))
            decoded = upsample(decoded)
            decoder = upsample(decoder)

        # Reduce from X filters to 1 in the output layer. Make sure its sigmoid for the [0..1] range
        convlayer = Convolution2D(1, dims[0][0], dims[0][1], activation='sigmoid', border_mode='same')
        decoded = convlayer(decoded)
        decoder = convlayer(decoder)

        self.autoencoder = Model(input=input_img, output=decoded)
        self.encoder = Model(input=input_img, output=encoded)
        self.decoder = Model(input=decoder_input, output=decoder)

        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')