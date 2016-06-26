from keras.layers import Input, Dense
from keras.models import Model

from autoencoder_base import AutoencoderBase

class DeepAutoencoder(AutoencoderBase):

	def __init__(self, dims):
		dim_in = dims[0]
		dim_out = dims[-1]
		dims_encoder = dims[1:]
		dims_decoding = dims[:-1]
		dims_decoding.reverse()

		input_img = Input(shape=(dim_in,), name='EncoderIn')
		decoder_input = Input(shape=(dim_out,), name='DecoderIn')

		encoded = input_img

		# Construct encoder layers
		for i, dim in enumerate(dims_encoder):
			name = 'Encoder{0}'.format(i)
			encoded = Dense(dim, activation='relu', name=name)(encoded)

		# Construct decoder layers
		# The decoded is connected to the encoders, whereas the decoder is not
		decoded = encoded
		decoder = decoder_input
		for i, dim in enumerate(dims_decoding):
			name = 'Decoder{0}'.format(i)

			activation = 'relu'
			if i == len(dims_decoding) - 1:
				activation = 'sigmoid'

			layer = Dense(dim, activation=activation, name=name)

			decoded = layer(decoded)
			decoder = layer(decoder)

		self.autoencoder = Model(input=input_img, output=decoded)
		self.encoder = Model(input=input_img, output=encoded)
		self.decoder = Model(input=decoder_input, output=decoder)

		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
