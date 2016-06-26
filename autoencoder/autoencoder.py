from keras.layers import Input, Dense
from keras.models import Model

from autoencoder_base import AutoencoderBase

class Autoencoder(AutoencoderBase):

	def __init__(self, dim_in, encoding_dim):
		input_img = Input(shape=(dim_in,), name='EncoderIn')

		encoded = Dense(encoding_dim, activation='relu', name='Encoder')(input_img)		

		decoded = Dense(dim_in, activation='sigmoid', name='Decoder')(encoded)

		self.autoencoder = Model(input=input_img, output=decoded)

		self.encoder = Model(input=input_img, output=encoded)

		encoded_input = Input(shape=(encoding_dim,), name='DecoderIn')
		decoder_layer = self.autoencoder.layers[-1]
		self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
