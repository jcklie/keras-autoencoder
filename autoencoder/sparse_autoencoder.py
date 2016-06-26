from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers

from autoencoder_base import AutoencoderBase

class SparseAutoencoder(AutoencoderBase):

	def __init__(self, dim_in, encoding_dim, sparsity):
		input_img = Input(shape=(dim_in,))

		regulizer = regularizers.activity_l2(sparsity)
		encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regulizer)(encoded)		

		decoded = Dense(dim_in, activation='sigmoid')(decoded)

		self.autoencoder = Model(input=input_img, output=decoded)

		self.encoder = Model(input=input_img, output=encoded)

		encoded_input = Input(shape=(encoding_dim,))
		decoder_layer = self.autoencoder.layers[-1]
		self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

