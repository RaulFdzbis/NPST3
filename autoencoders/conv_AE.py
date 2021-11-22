# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                                Autoencoder training
# ----------------------------------------------------------------------------------------

from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
from keras.callbacks import TensorBoard
import pickle
import sys
sys.path.append('../')
from utils import input_processing


# Params
INPUT_SIZE = 50
robot_threshold = 300

# Input
input_motion = Input(shape=(INPUT_SIZE,3))

# Encoder
x = Conv1D(256, 5, activation='relu', padding='same')(input_motion)
x = MaxPooling1D(2, padding='same')(x)
encoded = Dropout(0.2)(x)

# Decoder
x = UpSampling1D(2)(encoded)
decoded = Conv1D(3, 5, padding='same')(x)

# Compile model
autoencoder = Model(input_motion, decoded)
autoencoder.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')

# Encoder model
encoder = Model(input_motion, encoded)

# Decoder model
encoded_input = Input(shape=(25,256))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


# load dataset
with open('../dataset/NPST3_dataset.pickle', 'rb') as data:
    marker_data = pickle.load(data)

# Create the inputs for training and testing
train_data, test_data = input_processing.dataset_input_generator(marker_data, INPUT_SIZE, robot_threshold)

autoencoder.fit(train_data, train_data,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(test_data, test_data),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]) # what is the goal of this

# Save the model with the weights
autoencoder.save("autoencoder.h5")
encoder.save("encoder.h5")
decoder.save("decoder.h5")

