# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                                Autoencoder training
# ----------------------------------------------------------------------------------------
import numpy as np
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
import sys
sys.path.append('../')
from src import herm_traj_generator


# Params
INPUT_SIZE = 50
robot_threshold = 300
upper_bound = 0.1 * robot_threshold
lower_bound = -0.1 * robot_threshold

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


# create dataset
train_data = []
test_data = []
train_data_size = 100000
test_data_size = 10000
for i in range(train_data_size):
    train_data.append(herm_traj_generator.generate_base_traj(INPUT_SIZE,robot_threshold,upper_bound))

for i in range(test_data_size):
    test_data.append(herm_traj_generator.generate_base_traj(INPUT_SIZE, robot_threshold, upper_bound))

train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
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

