# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                                Autoencoder predict
# ----------------------------------------------------------------------------------------
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
from utils import input_processing

# Params
INPUT_SIZE = 50
robot_threshold = 300

# Load the Autoencoder
autoencoder = load_model("./trained-models/autoencoder.h5")

# Uncomment for using the dataset
'''
with open('../dataset/NPST3_dataset.pickle', 'rb') as data:
     marker_data = pickle.load(data)

# Generate inputs
train_data, test_data = input_processing.dataset_input_generator(marker_data, INPUT_SIZE, robot_threshold)

input_motion = np.expand_dims(test_data[2], axis=0) #choose any trajectory of the test_dataset
'''

# Simple example
content_motion = np.loadtxt("./../src/test-trajectories/line-test.txt", delimiter=" ")  # WARNING: array of arrays
content_motion = content_motion - content_motion[0]
content_motion = content_motion * robot_threshold
input_motion = np.expand_dims(content_motion, axis=0)

decoded_motion = autoencoder.predict(input_motion)

# Inplut plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(content_motion[:,0], content_motion[:,1], content_motion[:,2], label='original')
ax.set_xlim([-robot_threshold, robot_threshold])
ax.set_ylim([-robot_threshold, robot_threshold])
ax.set_zlim([-robot_threshold, robot_threshold])
plt.show()

# Decoded plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(decoded_motion[0][:,0], decoded_motion[0][:,1], decoded_motion[0][:,2], label='decoded')
ax.set_xlim([-robot_threshold, robot_threshold])
ax.set_ylim([-robot_threshold, robot_threshold])
ax.set_zlim([-robot_threshold, robot_threshold])
plt.show()


# Plot
# n = 3
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     decoded_motion = autoencoder.predict(test_data[0])
#     fig = plt.figure()
#     # display original
#     ax = fig.subplot(1,2,1,projection='3d')
#     ax.plot(test_data[0][:,0], test_data[0][:,1], test_data[0][:,2], label='original')
#     #ax.legend()
#     #plt.show()
#
#     # display reconstruction
#     ax = plt.subplot(1,2,2, projection='3d')
#     ax.plot(decoded_motion[0], decoded_motion[1], decoded_motion[2], label='original')
#     #plt.gray()
#     #ax.get_xaxis().set_visible(False)
#     #ax.get_yaxis().set_visible(False)
#
# plt.show()
