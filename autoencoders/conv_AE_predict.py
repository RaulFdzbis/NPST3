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
from src import herm_traj_generator
import IPython

# Params
INPUT_SIZE = 50
robot_threshold = 300
upper_bound = robot_threshold * 0.1

# Load the Autoencoder
autoencoder = load_model("./trained-models/01-07-22/autoencoder.h5")
#autoencoder = load_model("./trained-models/autoencoder.h5")

# Uncomment for using the dataset
'''
with open('../dataset/NPST3_dataset.pickle', 'rb') as data:
     marker_data = pickle.load(data)

# Generate inputs
train_data, test_data = input_processing.dataset_input_generator(marker_data, INPUT_SIZE, robot_threshold)

input_motion = np.expand_dims(test_data[2], axis=0) #choose any trajectory of the test_dataset
'''

'''
# Input Generator
input_motion = herm_traj_generator.generate_base_traj(INPUT_SIZE,robot_threshold,upper_bound)
input_motion = np.expand_dims(input_motion, axis=0) # To adapt to the input of the network

decoded_motion = autoencoder.predict(input_motion)

# Read from file
#IPython.embed()
'''


input_motion = []
n_data = 0
with open("./data/5.log") as f:
    for idx, line in enumerate(f):
    	if idx!= 0 and idx!=1:
    	    if (idx-2)%10 == 0 and n_data<50:
                n_data += 1
                line = line.split(" ")
                input_motion.append([float(line[1])*1000, float(line[2])*1000, float(line[3])*1000])
input_motion = np.expand_dims(input_motion, axis=0) # To adapt to the input of the network
input_motion = input_motion - input_motion[0][0]
input_motion = np.clip(input_motion, -robot_threshold, robot_threshold)

decoded_motion = autoencoder.predict(input_motion)

#IPython.embed()




# Inplut plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(input_motion[0][:,0], input_motion[0][:,1], input_motion[0][:,2], label='original')
ax.set_xlim([-robot_threshold*2, robot_threshold*2])
ax.set_ylim([-robot_threshold*2, robot_threshold*2])
ax.set_zlim([-robot_threshold*2, robot_threshold*2])
#plt.show()

# Decoded plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(decoded_motion[0][:,0], decoded_motion[0][:,1], decoded_motion[0][:,2], label='decoded')
ax.set_xlim([-robot_threshold*2, robot_threshold*2])
ax.set_ylim([-robot_threshold*2, robot_threshold*2])
ax.set_zlim([-robot_threshold*2, robot_threshold*2])
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
