# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                                   Execution step 
# ----------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import sys

sys.path.append('../')
from utils import input_processing
import pickle
from keras.models import load_model
from env import motion_ST_AE
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import IPython
import random
from operator import add
import argparse

# Arguments
parser = argparse.ArgumentParser(description='Select Style. 0: Happy; 1: Calm, 2: Sad, 3: Angry.')
parser.add_argument('--style', type=int, default=0)
args = parser.parse_args()


# Parameters
INPUT_SIZE = 50
robot_threshold = 300  # Absolute max range of robot movements in mm

# Velocity bound
upper_bound = 0.1 * robot_threshold
lower_bound = -0.1 * robot_threshold

total_episodes = 1

# Load model
# Happy:0, Calm:1, Sad:2 and Angry:3
actor_model = load_model("./definitive-models/"+str(args.style+1)+"/actor.h5") # Actor

# Path to AE
ae_path = "./../autoencoders/trained-models/autoencoder.h5"

style_data = []
file_list = sorted(os.listdir("./styles"))
for file in file_list:
    print("File extracted : ", file)
    with open(os.path.join("./styles", file)) as f:
        f = f.readlines()
        #IPython.embed()
        motion_traj = []
        for i in range(np.shape(f)[0]):
            if i == 0:
                continue  # Skip first comment line
            step = [float(j) for j in f[i].split()[3:6]]
            motion_traj.append(np.asarray(step))
        motion_traj = np.asarray(motion_traj)
        motion_traj = motion_traj[::10] * 1000  # to 10 hz and mm
        motion_traj = input_processing.input_trajectory_generator(motion_traj, INPUT_SIZE, robot_threshold)
    style_data.append(motion_traj)
selected_styles = []
selected_styles.append(style_data[0][4])  # Happy
selected_styles.append(style_data[1][0])  # Calm
selected_styles.append(style_data[2][5])  # Sad
selected_styles.append(style_data[3][2])  # Angry
selected_styles = input_processing.scale_input(selected_styles, robot_threshold)  # Scale all the styles

# The position [0] is the starting position and we set that position to be zero.
style_motion = selected_styles[args.style]
style_motion = style_motion - style_motion[0]
tf_style_motion = tf.expand_dims(tf.convert_to_tensor(style_motion), 0)

# Init Content motion
content_motion = [[0, 0, 0]]

# env
env = motion_ST_AE.ae_env(content_motion, style_motion, INPUT_SIZE, ae_path)


def policy(state):
    sampled_actions = tf.squeeze(actor_model(state))
    sampled_actions = sampled_actions.numpy()

    # We make sure action is within bounds
    legal_action = [max(min(x, upper_bound), lower_bound) for x in
                    sampled_actions]  # Make sure actions are in the desired range
    return list(np.squeeze(legal_action))


for ep in range(total_episodes):

    ### Init Content motion
    # Select random seed for the generation of the content
    content_seed = [random.uniform(lower_bound/5, upper_bound/5), random.uniform(lower_bound/5, upper_bound/5),
                    random.uniform(lower_bound/5, upper_bound/5)]
    content_seed = [5,5,5]

    content_motion = []

    # Generate Content motion. It is initialized with 0.
    content_motion.append([0, 0, 0])
    content_motion_input = input_processing.input_generator(content_motion, INPUT_SIZE)

    # Define next step of the content using the seed (is one step ahead)
    content_motion.append(list(np.clip(list(map(add, content_motion[0], content_seed)),
                                       -robot_threshold, robot_threshold)))
    
    # Change the seed?
    if random.random() <0:#< 0.03:
        content_seed = [random.uniform(lower_bound/5, upper_bound/5),
                        random.uniform(lower_bound/5, upper_bound/5),
                        random.uniform(lower_bound/5, upper_bound/5)]

    # Init env and generated_motion
    generated_motion = env.reset(content_motion, style_motion)
    episodic_reward = 0

    step = 1
    done = 0
    print("Episode ", ep)
    end_traj = 0
    
    while True:
        # Call the policy using the *last* content motion and generated motion as tensors
        generated_motion_input = input_processing.input_generator(generated_motion, INPUT_SIZE)
        tf_generated_motion = tf.expand_dims(tf.convert_to_tensor(generated_motion_input), 0)
        content_motion_input = input_processing.input_generator(content_motion, INPUT_SIZE)
        tf_content_motion = tf.expand_dims(tf.convert_to_tensor(content_motion_input), 0)
        tf_prev_state = [tf_content_motion, tf_generated_motion]
        action = policy(tf_prev_state)

        # Define next step of the content (one step ahead)
        if np.shape(content_motion)[0] < INPUT_SIZE:
            content_motion.append(list(np.clip(list(map(add, content_motion[step], content_seed)),
                                               -robot_threshold, robot_threshold)))

        if random.random() <0:#< 0.03:
            content_seed = [random.uniform(lower_bound/5, upper_bound/5), random.uniform(lower_bound/5, upper_bound/5),
                            random.uniform(lower_bound/5, upper_bound/5)]

        # Receive state and reward from environment.
        generated_motion, reward, done = env.step(action, content_motion)  # Step outpus a list for generated
        step += 1
        episodic_reward += reward

        # End episode if done true
        if done:
            break
        print("#", end="")
        sys.stdout.flush()

    print("/")
    print("The total reward is", episodic_reward)
    content_motion_array = np.asarray(content_motion)
    generated_motion_array = np.asarray(generated_motion)

    # Do some plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    for i in range(30):
        if i < 25:
            ax.plot(generated_motion_array[:, 0][i:i + 2], generated_motion_array[:, 1][i:i + 2], generated_motion_array[:, 2][i:i + 2], 'b', linewidth=2)
        else:
            # The last part of the trajectory is in red
            ax.plot(generated_motion_array[:, 0][i:i + 2], generated_motion_array[:, 1][i:i + 2],
                   generated_motion_array[:, 2][i:i + 2], 'r', linewidth=2)

    for i in range(30):
        ax.plot(content_motion_array[:, 0][i:i + 2], content_motion_array[:, 1][i:i + 2], content_motion_array[:, 2][i:i + 2], 'y', linewidth=2)
        
    plt.show()
