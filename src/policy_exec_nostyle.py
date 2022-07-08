# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                                   Execution step 
# ----------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import sys

sys.path.append('../')
from utils import input_processing
from keras.models import load_model
from env import motion_ST_AE
import matplotlib.pyplot as plt
import os
import IPython
import argparse
import herm_traj_generator
from dtw import *

# Arguments
parser = argparse.ArgumentParser(description='Select Style. 0: Happy; 1: Calm, 2: Sad, 3: Angry.')
parser.add_argument('--style', type=int, default=0)
args = parser.parse_args()

# Parameters
INPUT_SIZE = 50
robot_threshold = 300  # Absolute max range of robot movements in mm
generated_scale = 3
noise_scale = 25

# Velocity bound per step (right now 10Hz so it is uppedbound mm/0.1s)
upper_bound = 0.1 * robot_threshold
lower_bound = -0.1 * robot_threshold

total_episodes = 10

# Load model
# Happy:1, Calm:2, Sad:3 and Angry:4
# actor_model = load_model("./definitive-models/"+str(args.style+1)+"/actor.h5") # Actor
actor_model = load_model("./NPST3-2-models/06-09-22/actor.h5")  # Actor

# Path to AE
ae_path = "./../autoencoders/trained-models/08-07-22/autoencoder.h5"
ae_path_2 = "./../autoencoders/trained-models/autoencoder.h5"

style_data = []
file_list = sorted(os.listdir("./styles"))
for file in file_list:
    print("File extracted : ", file)
    with open(os.path.join("./styles", file)) as f:
        f = f.readlines()
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


def policy(state):
    sampled_actions = tf.squeeze(actor_model(state))
    sampled_actions = sampled_actions.numpy()

    # We make sure action is within bounds
    legal_action = [max(min(x, upper_bound), lower_bound) for x in
                    sampled_actions]  # Make sure actions are in the desired range
    return list(np.squeeze(legal_action))


# env
content_motion = herm_traj_generator.generate_base_traj(INPUT_SIZE, robot_threshold, upper_bound)  # Actually not used
env = motion_ST_AE.ae_env(content_motion, style_motion, INPUT_SIZE, ae_path)
env2 = motion_ST_AE.ae_env(content_motion, style_motion, INPUT_SIZE, ae_path_2)

# content_motion = herm_traj_generator.generate_base_traj(INPUT_SIZE, robot_threshold, upper_bound)

for ep in range(total_episodes):
    # reward history
    cl_hist = []
    sl_hist = []
    vel_hist = []
    poss_hist = []
    end_poss_hist = []
    cl_hist2 = []
    sl_hist2 = []
    vel_hist2 = []
    poss_hist2 = []
    end_poss_hist2 = []

    content_motion = herm_traj_generator.generate_base_traj(INPUT_SIZE, robot_threshold, upper_bound)

    # Init env and generated_motion
    generated_motion = env.reset(content_motion, style_motion)
    generated_motion2 = env2.reset(content_motion, style_motion)
    episodic_reward = 0

    # Generate Content motion
    tf_content_motion = tf.expand_dims(tf.convert_to_tensor(content_motion), 0)
    # content_motion_input = input_processing.input_generator(content_motion, INPUT_SIZE)

    # Hard coded Generated motion
    g_x = [i[0] for i in content_motion]
    g_y = [i[1] for i in content_motion]
    g_z = [i[2] for i in content_motion]

    ## Define noise

    # Add Style noise
    # noise=[]
    # for i in range(np.shape(style_motion)[0] - 1):
    #    noise.append([x - y for (x, y) in zip(style_motion[i + 1], style_motion[i])])

    # Add sine noise
    # sin_range = np.arange(0, 200, 4)
    # noise = np.sin(sin_range)
    # noise = noise_scale * np.asarray(noise)

    # For Style Motion
    # generated_motion[0]=style_motion[0]

    step = 1
    done = 0
    print("Episode ", ep)
    end_traj = 0

    while True:
        # Call the policy using the *last* content motion and generated motion as tensors
        generated_motion_input = input_processing.input_generator(generated_motion, INPUT_SIZE)
        tf_generated_motion = tf.expand_dims(tf.convert_to_tensor(generated_motion_input), 0)
        tf_prev_state = [tf_content_motion, tf_generated_motion]
        action = policy(tf_prev_state)
        # print(action)

        '''
        # Hard coded action
        if ep == 0:
            escala = 1
        elif ep == 1:
            escala = 3
        elif ep == 2:
            escala = 5
        elif ep == 3:
            alignment = dtw(content_motion, style_motion, keep_internals=True)
            wq = warp(alignment, index_reference=False)  # Find the warped trajectory
            warped_g = np.asarray(content_motion)[wq]
            warped_g = np.append(warped_g, [np.asarray(content_motion)[-1]],
                                 axis=0)  # Add last point to warping (warp dont do this)
            g_x = [i[0] for i in warped_g]
            g_y = [i[1] for i in warped_g]
            g_z = [i[2] for i in warped_g]

        # Hard coded action
        ##escala = generated_scale
        if step * escala < INPUT_SIZE:
            # action = [-(g_x[(step-1)*escala]-g_x[(step)*escala])+noise[(step-1)*escala][0],-(g_y[(step-1)*escala]-g_y[(step)*escala])+noise[(step-1)*escala][1],-(g_z[(step-1)*escala]-g_z[(step)*escala])+noise[(step-1)*escala][2]] #XYZ noise
            # action = [-(g_x[(step-1)*escala]-g_x[(step)*escala])+noise[(step-1)*escala],-(g_y[(step-1)*escala]-g_y[(step)*escala]),-(g_z[(step-1)*escala]-g_z[(step)*escala])] # X noise
            # action = [-(g_x[(step-1)*escala]-g_x[(step)*escala])+noise[(step-1)],-(g_y[(step-1)*escala]-g_y[(step)*escala]),-(g_z[(step-1)*escala]-g_z[(step)*escala])] #Sine noise
            action = [-(g_x[(step - 1) * escala] - g_x[(step) * escala]),
                      -(g_y[(step - 1) * escala] - g_y[(step) * escala]),
                      -(g_z[(step - 1) * escala] - g_z[(step) * escala])]  # No noise
            # print(action)
            # action = style_motion[step*escala]-style_motion[(step-1)*escala] # Style
            action = np.clip(action, -upper_bound, upper_bound)
        # print(noise[step-1][0])

        # action = [random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound)]
        # action = [-x for x in action] #Negate action
        else:
            action = [0, 0, 0]
        '''

        # Receive state and reward from environment.
        generated_motion, reward, cl, sl, vel_loss, pos_loss_cont, pos_loss, done, warped_traj = env.step(action,
                                                                                                          content_motion)
        generated_motion2, reward2, cl2, sl2, vel_loss2, pos_loss_cont2, pos_loss2, done2, warped_traj = env2.step(
            action, content_motion)

        cl_hist.append(cl)
        sl_hist.append(sl)
        vel_hist.append(vel_loss)
        poss_hist.append(pos_loss_cont)
        end_poss_hist.append(pos_loss)

        cl_hist2.append(cl2)
        sl_hist2.append(sl2)
        vel_hist2.append(vel_loss2)
        poss_hist2.append(pos_loss_cont2)
        end_poss_hist2.append(pos_loss2)

        # Step outpus a list for generated
        step += 1
        episodic_reward += reward

        # End episode if done true
        if done:
            break
        print("#", end="")
        sys.stdout.flush()

    content_motion_array = np.asarray(content_motion)
    generated_motion_array = np.asarray(generated_motion)
    style_motion_array = np.asarray(style_motion)
    warped_traj_array = np.asarray(warped_traj)

    print("Content is: ", content_motion)
    print("Generated is: ", generated_motion_array)
    print("Warped is: ", warped_traj_array)

    print("/")
    print("The total loss is", "{:,}".format(episodic_reward))
    print("The CL loss is:", "{:,}".format(np.sum(cl_hist)))
    print("The SL loss is:", "{:,}".format(np.sum(sl_hist)))
    print("The Vel loss is:", "{:,}".format(np.sum(vel_hist)))
    print("The Poss loss is:", "{:,}".format(np.sum(poss_hist)))
    print("The EndPoss loss is:", "{:,}".format(np.sum(end_poss_hist)))

    print("The CL2 loss is:", "{:,}".format(np.sum(cl_hist2)))
    print("The SL2 loss is:", "{:,}".format(np.sum(sl_hist2)))
    print("The Vel2 loss is:", "{:,}".format(np.sum(vel_hist2)))
    print("The Poss2 loss is:", "{:,}".format(np.sum(poss_hist2)))
    print("The EndPoss2 loss is:", "{:,}".format(np.sum(end_poss_hist2)))

    # Do some plotting

    # Losses
    plt.figure()
    plt.plot(np.linspace(1, 49, num=49), cl_hist, label="content_loss")
    plt.plot(np.linspace(1, 49, num=49), sl_hist, label="style_loss")
    plt.plot(np.linspace(1, 49, num=49), poss_hist, label="poss_loss")
    plt.plot(np.linspace(1, 49, num=49), end_poss_hist, label="end_poss_loss")
    plt.plot(np.linspace(1, 49, num=49), vel_hist, label="vel_loss")
    plt.legend(loc="upper left")

    plt.figure()
    plt.plot(np.linspace(1, 49, num=49), cl_hist2, label="content_loss_2")
    plt.plot(np.linspace(1, 49, num=49), sl_hist2, label="style_loss_2")
    plt.plot(np.linspace(1, 49, num=49), poss_hist2, label="poss_loss_2")
    plt.plot(np.linspace(1, 49, num=49), end_poss_hist2, label="end_poss_loss_2")
    plt.plot(np.linspace(1, 49, num=49), vel_hist2, label="vel_loss_2")
    plt.legend(loc="upper left")
    # plt.ylim(0, 0.2)

    # Generated trajectory
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # To remove labels from axis
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.set_zticklabels([])
    ax.axes.set_xlim3d(left=-robot_threshold, right=robot_threshold)
    ax.axes.set_ylim3d(bottom=-robot_threshold, top=robot_threshold)
    ax.axes.set_zlim3d(bottom=-robot_threshold, top=robot_threshold)
    # Velocity scaled to a maximum of 0.8m/s

    for i in range(0, INPUT_SIZE - 1):
        '''
        if np.linalg.norm(generated_motion_array[i] - generated_motion_array[i + 1]) != 0:
            ax.plot(generated_motion_array[:, 0][i:i + 2], generated_motion_array[:, 1][i:i + 2],
                    generated_motion_array[:, 2][i:i + 2], c=plt.cm.jet(
                    int(np.linalg.norm(generated_motion_array[i] - generated_motion_array[i + 1]) * 255 / 50)),
                    linewidth=2)
            # print("Generated: ", np.linalg.norm(generated_motion_array[i]-generated_motion_array[i-1]))
            # print("Content: ", np.linalg.norm(content_motion_array[i]-content_motion_array[i-1]))
            # print("Style: ", np.linalg.norm(style_motion_array[i]-style_motion_array[i-1]))
            # print("Velocity loss: ", vel_hist[i-1])
        '''

        if np.linalg.norm(content_motion_array[i] - content_motion_array[i + 1]) != 0:
            ax.plot(warped_traj_array[:, 0][i:i + 2], warped_traj_array[:, 1][i:i + 2],
                    warped_traj_array[:, 2][i:i + 2],
                    c=plt.cm.jet(int(np.linalg.norm(warped_traj_array[i] - warped_traj_array[i + 1]) * 255 / 50)),
                    linewidth=6)

        if (np.linalg.norm(content_motion_array[i] - content_motion_array[i + 1]) != 0):
            ax.plot(content_motion_array[:, 0][i:i + 2], content_motion_array[:, 1][i:i + 2],
                    content_motion_array[:, 2][i:i + 2],
                    c=plt.cm.jet(int(np.linalg.norm(content_motion_array[i] - content_motion_array[i + 1]) * 255 / 50)),
                    linewidth=2)
        '''
        if (np.linalg.norm(style_motion_array[i] - style_motion_array[i + 1]) != 0):
            ax.plot(style_motion_array[:, 0][i:i + 2], style_motion_array[:, 1][i:i + 2],
                    style_motion_array[:, 2][i:i + 2],
                    c=plt.cm.jet(int(np.linalg.norm(style_motion_array[i] - style_motion_array[i + 1]) * 255 / 50)),
                    linewidth=2)
        '''
    plt.show()
