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
# Happy:1, Calm:2, Sad:3 and Angry:4
#actor_model = load_model("./definitive-models/"+str(args.style+1)+"/actor.h5") # Actor
actor_model = load_model("./NPST3-2-models/05-05-22/actor.h5") # Actor

# Path to AE
ae_path = "./../autoencoders/trained-models/autoencoder.h5"

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


# Generate Content Motion
#Simple straight line
#c_x=np.linspace(0,20,num=50).reshape(-1,1) #Generate 1 column array
#c_y=np.linspace(0,200,num=50).reshape(-1,1)
#c_z=np.linspace(0,50,num=50).reshape(-1,1)

# Simple pick&place task
#x
c_x_0 = np.linspace(10,10,num=15)
c_x_1 = np.linspace(10,10,num=20)
c_x_2 = np.linspace(10,10,num=15)
c_x = np.concatenate((c_x_0, c_x_1, c_x_2)).reshape(-1,1)
#y
c_y_0 = np.linspace(10,10,num=15)
c_y_1 = np.linspace(10,200,num=20)
c_y_2 = np.linspace(200,200,num=15)
c_y = np.concatenate((c_y_0, c_y_1, c_y_2)).reshape(-1,1)

#z
c_z_0 = np.linspace(10,100,num=15)
c_z_1 = np.linspace(100,100,num=20)
c_z_2 = np.linspace(100,10,num=15)
c_z = np.concatenate((c_z_0, c_z_1, c_z_2)).reshape(-1,1)

# Generated test
#Add sine noise
#sin_range = np.arange(0, 30, 1)
#noise = np.sin(sin_range)*100

#Add Style noise
noise=[]
for i in range(np.shape(style_motion)[0] - 1):
    noise.append([x - y for (x, y) in zip(style_motion[i + 1], style_motion[i])])

noise = 0.1*np.asarray(noise)
#x
g_x_0 = np.linspace(10,10,num=15)
g_x_1 = np.linspace(10,10,num=20)
g_x_2 = np.linspace(10,10,num=15)
g_x = np.concatenate((c_x_0, c_x_1, c_x_2)).reshape(-1,1)
#y
g_y_0 = np.linspace(10,10,num=15)
g_y_1 = np.linspace(10,200,num=20)
g_y_2 = np.linspace(200,200,num=15)
g_y = np.concatenate((c_y_0, c_y_1, c_y_2)).reshape(-1,1)

#z
g_z_0 = np.linspace(10,100,num=15)
g_z_1 = np.linspace(100,100,num=20)
g_z_2 = np.linspace(100,10,num=15)
g_z = np.concatenate((c_z_0, c_z_1, c_z_2)).reshape(-1,1)


# Full trajectory
content_motion = c_x
content_motion = np.append(content_motion, c_y, axis=1)
content_motion = np.append(content_motion, c_z, axis=1)
tf_content_motion = tf.expand_dims(tf.convert_to_tensor(content_motion), 0)

# env
env = motion_ST_AE.ae_env(content_motion, style_motion, INPUT_SIZE, ae_path)

# reward history
cl_hist=[]
sl_hist=[]
vel_hist=[]
poss_hist=[]
end_poss_hist=[]


def policy(state):
    sampled_actions = tf.squeeze(actor_model(state))
    sampled_actions = sampled_actions.numpy()

    # We make sure action is within bounds
    legal_action = [max(min(x, upper_bound), lower_bound) for x in
                    sampled_actions]  # Make sure actions are in the desired range
    return list(np.squeeze(legal_action))


for ep in range(total_episodes):

    # Init env and generated_motion
    generated_motion = env.reset(content_motion, style_motion)
    episodic_reward = 0
    
    # For Style Motion
    #generated_motion[0]=style_motion[0]

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
        
        # Hard coded action
        escala = 5
        if step*escala<INPUT_SIZE:
            action = [-(g_x[(step-1)*escala]-g_x[(step)*escala])[0]+noise[(step-1)*escala][0],-(g_y[(step-1)*escala]-g_y[(step)*escala])[0]+noise[(step-1)*escala][1],-(g_z[(step-1)*escala]-g_z[(step)*escala])[0]+noise[(step-1)*escala][2]]
            #action = [-(g_x[(step-1)*escala]-g_x[(step)*escala])[0]+noise[(step-1)*escala][0],-(g_y[(step-1)*escala]-g_y[(step)*escala])[0],-(g_z[(step-1)*escala]-g_z[(step)*escala])[0]]
            action = [max(min(x, upper_bound), lower_bound) for x in
                    action]
        	#print(noise[step-1][0])
        	#action = style_motion[step*escala]-style_motion[(step-1)*escala]
        	#action = [random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound)]
        	#action = [-x for x in action] #Negate action
        else: 
        	action = [0,0,0]
        	
        
        # Receive state and reward from environment.
        generated_motion, reward, cl, sl, vel_loss, pos_loss_cont, pos_loss, done = env.step(action, content_motion)
        #print(cl, sl, vel_loss, pos_loss_cont, pos_loss)
        #print("Generated motion", generated_motion[step-1])
        #print("Content motion", content_motion[step-1])
        #print("Style motion", style_motion[step-1])
        

        cl_hist.append(cl)
        sl_hist.append(sl)
        vel_hist.append(vel_loss)
        poss_hist.append(pos_loss_cont)
        end_poss_hist.append(pos_loss)

        # Step outpus a list for generated
        step += 1
        episodic_reward += reward

        # End episode if done true
        if done:
            break
        print("#", end="")
        sys.stdout.flush()

    print("/")
    print("The total loss is", episodic_reward)
    print("The CL loss is:", np.sum(cl_hist))
    print("The SL loss is:", np.sum(sl_hist))
    print("The Vel loss is:", np.sum(vel_hist))
    print("The Poss loss is:", np.sum(poss_hist))
    print("The EndPoss loss is:", np.sum(end_poss_hist))
    
    content_motion_array = np.asarray(content_motion)
    generated_motion_array = np.asarray(generated_motion)
    style_motion_array = np.asarray(style_motion)

    # Do some plotting

    # Losses
    plt.plot(np.linspace(1, 49, num=49), cl_hist, label="content_loss")
    plt.plot(np.linspace(1, 49, num=49), sl_hist, label="style_loss")
    plt.plot(np.linspace(1, 49, num=49), poss_hist, label="poss_loss")
    plt.plot(np.linspace(1, 49, num=49), end_poss_hist, label="end_poss_loss")
    plt.plot(np.linspace(1, 49, num=49), vel_hist, label="vel_loss")
    plt.legend(loc="upper left")
    #plt.ylim(0, 0.2)

    # Generated trajectory
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # To remove labels from axis
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    #ax.set_zticklabels([])
    #Velocity scaled to a maximum of 0.8m/s
    for i in range(1,INPUT_SIZE):
            ax.plot(generated_motion_array[:, 0][i:i + 2], generated_motion_array[:, 1][i:i + 2], generated_motion_array[:, 2][i:i + 2], c=plt.cm.jet(int(np.linalg.norm(generated_motion_array[i]-generated_motion_array[i-1])*255/80)), linewidth=2)
            #print("Generated: ", np.linalg.norm(generated_motion_array[i]-generated_motion_array[i-1]))
            ##print("Content: ", np.linalg.norm(content_motion_array[i]-content_motion_array[i-1]))
            #print("Style: ", np.linalg.norm(style_motion_array[i]-style_motion_array[i-1]))
            #print("Velocity loss: ", vel_hist[i-1])
            #ax.plot(content_motion_array[:, 0][i:i + 2], content_motion_array[:, 1][i:i + 2], content_motion_array[:, 2][i:i + 2], c=plt.cm.jet(int(np.linalg.norm(content_motion_array[i]-content_motion_array[i-1])*255/80)), linewidth=2)
            ax.plot(style_motion_array[:, 0][i:i + 2], style_motion_array[:, 1][i:i + 2], style_motion_array[:, 2][i:i + 2], c=plt.cm.jet(int(np.linalg.norm(style_motion_array[i]-style_motion_array[i-1])*255/80)), linewidth=2)
    plt.show()
