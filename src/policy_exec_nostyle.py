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
import random
import argparse
import copy
import time

# Arguments
parser = argparse.ArgumentParser(description='Select Style. 0: Happy; 1: Calm, 2: Sad, 3: Angry.')
parser.add_argument('--style', type=int, default=0)
args = parser.parse_args()


# Parameters
INPUT_SIZE = 50
robot_threshold = 300  # Absolute max range of robot movements in mm
generated_scale = 1
noise_scale = 25

# Velocity bound
upper_bound = 0.1 * robot_threshold
lower_bound = -0.1 * robot_threshold

total_episodes = 3

# Load model
# Happy:1, Calm:2, Sad:3 and Angry:4
#actor_model = load_model("./definitive-models/"+str(args.style+1)+"/actor.h5") # Actor
actor_model = load_model("./NPST3-2-models/06-09-22/actor.h5") # Actor

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


def test_trajectories(selected_trajectory):
    if selected_trajectory==0: #Straight line
        content_motion = []
        content_motion.append([0, 0, 0])
        for j in range(INPUT_SIZE-1):
            content_motion.append([content_motion[j][0]+50/200,content_motion[j][1]+50/200,content_motion[j][2]+50/200])

    elif selected_trajectory==1: #Pick and place task
        content_motion = []
        content_motion.append([0, 0, 0])
        # Simple pick&place task
        # x
        c_x_0 = np.linspace(0, 0, num=15)
        c_x_1 = np.linspace(0, 150, num=20)
        c_x_2 = np.linspace(150, 150, num=15)
        c_x = np.concatenate((c_x_0, c_x_1, c_x_2)).reshape(-1, 1)
        # y
        c_y_0 = np.linspace(0, 0, num=15)
        c_y_1 = np.linspace(0, 150, num=20)
        c_y_2 = np.linspace(150, 150, num=15)
        c_y = np.concatenate((c_y_0, c_y_1, c_y_2)).reshape(-1, 1)
        # z
        c_z_0 = np.linspace(0, 100, num=15)
        c_z_1 = np.linspace(100, 100, num=20)
        c_z_2 = np.linspace(100, 0, num=15)
        c_z = np.concatenate((c_z_0, c_z_1, c_z_2)).reshape(-1, 1)
        content_motion = c_x
        content_motion = np.append(content_motion, c_y, axis=1)
        content_motion = np.append(content_motion, c_z, axis=1)

    elif selected_trajectory==2: # "free" randomly generated trajectory
        content_motion = []
        # Generate random trajectory
        s_z = np.arange(0,100,2)
        s_x = np.sin(s_z*2*np.pi/100)*100; # Full normalized sine period escaled *100
        s_y = np.cos(s_z*2*np.pi/100)*100; # Full normalized cosine period escaled *100

        for j in range(INPUT_SIZE):
            content_motion.append([s_x[j],s_y[j],s_z[j]])

    else:
        print("Please introduce valid int->0: Straight line; 1: Pick&Place; 2: CMU db trajectory")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    content_motion_array = np.asarray(content_motion)
    for i in range(1, INPUT_SIZE):
        ax.plot(content_motion_array[:, 0][i:i + 2],
                content_motion_array[:, 1][i:i + 2],
                content_motion_array[:, 2][i:i + 2],
                c=plt.cm.jet(int(np.linalg.norm(content_motion_array[i]-content_motion_array[i-1])*255/80)),
                linewidth=2)
    plt.show()
    time.sleep(1)

    return content_motion


def generate_content_pp():
    # Simple straight line
    # c_x=np.linspace(0,20,num=50).reshape(-1,1) #Generate 1 column array
    # c_y=np.linspace(0,200,num=50).reshape(-1,1)
    # c_z=np.linspace(0,50,num=50).reshape(-1,1)

    # Simple pick&place task randomly generated

    # Select random seed for the generation of the content
    content_motion = []
    content_motion.append([0, 0, 0])

    # First section "pick"
    pick_z_vel = 10;
    num_pick_points = 9
    current_point = copy.deepcopy(content_motion[0])
    # IPython.embed()
    for i in range(num_pick_points):
        current_point[2] = current_point[2] + pick_z_vel
        content_motion.append(copy.deepcopy(current_point))


    # Second section "move"
    num_move_points = 30;

    # Compute distances for x,y,z
    var = 100
    dx = np.clip(np.random.normal(150, var), 0, robot_threshold)
    y_max = np.sqrt(max(0, robot_threshold ** 2 - dx ** 2))
    dy = np.clip(np.random.normal(y_max, var), 0, y_max)
    z_max = np.sqrt(max(0, robot_threshold ** 2 - dx ** 2 - dy ** 2))
    dz = np.clip(np.random.normal(0, var), 0, z_max)

    # Compute x,y,z
    x = dx if random.random() < 0.5 else -dx
    y = dy if random.random() < 0.5 else -dy
    z = dz  # No negative z

    # Generate move section
    for i in range(num_move_points):
        current_point[0] = current_point[0] + x / num_move_points
        current_point[1] = current_point[1] + y / num_move_points
        current_point[2] = current_point[2] + z / num_move_points
        content_motion.append(copy.deepcopy(current_point))

    # Third section "place"
    place_z_vel = 10;
    num_place_points = 10

    for i in range(num_place_points):
        current_point[2] = current_point[2] - place_z_vel
        content_motion.append(copy.deepcopy(current_point))

    return content_motion

def create_hermite_curve(p0,v0,p1,v1):
    # Define constant H
    H = np.array([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, -1, 1],
        [0, 0, 3, -2]
    ])
    """
    Creates a hermite curve between two points with given tangents
    """
    P = np.array([p0,v0,v1,p1]).transpose()
    PH = P @ H
    return lambda t: np.dot(PH, np.array([1,t,t**2,t**3]))

def generate_hermitian_traj(p,v,t_values, input_size=INPUT_SIZE):
    #Get num_points
    num_pairs = np.shape(p)[0]-1

    #Define trajectory
    traj = []
    for i in range(num_pairs):
        curve = create_hermite_curve(p[i], v[i], p[i+1],v[i+1],)
        curve_points = np.asarray([curve(t) for t in t_values[i]])
        for j in range(np.shape(curve_points)[0]):
            traj.append(curve_points[j])

    traj_array = np.asarray(traj)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(1,INPUT_SIZE):
            ax.plot(traj_array[:, 0][i:i + 2],
                    traj_array[:, 1][i:i + 2],
                    traj_array[:, 2][i:i + 2],
                    c=plt.cm.jet(int(np.linalg.norm(traj_array[i]-traj_array[i-1])*255/80)), linewidth=2)
    plt.show()

def generate_base_traj():
    # Scale hermitian velocity >1 angry. 1 means a smooth trajectory.
    v_p =  random.randrange(0,1)

    if v_p < 0.10: #10% of times we have a slow velocity scale
        scale_v = 0.5
    elif v_p < 0.80: #70% of times we have a normal velocity scale
        scale_v = 1
    elif v_p < 0.95: #15% of times we have a fast velocity scale
        scale_v = 2
    else: #5% of times we have a very fast velocity scale
        scale_v = 3

    ## Generate points

    # First we randomly generate the absolute increment between points
    num_points = random.randrange(2,5,1)
    p = []
    p.append([0,0,0])
    v = []
    it = 0
    for i in range(num_points-1):
        ix = random.randrange(0, int(200 / (num_points-1)), 1)
        iy = random.randrange(0, int(200 / (num_points-1)), 1)
        iz = random.randrange(0, int(200 / (num_points-1)), 1)
        v.append([ix*scale_v, iy*scale_v, iz*scale_v])
        p.append([p[i][0]+ix,p[i][1]+iy,p[i][2]+iz])
        it = np.linalg.norm(np.asarray(p[i])-np.asarray(p[i+1])) + it # Total displacement
    v.append([0,0,0]) # Las point velocity 0

    t_values = []
    num_tpoints = 0
    for i in range(num_points-2): # Each segment is assigned points as function of the longitude
        ip = np.linalg.norm(np.asarray(p[i])-np.asarray(p[i+1]))
        num_tpoints += int(ip/it*50)
        t_values.append(np.linspace(0,1,num_tpoints))

    t_values.append(np.linspace(0,1,50-num_tpoints)) # The rest of points are assigned to the last segment

    generate_hermitian_traj(p,v,t_values)

def policy(state):
    sampled_actions = tf.squeeze(actor_model(state))
    sampled_actions = sampled_actions.numpy()

    # We make sure action is within bounds
    legal_action = [max(min(x, upper_bound), lower_bound) for x in
                    sampled_actions]  # Make sure actions are in the desired range
    return list(np.squeeze(legal_action))



# env
content_motion = generate_content_pp()
env = motion_ST_AE.ae_env(content_motion, style_motion, INPUT_SIZE, ae_path)

for ep in range(total_episodes):
    # reward history
    cl_hist = []
    sl_hist = []
    vel_hist = []
    poss_hist = []
    end_poss_hist = []

    while(1):
        generate_base_traj()

    while(1):
        p=[[0,50,20],[50,100,40],[100,200,80]]
        vx=random.normalvariate(50,50*0.1) # Initial velocity for each point. It is in the same units as the position
        vy=random.normalvariate(50,50*0.1)
        vz=random.normalvariate(50,50*0.1)
        v=[[vx,vy,vz],[vx,vy,vz],[vx,vy,vz]]
        print("La velocidad es:", vx,vy,vz)

        #generate_hermitian_traj(p,v)
        time.sleep(1)


    # Init env and generated_motion
    generated_motion = env.reset(content_motion, style_motion)
    episodic_reward = 0

    if ep == 0:
        content_motion = test_trajectories(0)

    elif ep == 1:
        content_motion = test_trajectories(1)

    elif ep == 2:
        content_motion = test_trajectories(2)
    else:
        content_motion = generate_content_pp()

    # Generate Content motion
    tf_content_motion = tf.expand_dims(tf.convert_to_tensor(content_motion), 0)
    # content_motion_input = input_processing.input_generator(content_motion, INPUT_SIZE)

    # Hard coded Generated motion
    #g_x = [i[0] for i in content_motion]
    #g_y = [i[1] for i in content_motion]
    #g_z = [i[2] for i in content_motion]

    ## Define noise

    # Add Style noise
    # noise=[]
    # for i in range(np.shape(style_motion)[0] - 1):
    #    noise.append([x - y for (x, y) in zip(style_motion[i + 1], style_motion[i])])

    # Add sine noise
    #sin_range = np.arange(0, 200, 4)
    #noise = np.sin(sin_range)
    #noise = noise_scale * np.asarray(noise)
    
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
        #print(action)
        
        # Hard coded action
        #escala = generated_scale
        #if step*escala<INPUT_SIZE:
            # action = [-(g_x[(step-1)*escala]-g_x[(step)*escala])+noise[(step-1)*escala][0],-(g_y[(step-1)*escala]-g_y[(step)*escala])+noise[(step-1)*escala][1],-(g_z[(step-1)*escala]-g_z[(step)*escala])+noise[(step-1)*escala][2]] #XYZ noise
            # action = [-(g_x[(step-1)*escala]-g_x[(step)*escala])+noise[(step-1)*escala],-(g_y[(step-1)*escala]-g_y[(step)*escala]),-(g_z[(step-1)*escala]-g_z[(step)*escala])] # X noise
            #action = [-(g_x[(step-1)*escala]-g_x[(step)*escala])+noise[(step-1)],-(g_y[(step-1)*escala]-g_y[(step)*escala]),-(g_z[(step-1)*escala]-g_z[(step)*escala])] #Sine noise
            #action = [-(g_x[(step-1)*escala]-g_x[(step)*escala]),-(g_y[(step-1)*escala]-g_y[(step)*escala]),-(g_z[(step-1)*escala]-g_z[(step)*escala])] # No noise
            #print(action)
            # action = style_motion[step*escala]-style_motion[(step-1)*escala] # Style
            #action = [max(min(x, upper_bound), lower_bound) for x in action]
        	#print(noise[step-1][0])

        	#action = [random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound)]
        	#action = [-x for x in action] #Negate action
        #else:
        #	action = [0,0,0]
        	
        
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
    print("The total loss is", "{:,}".format(episodic_reward))
    print("The CL loss is:", "{:,}".format(np.sum(cl_hist)))
    print("The SL loss is:", "{:,}".format(np.sum(sl_hist)))
    print("The Vel loss is:", "{:,}".format(np.sum(vel_hist)))
    print("The Poss loss is:", "{:,}".format(np.sum(poss_hist)))
    print("The EndPoss loss is:", "{:,}".format(np.sum(end_poss_hist)))
    
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
            ax.plot(content_motion_array[:, 0][i:i + 2], content_motion_array[:, 1][i:i + 2], content_motion_array[:, 2][i:i + 2], c=plt.cm.jet(int(np.linalg.norm(content_motion_array[i]-content_motion_array[i-1])*255/80)), linewidth=2)
            #ax.plot(style_motion_array[:, 0][i:i + 2], style_motion_array[:, 1][i:i + 2], style_motion_array[:, 2][i:i + 2], c=plt.cm.jet(int(np.linalg.norm(style_motion_array[i]-style_motion_array[i-1])*255/80)), linewidth=2)
    plt.show()
