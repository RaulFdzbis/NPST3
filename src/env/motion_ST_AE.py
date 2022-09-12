# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                      Franka Emika Panda robot environment model
# ----------------------------------------------------------------------------------------

from keras.models import load_model
import numpy as np
import pickle
from keras import backend as K
import sys

sys.path.append('../../')
from utils import input_processing
from operator import add
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
from dtw import *
import IPython
from scipy.spatial.distance import cdist

class ae_env():
    def __init__(self, content_motion, style_motion, input_size, ae_path, robot_threshold=300):
        # Init the desired motions
        self.content_motion = np.asarray(content_motion)
        self.style_motion = np.asarray(style_motion)
        self.input_size = input_size

        # Compute avg style/content velocity
        self.style_velocity = 0
        self.content_velocity = 0
        content_size = 0
        for i in range(1, input_size):
            self.style_velocity = self.style_velocity + np.linalg.norm(self.style_motion[i - 1] - self.style_motion[i])
            self.content_velocity = self.content_velocity + np.linalg.norm(self.content_motion[i-1]-self.content_motion[i])
            if np.linalg.norm(self.content_motion[i-1]-self.content_motion[i]) != 0:
                content_size +=1
        self.style_velocity = self.style_velocity / input_size
        self.content_velocity = self.content_velocity / content_size

        self.generated_motion = []
        self.generated_motion.append(list(content_motion[0]))  # Init position
        self.done = 0
        self.robot_threshold = robot_threshold
        self.vel_threshold = robot_threshold*0.1
        # Style Transfer and constraints weights
        self.wc = 20 # Ref tabla loss 2
        self.ws = 0.2 # Ref tabla loss 0.02
        self.wp = 50 # End pos Ref tabla loss 100
        self.wv = 0.0001 # Ref tabla loss 0.1
        self.wpc = 0.005*(1e-6) # DTW pos Ref tabla loss 0.1*(1e-5)

        # Debug
        self.tcl = 0
        self.tsl = 0
        self.tpl = 0
        self.tvl = 0

        # Load AE model
        self.autoencoder = load_model(ae_path)  # Actor
        self.ae_outputs = K.function([self.autoencoder.input], [self.autoencoder.layers[2].output])
        # Content and Style outputs
        input_content_motion = input_processing.input_generator(content_motion,input_size)  # To NN friendly array for input
        input_style_motion = input_processing.input_generator(style_motion,input_size)  # To NN friendly array for input
        self.content_outputs = self.ae_outputs([np.expand_dims(input_content_motion, axis=0)])
        self.style_outputs = self.ae_outputs([np.expand_dims(input_style_motion, axis=0)])

    def reset(self, content_motion, style_motion):
        self.content_motion = np.asarray(content_motion)
        self.style_motion = np.asarray(style_motion)

        # Compute avg style/content velocity
        self.style_velocity = 0
        self.content_velocity = 0
        content_size = 0
        for i in range(1, self.input_size):
            self.style_velocity = self.style_velocity + np.linalg.norm(self.style_motion[i - 1] - self.style_motion[i])
            self.content_velocity = self.content_velocity + np.linalg.norm(self.content_motion[i-1]-self.content_motion[i])
            if np.linalg.norm(self.content_motion[i-1]-self.content_motion[i]) != 0:
                content_size +=1
        self.style_velocity = self.style_velocity / self.input_size
        self.content_velocity = self.content_velocity / content_size


        # Content and Style outputs
        input_content_motion = input_processing.input_generator(content_motion,self.input_size)  # To NN friendly array for input
        input_style_motion = input_processing.input_generator(style_motion,self.input_size)  # To NN friendly array for input
        self.content_outputs = self.ae_outputs([np.expand_dims(input_content_motion, axis=0)])
        self.style_outputs = self.ae_outputs([np.expand_dims(input_style_motion, axis=0)])

        self.generated_motion = []
        self.generated_motion.append(content_motion[0])  # Init position
        self.done = 0
        # Debug
        self.tcl = 0
        self.tsl = 0
        self.tpl = 0
        self.tvl = 0

        return self.generated_motion

    def content_loss(self, generated_outputs):
        # Compute normalized loss
        cl = np.mean((np.squeeze(generated_outputs) / self.robot_threshold - np.squeeze(
            self.content_outputs) / self.robot_threshold) ** 2)

        return cl

    def style_loss(self, generated_outputs):
        # For the Style loss the Gram Matrix of the AE is computed
        # Get generated Gram Matrix 
        squeezed_generated_outputs = np.squeeze(generated_outputs) / self.robot_threshold
        gram_generated = np.dot(squeezed_generated_outputs, np.transpose(squeezed_generated_outputs))

        # Get style Gram Matrix 
        squeezed_style_outputs = np.squeeze(self.style_outputs) / self.robot_threshold
        gram_style = np.dot(squeezed_style_outputs, np.transpose(squeezed_style_outputs))

        # Compute loss
        sl = np.mean((gram_generated - gram_style) ** 2)

        return sl

    def compute_reward(self):
        # Generated motion outputs for both cl and sl
        num_points = np.shape(self.generated_motion)[0]


        # IPython.embed()
        # Compute DTW loss
        alignment = dtw(self.generated_motion, self.content_motion, keep_internals=True)
        pos_loss_cont = alignment.distance

        #Compute vel loss
        gen_velocity = 0
        gen_points = 0
        for i in range(1, num_points):
            #print("Num points: ", i)
            vel_i = np.linalg.norm(np.asarray(self.generated_motion[i]) - np.asarray(self.generated_motion[i - 1]))
            if abs(vel_i) > (self.style_velocity)*0.1:  # 10Hz
                gen_velocity = gen_velocity + vel_i
                gen_points += 1
            else:
                print("PARADA!!")
        #IPython.embed()
        if gen_points != 0: # If not stopped all the time
            gen_velocity = gen_velocity / gen_points

        vel_loss = abs(gen_velocity - self.style_velocity) / self.robot_threshold

        ## dtw compute Content and Style loss ##
        wq = warp(alignment, index_reference=False)  # Find the warped trajectory
        warped_g = np.asarray(self.generated_motion)[wq]
        warped_g = np.append(warped_g, [np.asarray(self.generated_motion)[-1]],
                             axis=0)  # Add last point to warping (warp dont do this)
        if (np.shape(warped_g)[0] > 50):
            print("ERROR WARPED TRAJECTORY TOO LONG")
        warped_traj = [[0, 0, 0]]
        for index in range(np.shape(warped_g)[0] - 1):
            warped_ix = warped_g[index + 1][0] - warped_traj[index][0]
            warped_iy = warped_g[index + 1][1] - warped_traj[index][1]
            warped_iz = warped_g[index + 1][2] - warped_traj[index][2]
            # print("Incrementos: ", warped_ix, warped_iy, warped_iz)
            while (abs(warped_ix) > self.vel_threshold or
                   abs(warped_iy) > self.vel_threshold or
                   abs(warped_iz) > self.vel_threshold):  # Make sure not outside the max vel
                warped_ix -= np.sign(warped_ix)  # Reduce abs value of ix by 1
                warped_iy -= np.sign(warped_iy)
                warped_iz -= np.sign(warped_iz)
            warped_traj.append([x + y for x, y in zip(warped_traj[index], [warped_ix, warped_iy, warped_iz])])

        input_generated_motion_content = input_processing.input_generator(warped_traj,
                                                                          self.input_size)  # generated_motion to NN friendly array for input
        input_generated_motion_style = input_processing.input_generator(self.generated_motion, self.input_size)
        # Plot for debug purposes
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 3, 1, projection='3d')
        # ax.plot(self.content_motion[:, 0], self.content_motion[:, 1], self.content_motion[:, 2], label="content")
        #
        # ax.plot(generated_motion_arr[:, 0][alignment.index1], generated_motion_arr[:, 1][alignment.index1], generated_motion_arr[:, 2][alignment.index1], label="content")
        # plt.show()

        # Generated outputs
        input_generated_motion_content = np.expand_dims(input_generated_motion_content, axis=0)
        input_generated_motion_style = np.expand_dims(input_generated_motion_style, axis=0)

        # print("Input generated motion is: ", input_generated_motion)
        generated_outputs_content = self.ae_outputs([input_generated_motion_content])
        generated_outputs_style = self.ae_outputs([input_generated_motion_style])

        # Generated raw outputs (No warp)
        # input_generated_motion_2 = input_processing.input_generator(self.generated_motion, self.input_size)  # generated_motion to NN friendly array for input
        # input_generated_motion_2 = np.expand_dims(input_generated_motion_2, axis=0)
        # generated_outputs_2 = self.ae_outputs([input_generated_motion_2])

        # cl and sl
        cl = self.content_loss(generated_outputs_content)
        sl = self.style_loss(generated_outputs_style)

        # End position constraint
        if np.shape(self.generated_motion)[0] == self.input_size:
            pos_loss = np.mean((np.asarray(self.generated_motion[-1]) / self.robot_threshold - self.content_motion[
                -1] / self.robot_threshold) ** 2)

            #IPython.embed()

        else:
            pos_loss = 0
            warped_traj = 0

        # Time step
        n_timestep = num_points

        # Total reward
        comp_reward = -(self.wc * n_timestep * cl + self.ws * n_timestep * sl +
                        self.wp * pos_loss + self.wv * n_timestep * vel_loss +
                        pos_loss_cont * n_timestep * self.wpc)
        # print("Velocity reward is: ", self.wv * n_timestep * vel_loss)
        # print("Content reward is: ", self.wc * n_timestep * cl)
        # print("Style reward is: ", self.ws * n_timestep * sl)
        # print("DTW reward is: ", n_timestep * self.wpc * pos_loss_cont)

        # Debug for training
        self.tcl += cl * self.wc
        self.tsl += sl * self.ws
        self.tpl += pos_loss * self.wp
        self.tvl += vel_loss * self.wv

        # if np.shape(self.generated_motion)[0] == self.input_size:
        #    print("totals losses are: ", self.tcl, self.tsl, self.tpl, self.tvl)
        #    print("WARNING: CONT POSS ALSO ADDED IN THIS VERSION")
        return comp_reward, self.wc * n_timestep * cl, self.ws * n_timestep * sl, \
               self.wv * n_timestep * vel_loss, n_timestep * self.wpc * pos_loss_cont, \
               self.wp * pos_loss, warped_traj

    def step(self, step_action, content_motion):  # Step outpus a list for generated
        self.content_motion = np.asarray(content_motion)
        self.generated_motion.append(list(
            np.clip(list(map(add, self.generated_motion[-1], step_action)), -300, 300)))  # add next step

        step_reward, cl, sl, vel_loss, pos_loss_cont, pos_loss, warped_traj = self.compute_reward()

        if np.shape(self.generated_motion)[0] == self.input_size:
            self.done = 1

        return self.generated_motion, step_reward, cl, sl, vel_loss, pos_loss_cont, pos_loss, self.done, warped_traj


if __name__ == "__main__":

    INPUT_SIZE = 50
    robot_threshold_test = 300  # in mm
    upper_bound = 0.1 * robot_threshold_test
    lower_bound = -0.1 * robot_threshold_test
    ae_path = "./../../autoencoders/trained-models/autoencoder.h5"

    # Load dataset
    with open('../../dataset/NPST3_dataset.pickle', 'rb') as data:
        marker_data = pickle.load(data)

    # Extract data from the dataset
    train_data, test_data = input_processing.dataset_input_generator(marker_data, INPUT_SIZE, robot_threshold_test)

    # Define initial content and style motion
    content_motion = np.loadtxt("./../test-trajectories/line-test.txt", delimiter=" ")  # WARNING: array of arrays
    content_motion = content_motion - content_motion[0]
    content_motion = content_motion * robot_threshold_test
    line_motion = copy.deepcopy(content_motion)
    tf_content_motion = tf.expand_dims(tf.convert_to_tensor(content_motion), 0)

    style_motion = np.loadtxt("./../test-trajectories/sine-test.txt", delimiter=" ")
    style_motion = style_motion - style_motion[0]
    style_motion = style_motion * robot_threshold_test
    tf_style_motion = tf.expand_dims(tf.convert_to_tensor(style_motion), 0)

    # Load Autoencoder env
    env = ae_env(content_motion, style_motion, 50, ae_path)

    num_ep = 20
    for i in range(num_ep):
        step = 0
        if i == 1 or i == 7:
            content_motion[:, 2] = -content_motion[:, 2]
        elif i == 2 or i == 8:
            content_motion[:, 2] = -content_motion[:, 2]
            content_motion[:, 1] = -content_motion[:, 1]
        elif i == 3 or i == 9:
            content_motion[:, 1] = -content_motion[:, 1]
            content_motion[:, 0] = -content_motion[:, 0]
        elif i == 4 or i == 10:
            content_motion[:, 0] = -content_motion[:, 0]
            content_motion = content_motion / 2
        elif i == 5 or i == 11:
            content_motion = style_motion
        elif i == 6:
            line_motion = copy.deepcopy(content_motion)

        generated_motion = env.reset(content_motion, style_motion)
        generated_motion = input_processing.input_generator(generated_motion, INPUT_SIZE)
        total_reward = 0
        total_cl = 0
        total_sl = 0

        while True:
            step += 1
            action = [line_motion[step][0] - line_motion[step - 1][0],
                      line_motion[step][1] - line_motion[step - 1][1],
                      line_motion[step][2] - line_motion[step - 1][2]]

            legal_action = [max(min(x, upper_bound), lower_bound) for x in action]
            generated_motion, reward, done = env.step(legal_action)

            total_reward = total_reward + reward

            if done:
                break

            # Debug and plotting
        generated_motion = input_processing.input_generator(env.generated_motion, INPUT_SIZE)  # For the plot
        print(generated_motion[0])
        print(content_motion[0])
        print("The total reward is: ", total_reward)

        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.set_xlim([-robot_threshold_test, robot_threshold_test])
        ax.set_ylim([-robot_threshold_test, robot_threshold_test])
        ax.set_zlim([-robot_threshold_test, robot_threshold_test])
        ax.plot(env.content_motion[:, 0], env.content_motion[:, 1], env.content_motion[:, 2], label='content')

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.set_xlim([-robot_threshold_test, robot_threshold_test])
        ax.set_ylim([-robot_threshold_test, robot_threshold_test])
        ax.set_zlim([-robot_threshold_test, robot_threshold_test])
        ax.plot(style_motion[:, 0], style_motion[:, 1], style_motion[:, 2], label='style')

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.set_xlim([-robot_threshold_test, robot_threshold_test])
        ax.set_ylim([-robot_threshold_test, robot_threshold_test])
        ax.set_zlim([-robot_threshold_test, robot_threshold_test])
        ax.plot(generated_motion[:, 0], generated_motion[:, 1], generated_motion[:, 2], label='generated')
        plt.show()
