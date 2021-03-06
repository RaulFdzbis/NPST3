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
from mpl_toolkits.mplot3d import Axes3D
#import IPython
import copy


class ae_env():
    def __init__(self, content_motion, style_motion, input_size, ae_path, robot_threshold=300):
        # Init the desired motions
        self.content_motion = np.asarray(content_motion)
        self.style_motion = np.asarray(style_motion)
        self.generated_motion = []
        self.generated_motion.append(list(content_motion[0]))  # Init position
        self.done = 0
        self.input_size = input_size
        self.robot_threshold = robot_threshold
        # Style Transfer and constraints weights
        self.wc = 100
        self.ws = 1
        self.wp = 1
        self.wv = 20
        self.wpc = 0.1
        # Debug
        self.tcl = 0
        self.tsl = 0
        self.tpl = 0
        self.tvl = 0

        # Load AE model
        self.autoencoder = load_model(ae_path)  # Actor
        self.ae_outputs = K.function([self.autoencoder.input], [self.autoencoder.layers[2].output])

    def reset(self, content_motion, style_motion):
        self.content_motion = np.asarray(content_motion)
        self.style_motion = np.asarray(style_motion)
        self.generated_motion = []
        self.generated_motion.append(content_motion[0])  # Init position
        self.done = 0
        # Debug
        self.tcl = 0
        self.tsl = 0
        self.tpl = 0
        self.tvl = 0

        return self.generated_motion

    def content_loss(self, content_outputs, generated_outputs):
        # Compute normalized loss
        cl = np.mean((np.squeeze(generated_outputs)/ self.robot_threshold - np.squeeze(content_outputs)/ self.robot_threshold) ** 2)

        return cl

    def style_loss(self, style_outputs, generated_outputs):
        # For the Style loss the Gram Matrix of the AE is computed
        # Get generated Gram Matrix 
        squeezed_generated_outputs = np.squeeze(generated_outputs)/ self.robot_threshold
        gram_generated = np.dot(squeezed_generated_outputs, np.transpose(squeezed_generated_outputs))

        # Get style Gram Matrix 
        squeezed_style_outputs = np.squeeze(style_outputs)/ self.robot_threshold
        gram_style = np.dot(squeezed_style_outputs, np.transpose(squeezed_style_outputs))

        # Compute loss
        sl = np.mean((gram_generated - gram_style) ** 2)

        return sl

    def compute_reward(self):
        # Generated motion outputs for both cl and sl
        num_points = np.shape(self.generated_motion)[0]
        input_motion = input_processing.input_generator(self.generated_motion,
                                                        self.input_size)  # generated_motion to NN friendly array for input

        # Obtain the corresponding content and style vectors 
        eq_content_motion = input_processing.input_generator(self.content_motion[0:num_points], self.input_size)
        eq_style_motion = input_processing.input_generator(self.style_motion[0:num_points], self.input_size)
		
		# Content and Style outputs
        content_outputs = self.ae_outputs([np.expand_dims(eq_content_motion, axis=0)])
        style_outputs = self.ae_outputs([np.expand_dims(eq_style_motion, axis=0)])

		# Generated outputs
        input_motion = np.expand_dims(input_motion, axis=0)
        generated_outputs = self.ae_outputs([input_motion])

		# Compute losses
        cl = self.content_loss(content_outputs, generated_outputs)
        sl = self.style_loss(style_outputs, generated_outputs)

        # Velocity constraint
        gen_velocity = np.asarray(self.generated_motion[num_points - 1]) - np.asarray(
            self.generated_motion[num_points - 2])
        style_velocity = self.style_motion[num_points - 1] - self.style_motion[num_points - 2]
        vel_loss = np.mean((abs(gen_velocity)/self.robot_threshold - abs(style_velocity)/self.robot_threshold) ** 2)

        # Position constraint
        pos_loss_cont = np.mean((np.asarray(self.generated_motion[num_points - 1]) / self.robot_threshold - self.content_motion[num_points - 1] / self.robot_threshold) ** 2)
		
		# End position constraint
        if np.shape(self.generated_motion)[0] == self.input_size:
           pos_loss = np.mean((np.asarray(self.generated_motion[-1])/ self.robot_threshold - self.content_motion[-1]/ self.robot_threshold) ** 2)

        else:
           pos_loss = 0

		# Total reward
        comp_reward = -(self.wc * cl + self.ws * sl + self.wp * pos_loss + self.wv * vel_loss + pos_loss_cont * self.wpc)

		# Debug for training
        self.tcl += cl * self.wc
        self.tsl += sl * self.ws
        self.tpl += pos_loss * self.wp
        self.tvl += vel_loss * self.wv

        if np.shape(self.generated_motion)[0] == self.input_size:
            print("totals losses are: ", self.tcl, self.tsl, self.tpl, self.tvl)
            print("WARNING: CONT POSS ALSO ADDED IN THIS VERSION")
        return comp_reward

    def step(self, step_action, content_motion):  # Step outpus a list for generated
        self.content_motion = np.asarray(content_motion)
        self.generated_motion.append(list(
            np.clip(list(map(add, self.generated_motion[-1], step_action)), -300, 300)))  # add next step

        step_reward = self.compute_reward()

        if np.shape(self.generated_motion)[0] == self.input_size:
            self.done = 1

        return self.generated_motion, step_reward, self.done


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
