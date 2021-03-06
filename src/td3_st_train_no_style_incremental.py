#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                              Training the TD3 network
# TD3 algorithm implemented with the help of [amifunny](https://github.com/amifunny) code 
# ----------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import initializers
from tensorflow.keras import layers
import os
import sys
sys.path.append('../')
from utils import input_processing
from env import motion_ST_AE
from scipy.spatial import distance
import seaborn as sns
import pickle
#import IPython
import random
from operator import add
import argparse

# Arguments
parser = argparse.ArgumentParser(description='Select Style. 0: Happy; 1: Calm, 2: Sad, 3: Angry.')
parser.add_argument('--style', type=int, default=0)
args = parser.parse_args()

# Env parameters
num_actions = 3  # X,Y,Z
INPUT_SIZE = 50
robot_threshold = 300  # in mm
upper_bound = 0.1*robot_threshold
lower_bound = -0.1*robot_threshold
total_episodes = 2500
noise_ep_bound = int(total_episodes * 0.95)
q_noise = 0.002*robot_threshold
action_noise = 0.02*robot_threshold

# Here we define the Actor and Critic networks. `BatchNormalization` is used to normalize dimensions across
# samples in a mini-batch, as activations can vary a lot due to fluctuating values of input state and action.


def get_actor():
    # Inputs: Initialize weights between -3e-3 and 3-e3
    last_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
    content_input = layers.Input(shape=(INPUT_SIZE, 3,))
    generated_input = layers.Input(shape=(INPUT_SIZE, 3,))

    # Content Input
    content_out = layers.Conv1D(256, 5, activation="relu", padding='same')(content_input)
    content_out = layers.BatchNormalization()(content_out)
    content_out = layers.Conv1D(128, 5, activation="relu", padding='same')(content_out)
    content_out = layers.BatchNormalization()(content_out)
    content_out = layers.Conv1D(128, 5, activation="relu", padding='same')(content_out)
    content_out = layers.BatchNormalization()(content_out)
    content_out = layers.Flatten()(content_out)  # Flatten to later concat with action

    # Generated Input
    generated_out = layers.Conv1D(256, 5, activation="relu", padding='same')(generated_input)
    generated_out = layers.BatchNormalization()(generated_out)
    generated_out = layers.Conv1D(128, 5, activation="relu", padding='same')(generated_out)
    generated_out = layers.BatchNormalization()(generated_out)
    generated_out = layers.Conv1D(128, 5, activation="relu", padding='same')(generated_out)
    generated_out = layers.BatchNormalization()(generated_out)
    generated_out = layers.Flatten()(generated_out)  # Flatten to later concat with action

    # Concatenate and pass to a dense layer
    concat = layers.Concatenate()([content_out, generated_out])
    out = layers.Dense(512, activation="relu")(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="relu", kernel_regularizer='l2')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(400, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(300, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound
    model = tf.keras.Model([content_input, generated_input], outputs)
    return model


def get_critic():
    # Inputs: Initialize weights between -3e-3 and 3-e3
    last_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
    action_input = layers.Input(shape=(num_actions,))
    content_input = layers.Input(shape=(INPUT_SIZE, 3,))
    generated_input = layers.Input(shape=(INPUT_SIZE, 3,))

    # Content Input
    content_s_out = layers.Conv1D(256, 5, activation="relu", padding='same')(content_input)
    content_s_out = layers.BatchNormalization()(content_s_out)
    content_s_out = layers.Conv1D(128, 5, activation="relu", padding='same')(content_s_out)
    content_s_out = layers.BatchNormalization()(content_s_out)
    content_s_out = layers.Conv1D(128, 5, activation="relu", padding='same')(content_s_out)
    content_s_out = layers.BatchNormalization()(content_s_out)
    content_s_out = layers.Flatten()(content_s_out)  # Flatten to later concat with action

    # Generated Input
    generated_s_out = layers.Conv1D(256, 5, activation="relu", padding='same')(generated_input)
    generated_s_out = layers.BatchNormalization()(generated_s_out)
    generated_s_out = layers.Conv1D(128, 5, activation="relu", padding='same')(generated_s_out)
    generated_s_out = layers.BatchNormalization()(generated_s_out)
    generated_s_out = layers.Conv1D(128, 5, activation="relu", padding='same')(generated_s_out)
    generated_s_out = layers.BatchNormalization()(generated_s_out)
    generated_s_out = layers.Flatten()(generated_s_out)  # Flatten to later concat with action
    concat = layers.Concatenate()([content_s_out, generated_s_out])

    # State out
    state_out = layers.Dense(512, activation="relu", kernel_regularizer='l2')(concat)
    state_out = layers.BatchNormalization()(state_out)

    # Action out
    action_out = layers.Dense(512, activation="relu", kernel_regularizer='l2')(action_input)
    action_out = layers.BatchNormalization()(action_out)

    # Concatenate action and state
    concat = layers.Concatenate()([state_out, action_out])

    # Out Q(s,a)
    out = layers.Dense(512, activation="relu")(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="relu", kernel_regularizer='l2')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(400, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(300, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1, activation="linear", kernel_initializer=last_init)(out)

    # Outputs q-value
    model = tf.keras.Model([content_input, generated_input, action_input], outputs)

    return model

# Critic for TD3
class CriticTD3(tf.keras.Model):
   def __init__(self):
    # Policy Network
    super(CriticTD3, self).__init__()
    self.critic1 = get_critic()
    self.critic2 = get_critic()

   def call(self, input):
	# Critic return for a given input
    return self.critic1(input), self.critic2(input)


"""
The `Buffer` class implements Experience Replay.
"""

class Buffer:
    def __init__(self, buffer_capacity=10000, batch_size=64):
        # Max experiences
        self.buffer_capacity = buffer_capacity

        # Batch Size
        self.batch_size = batch_size

        # Num times record was called
        self.buffer_counter = 0
        self.policy_freq = 2

        # np.arrays
        self.state_buffer = np.zeros((self.buffer_capacity, 2, INPUT_SIZE, 3))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 2, INPUT_SIZE, 3))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    # Save observation
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    # Compute loss and update parameters
    def learn(self, total_it):
        # Randomly take some observations
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        content_state = tf.convert_to_tensor(self.state_buffer[batch_indices][:, 0])
        generated_state = tf.convert_to_tensor(self.state_buffer[batch_indices][:, 1])
        state_batch = [content_state, generated_state]
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_content_state = tf.convert_to_tensor(self.next_state_buffer[batch_indices][:, 0])
        next_generated_state = tf.convert_to_tensor(self.next_state_buffer[batch_indices][:, 1])
        next_state_batch = [next_content_state, next_generated_state]
        done_batch = self.done_buffer[batch_indices]

        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            noise = np.clip(np.random.normal(size=[self.batch_size, 3], scale=q_noise).astype('float32'), lower_bound/2, upper_bound/2)
            target_actions = np.clip(target_actions + noise, lower_bound, upper_bound)
            q1, q2 = target_critic([next_state_batch, target_actions])
            q = np.minimum(q1, q2)
            y = reward_batch + (1 - done_batch) * gamma * q  # Done is 1 when done
            critic_pred_1, critic_pred_2 = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_pred_1)) + tf.math.reduce_mean(
                tf.math.square(y - critic_pred_2))
            ep_critic_loss.append(critic_loss)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        grad_magnitude_critic = tf.reduce_sum([tf.reduce_sum(g ** 2)
                                               for g in critic_grad]) ** 0.5

        ep_grad_critic.append(grad_magnitude_critic.numpy())

        if total_it % self.policy_freq == 0:
            with tf.GradientTape() as tape:
                actions = actor_model(state_batch)
                critic_value = critic_model([state_batch, actions])  # We only use 1 of the 2
                # We want to maximize the value so -value for the loss
                actor_loss = -tf.math.reduce_mean(critic_value)
                ep_actor_loss.append(actor_loss)

            actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(
                zip(actor_grad, actor_model.trainable_variables)
            )
            grad_magnitude_actor = tf.reduce_sum([tf.reduce_sum(g ** 2)
                                                  for g in actor_grad]) ** 0.5

            ep_grad_actor.append(grad_magnitude_actor.numpy())

            # for debug target actor only
            targetd_action = target_actor(state_batch)
            criticd_value = critic_model([state_batch, targetd_action])  # We only use 1 of the 2
            actord_loss = -tf.math.reduce_mean(criticd_value)
            ep_actor_loss.append(actord_loss)

            ep_target_actor_loss.append(ep_actor_loss)


# Update tau
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for w, variable in enumerate(critic_model.weights):
        new_weights.append(variable * tau + target_variables[w] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for w, variable in enumerate(actor_model.weights):
        new_weights.append(variable * tau + target_variables[w] * (1 - tau))

    target_actor.set_weights(new_weights)


# The policy is defined with some noise following TD3 definition

def policy(state, ep):
    sampled_actions = tf.squeeze(actor_model(state))

    if ep < noise_ep_bound:

        noise = ((noise_ep_bound - ep) / noise_ep_bound) * np.clip(np.random.normal(size=num_actions, scale=action_noise).astype('float32'), lower_bound/2, upper_bound/2)
        sampled_actions = sampled_actions.numpy() + noise

    else:
        sampled_actions = sampled_actions.numpy()

    legal_action = [max(min(x, upper_bound), lower_bound) for x in
                    sampled_actions]

    return list(np.squeeze(legal_action))


"""
## Training hyperparameters and definition of the networks
"""
# actor and critic
actor_model = get_actor()
critic_model = CriticTD3()

# targets
target_actor = get_actor()
target_critic = CriticTD3()

# Init weights equal
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for critic and actor models
critic_lr = 0.00001
actor_lr = 0.000001

# Optimizer for models
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Discount factor and tau
gamma = 0.99
tau = 0.001

buffer = Buffer(10000, 64)

# Load the dataset
with open('../dataset/NPST3_dataset.pickle', 'rb') as data:
    marker_data = pickle.load(data)

# Generate inputs from dataset (arrays)
train_data, test_data = input_processing.dataset_input_generator(marker_data, INPUT_SIZE, robot_threshold)

# To store reward history of each episode
ep_reward_list = []
avg_reward_list = []
hist_value = []
hist_target_value = []
hist_action = []
hist_critic_loss = []
hist_actor_loss = []
hist_target_actor_loss = []
hist_grad_critic = []
hist_grad_actor = []
hist_value_error = []

# Select AE
ae_path = "./../autoencoders/trained-models/autoencoder.h5"

## Extract VICON data
style_data = []
for file in os.listdir("./styles"):
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
        motion_traj = motion_traj[::10] * 1000 # to 10 hz and mm
        motion_traj = input_processing.input_trajectory_generator(motion_traj, INPUT_SIZE, robot_threshold)
    style_data.append(motion_traj)
selected_styles = []
selected_styles.append(style_data[0][4])  # Happy
selected_styles.append(style_data[1][0])  # Calm
selected_styles.append(style_data[2][5])  # Sad
selected_styles.append(style_data[3][2])  # Angry
selected_styles = input_processing.scale_input(selected_styles, robot_threshold) #Scale styles

# Select the Style
style_motion = selected_styles[args.style]
style_motion = style_motion - style_motion[0]

# content init
content_motion = []
# Generate Content motion
content_motion.append([0, 0, 0])

# env
env = motion_ST_AE.ae_env(content_motion, style_motion, INPUT_SIZE, ae_path)

# Total it number
total_it = 0

# Start the training
for ep in range(total_episodes):
    # Select random seed for the generation of the content
    content_seed = [random.uniform(lower_bound / 5, upper_bound / 5), random.uniform(lower_bound / 5, upper_bound / 5),
                    random.uniform(lower_bound / 5, upper_bound / 5)]
    content_motion = []

    # Generate Content motion
    content_motion.append([0,0,0])
    content_motion_input = input_processing.input_generator(content_motion, INPUT_SIZE)

    # Define next step of the content (is one step ahead)
    content_motion.append(list(np.clip(list(map(add, content_motion[0], content_seed)),
                                       -robot_threshold, robot_threshold)))
    if random.random() < 0.02:
        content_seed = [random.uniform(lower_bound / 5, upper_bound / 5),
                        random.uniform(lower_bound / 5, upper_bound / 5),
                        random.uniform(lower_bound / 5, upper_bound / 5)]

    # Generate env
    generated_motion = env.reset(content_motion, style_motion)
    generated_motion_input = input_processing.input_generator(generated_motion, INPUT_SIZE)
    start_state = [content_motion_input, generated_motion_input]
    prev_state = start_state
    episodic_reward = 0

    step = 1
    done = 0
    print("Episode ", ep)
    ep_value = []
    ep_target_value = []
    ep_critic_loss = []
    ep_actor_loss = []
    ep_target_actor_loss = []
    ep_grad_critic = []
    ep_grad_actor = []
    ep_value_error = []
    while True:
        # Get action from critic
        tf_generated_motion = tf.expand_dims(tf.convert_to_tensor(generated_motion_input), 0)
        tf_content_motion = tf.expand_dims(tf.convert_to_tensor(content_motion_input), 0)
        tf_prev_state = [tf_content_motion, tf_generated_motion]

        # Exploration
        if ep > 50:
            action = policy(tf_prev_state, ep)
        else:
            action = [random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound),
                      random.uniform(lower_bound, upper_bound)]

        # Define next step of the content
        if np.shape(content_motion)[0] < INPUT_SIZE:
            content_motion.append(list(np.clip(list(map(add, content_motion[step], content_seed)),
                                          -robot_threshold, robot_threshold)))
        if random.random() < 0.02:
            content_seed = [random.uniform(lower_bound/5, upper_bound/5), random.uniform(lower_bound/5, upper_bound/5),
                      random.uniform(lower_bound/5, upper_bound/5)]

        # Recieve state and reward from environment.
        generated_motion, reward, done = env.step(action, content_motion)  # Step outputs a list for generated

        # Generate state
        generated_motion_input = input_processing.input_generator(generated_motion, INPUT_SIZE)
        content_motion_input = input_processing.input_generator(content_motion, INPUT_SIZE)
        state = [content_motion_input, generated_motion_input]
        buffer.record((prev_state, action, reward, state, done))
        episodic_reward += reward

        # if ep > 1:  # First two episodes only for exploring
        buffer.learn(total_it)
        total_it += 1
        update_target(tau)

        # End this episode when `done` is True
        if done:
            break

        ########################### PARAMS FOR TRAINING TUNING ###########################
        true_action = [content_motion[step][0] - content_motion[step - 1][0],
                  content_motion[step][1] - content_motion[step - 1][1],
                  content_motion[step][2] - content_motion[step - 1][2]]

        true_action = tf.expand_dims(tf.convert_to_tensor(true_action), 0)

        action = tf.expand_dims(tf.convert_to_tensor(action), 0)
        critic_value = critic_model([tf_prev_state, action])
        target_value = target_critic([tf_prev_state, action])
        true_value_action = critic_model([tf_prev_state, true_action])
        ep_value = np.squeeze(critic_value[0].numpy())
        ep_target_value = np.squeeze(target_value[0].numpy())
        ep_value_error = np.squeeze(true_value_action[0].numpy()-critic_value[0].numpy())

        step += 1
        prev_state = state
        print("#", end="")
        sys.stdout.flush()
    print("/")

    content_motion_array = np.asarray(content_motion)
    generated_motion_array = np.asarray(generated_motion)
    hist_value.append(np.mean(ep_value))
    hist_target_value.append(np.mean(ep_target_value))
    hist_critic_loss.append(np.mean(ep_critic_loss))
    hist_actor_loss.append(np.mean(ep_actor_loss))
    hist_target_actor_loss.append(np.mean(ep_target_actor_loss))
    hist_grad_critic.append(np.mean(ep_grad_critic))
    hist_grad_actor.append(np.mean(ep_grad_actor))
    hist_value_error.append(np.mean(ep_value_error))

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Avg Reward is ==> {} * Last reward is {}".format(avg_reward, episodic_reward))
    avg_reward_list.append(avg_reward)

    if ep % 100 == 0:
        # Save the weights
        actor_model.save("actor-back.h5")

########################### PLOT TRAINING PARAMS ###########################
# Cosine distance actions
euclidean_distance = []
cosine_distance = []
for i in range(np.shape(generated_motion_array)[0]):
    if i == 0:
        continue
    else:
        euclidean_distance.append([])
        cosine_distance.append([])
        for j in range(np.shape(generated_motion_array)[0]):
            if j == 0:
                continue
            else:
                ai = generated_motion_array[i] - generated_motion_array[i - 1]
                aj = generated_motion_array[j] - generated_motion_array[j - 1]
                euclidean_distance[i - 1].append(np.linalg.norm(ai - aj))
                cosine_distance[i - 1].append(distance.cosine(ai, aj))

# Euc distance
ax = sns.heatmap(euclidean_distance)
plt.savefig("euc-distance.png")
plt.clf()

# Cosine distance
ax = sns.heatmap(cosine_distance)
plt.savefig("cos-distance.png")
plt.clf()

# reward
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("ep_reward_list")
plt.savefig("ep_reward_list.png")
plt.clf()

# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.savefig("avg-reward.png")
plt.clf()

plt.plot(hist_actor_loss)
plt.xlabel("Step")
plt.ylabel("Actor loss")
plt.savefig("actor-hist.png")
plt.clf()

plt.plot(hist_target_actor_loss)
plt.xlabel("Step")
plt.ylabel("hist_target_actor_loss")
plt.savefig("hist_target_actor_loss.png")
plt.clf()

plt.plot(hist_critic_loss)
plt.xlabel("Step")
plt.ylabel("Critic loss")
plt.savefig("critic-hist.png")
plt.clf()

plt.plot(hist_value)
plt.xlabel("Step")
plt.ylabel("hist_value")
plt.savefig("hist_value.png")
plt.clf()

plt.plot(hist_value_error)
plt.xlabel("Step")
plt.ylabel("hist_value_error")
plt.savefig("hist_value_error.png")
plt.clf()

plt.plot(hist_target_value)
plt.xlabel("Step")
plt.ylabel("target_value")
plt.savefig("target_value.png")
plt.clf()

plt.plot(hist_grad_critic)
plt.xlabel("Step")
plt.ylabel("grad_critic")
plt.savefig("hist_grad_critic.png")
plt.clf()

plt.plot(hist_grad_actor)
plt.xlabel("Step")
plt.ylabel("grad_actor")
plt.savefig("hist_grad_actor.png")
plt.clf()

# Save the weights
actor_model.save("actor.h5")
target_actor.save("target_actor.h5")

