# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                                 Input Processor
# ----------------------------------------------------------------------------------------
import numpy as np
#import IPython


def scale_input(raw_motion, threshold):
    # Scale
    max_value = abs(max(np.max(raw_motion), np.min(raw_motion), key=abs))

    scale_value = max_value / threshold

    if scale_value > 1:
        raw_motion_scaled = raw_motion / scale_value
    else:
        raw_motion_scaled = raw_motion

    # Normalize
    #raw_motion_normalized = raw_motion_scaled / threshold

    return raw_motion_scaled


def dataset_input_generator(dataset, input_size, robot_threshold):
    # Create the inputs for training and testing
    train_data = []
    test_data = []
    # xy_scale = 1

    # Loop over all the motions
    for n_motion in range(np.shape(dataset)[0]):
        motion = np.transpose(dataset[n_motion])
        motion = motion[0::12]  # We work at 10hz dataset at 120hz reduce to work to 10hz
        motion_size = np.shape(motion)[0]
        n_inputs = int(motion_size / input_size)  # We will divide in input_size arrays
        for i in range(n_inputs):
            motion_input = motion[i * input_size:input_size * (1 + i)]  # select a subset of input_size
            motion_input = motion_input - motion_input[0]  # Relatives motions wrt 0
            motion_input = scale_input(motion_input, robot_threshold)
            if n_motion % 6 != 0:
                train_data.append(motion_input)
            else:
                test_data.append(motion_input)  # Every sixth motion is sent to the test dataset

        ims = motion_size % input_size  # ims stands for incomplete motion size
        if ims != 0 and ims > 25: # We only take motions that were at least 50% of the total motion for training
            pad_value = input_size - ims
            motion_input = motion[n_inputs * input_size: n_inputs * input_size + ims]
            motion_input = motion_input - motion_input[0]  # Relatives motions
            motion_input = np.pad(motion_input, ((0, pad_value), (0, 0)), 'constant', constant_values=(0, 0))
            motion_input = scale_input(motion_input, robot_threshold)
            if n_motion % 6 != 0:
                train_data.append(motion_input)
            else:
                test_data.append(motion_input)  # Every sixth motion is sent to the test dataset

    # As arrays
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    return train_data, test_data


def scale_motion_vel(raw_motion, upper_bound, INPUT_SIZE):
    x_vel = 0
    y_vel = 0
    z_vel = 0
    max_vel = upper_bound * INPUT_SIZE
    for i in range(np.shape(raw_motion)[0] - 1):
        x_vel += abs(raw_motion[i][0] - raw_motion[i + 1][0])
        y_vel += abs(raw_motion[i][1] - raw_motion[i + 1][1])
        z_vel += abs(raw_motion[i][2] - raw_motion[i + 1][2])
    max_motion_vel = max(x_vel, y_vel, z_vel)

    scale_value = max_motion_vel / max_vel

    if scale_value > 1:
        raw_motion_scaled = raw_motion / scale_value
    else:
        raw_motion_scaled = raw_motion

    return raw_motion_scaled


def input_generator(input_motion, input_size):
    pad_value = input_size - np.shape(input_motion)[0]
    input_motion = np.pad(input_motion, ((0, pad_value), (0, 0)), 'constant', constant_values=(0, 0))

    input_motion = np.asarray(input_motion)

    return input_motion


def input_trajectory_generator(input_motion, input_size, robot_threshold):
    if np.shape(input_motion)[0] == input_size:
        input_motion = input_motion - input_motion[0]
        input_motion = [np.asarray(input_motion)]
    elif np.shape(input_motion)[0] < input_size:
        input_motion = input_motion - input_motion[0]
        pad_value = input_size - np.shape(input_motion)[0]
        input_motion = np.pad(input_motion, ((0, pad_value), (0, 0)), 'constant', constant_values=(0, 0))
        input_motion = [np.asarray(input_motion)]
    else:  # Is bigger
        motion_size = np.shape(input_motion)[0]
        n_motion = int(motion_size / 50)
        trajectory = []
        for i in range(n_motion):
            int_motion = input_motion[i * input_size: (i + 1) * input_size]
            int_motion = int_motion - int_motion[0]
            trajectory.append(np.asarray(int_motion))
        ims = motion_size % input_size  # ims stands for incomplete motion size
        if ims != 0:
            pad_value = input_size - ims
            last_motion = input_motion[n_motion * input_size: n_motion * input_size + ims]
            last_motion = last_motion - last_motion[0]  # Relatives motions
            last_motion = np.pad(input_motion, ((0, pad_value), (0, 0)), 'constant', constant_values=(0, 0))
            trajectory.append(np.asarray(last_motion))  # if <50 pad to be 50
        input_motion = trajectory

    return input_motion
