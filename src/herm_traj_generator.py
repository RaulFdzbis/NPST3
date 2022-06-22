# ----------------------------------------------------------------------------------------
#                       Neural Policy Style Transfer TD3 (NPST3) 
#                             Hermitian Trajectory Generator
# ----------------------------------------------------------------------------------------
import numpy as np
import random
import matplotlib.pyplot as plt

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

def nerv_noise(nerv_s): #Return random noise
    return [random.uniform(-nerv_s,nerv_s),random.uniform(-nerv_s,nerv_s),random.uniform(-nerv_s,nerv_s)]

def generate_hermitian_traj(p,v,t_values, traj_type, input_size, vel_threshold):
    #Get num_points
    num_pairs = np.shape(p)[0]-1

    #Define trajectory
    traj = []
    for i in range(num_pairs):
        curve = create_hermite_curve(p[i], v[i], p[i+1],v[i+1])
        curve_points = np.asarray([curve(t) for t in t_values[i]])

        for j in range(np.shape(curve_points)[0]):
            traj.append(curve_points[j])

    # Increase traj velocity
    if traj_type == "slow":
        escala = 0.5
    elif traj_type == "normal":
        escala = 1
    elif traj_type == "fast":
        escala = 2
    elif traj_type == "veryfast":
        escala = 3 #Maybe 5 if we want faster

    nervous = 0

    if (traj_type != "veryfast" and random.random()<0.2): # 20% of the trajectories are nervous ones
        nervous =1

    # The noise scale is random and defined as a function of the vel scale
    nerv_generator_scale = random.randrange(1,10,1)*escala
    #print("NOISE SCALE IS:", nerv_generator_scale)

    traj_escaled = []
    traj_escaled.append([0,0,0])
    if escala == 0.5:
        for i in range(1,input_size):
            if i%2 == 0:
                tmp_traj = traj[int(i/2)]
                if nervous == 0:
                    traj_escaled.append(tmp_traj)
                else:
                    traj_escaled.append([x+y for x, y in zip(tmp_traj, nerv_noise(nerv_generator_scale))])
            else:
                tmp_traj = [(x+y)/2 for x, y in zip(traj[int((i-1)/2)], traj[int((i-1)/2+1)])]
                if nervous == 0:
                    traj_escaled.append(tmp_traj)
                else:
                    traj_escaled.append([x+y for x, y in zip(tmp_traj, nerv_noise(nerv_generator_scale))])
    else:
        for i in range(1,input_size):
            if (i*escala) < (input_size-1):
                ix = traj[i*escala][0]-traj_escaled[-1][0]
                iy = traj[i*escala][1]-traj_escaled[-1][1]
                iz = traj[i*escala][2]-traj_escaled[-1][2]
                if nervous ==1:
                    nerv_noise_value = nerv_noise(nerv_generator_scale)
                    ix += nerv_noise_value[0]
                    iy += nerv_noise_value[1]
                    iz += nerv_noise_value[2]
                while(abs(ix) > vel_threshold or  abs(iy) > vel_threshold or  abs(iz)> vel_threshold): #Make sure not outside the max vel
                    ix -= np.sign(ix) # Reduce abs value by 1
                    iy -= np.sign(iy)
                    iz -= np.sign(iz)
                #print(ix,iy,iz)
                traj_escaled.append([x + y for x, y in zip(traj_escaled[-1], [ix,iy,iz])])
            else:
                traj_escaled.append(traj[-1])

    traj_array = np.asarray(traj_escaled)
    #traj_array2 = np.asarray(traj)



    fig = plt.figure()
    ax = fig.gca(projection='3d')
    total_vel = 0
    num_vel_points = 0
    for i in range(0,input_size-1):
            ax.plot(traj_array[:, 0][i:i + 2],
                    traj_array[:, 1][i:i + 2],
                    traj_array[:, 2][i:i + 2],
                    c=plt.cm.jet(int(np.linalg.norm(traj_array[i]-traj_array[i+1])*255/52)), linewidth=2)
            current_vel = np.linalg.norm(traj_array[i]-traj_array[i+1])*10
            #print("Current velocity (mm/s) is: ", current_vel)
            if current_vel != 0.0:
                total_vel += current_vel
                num_vel_points += 1

    print("Average velocity is (mm/s):", total_vel/num_vel_points)
    for i in range(np.shape(p)[0]):
        ax.scatter(p[i][0], p[i][1], p[i][2],color = 'red')

    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(0,INPUT_SIZE-1):
            ax.plot(traj_array2[:, 0][i:i + 2],
                    traj_array2[:, 1][i:i + 2],
                    traj_array2[:, 2][i:i + 2],
                    c=plt.cm.jet(int(np.linalg.norm(traj_array2[i]-traj_array2[i+1])*255/52)), linewidth=2)
            print("Current velocity2 (mm/s) is: ", np.linalg.norm(traj_array2[i] - traj_array2[i+1]) * 10)
    for i in range(np.shape(p)[0]):
        ax.scatter(p[i][0], p[i][1], p[i][2],color = 'red')
    '''

    plt.show()




def generate_base_traj(input_size, robot_threshold, vel_threshold):
    # Scale hermitian velocity >1 angry. 1 means a smooth trajectory.
    v_p =  random.random()

    if v_p < 0.2: #20% of times we have a slow velocity scale
        scale_v = 0.5
        traj_type = "slow"
        print("Trayectoria Lenta")
    elif v_p < 0.60: #40% of times we have a normal velocity scale
        scale_v = 1
        traj_type = "normal"
        print("Trayectoria Normal")
    elif v_p < 0.85: #25% of times we have a fast velocity scale
        scale_v = 2
        traj_type = "fast" #15% of times we have a fast velocity scale
        print("Trayectoria Rapida")
    else: #5% of times we have a very fast velocity scale
        scale_v = 3
        traj_type = "veryfast"
        print("Trayectoria Muy Rapida")

    ## Generate points

    # First we randomly generate the absolute increment between points
    num_points = random.randrange(2,5,1)
    p = []
    p.append([0,0,0])
    v = []
    it = 0
    for i in range(num_points-1): # Length
        ix = random.randrange(0, int(400 / (num_points-1)), 1)
        iy = random.randrange(0, int(400 / (num_points-1)), 1)
        iz = random.randrange(0, int(400 / (num_points-1)), 1)
        if random.random()<0.5: # We make negative moves with a 30% probability (to avoid lot of changes in direction)
            ix=-ix
        if random.random()<0.5:
            iy=-iy
        if random.random()<0.5:
            iz=-iz

        # Make sure we are not out of limits, if we are reduce increments
        while p[i][0]+ix >= robot_threshold or p[i][1]+iy >= robot_threshold or p[i][2]+iz>= robot_threshold:
            ix=int(ix/2)
            iy=int(iy/2)
            iz=int(iz/2)
        v.append([ix*scale_v, iy*scale_v, iz*scale_v]) # Tangent velocity
        p.append([p[i][0]+ix,p[i][1]+iy,p[i][2]+iz])
        it = np.linalg.norm(np.asarray(p[i])-np.asarray(p[i+1])) + it # Total displacement
    v.append([0,0,0]) # Las point velocity 0

    t_values = []
    total_tpoints = 0
    for i in range(num_points-2): # Each segment is assigned points as function of the longitude
        ip = np.linalg.norm(np.asarray(p[i])-np.asarray(p[i+1])) # Segment Longitude
        num_tpoints = int((ip/it)*50) # Number of points assigned as a function of the longitude
        total_tpoints += num_tpoints
        t_values.append(np.linspace(0,1,num_tpoints+1)) # 1 point more to remove the last point (see below)
        np.delete(t_values[i], -1) # Remove the last point since it will be the first of the next segment

    t_values.append(np.linspace(0,1,50-total_tpoints)) # The rest of points are assigned to the last segment

    generate_hermitian_traj(p,v,t_values, traj_type, input_size, vel_threshold)
