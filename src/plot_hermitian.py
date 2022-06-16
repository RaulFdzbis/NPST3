from audioop import lin2adpcm
from email.base64mime import header_length
import matplotlib.pyplot as plt
import numpy as np

H = np.array([
    [1,0,-3,2],
    [0,1,-2,1],
    [0,0,-1,1],
    [0,0,3,-2]
])

def create_hermite_curve(p0,v0,p1,v1):
    """
    Creates a hermite curve between two points with given tangents
    """
    P = np.array([p0,v0,v1,p1]).transpose()
    PH = P @ H
    return lambda t: np.dot(PH, np.array([1,t,t**2,t**3]))


# Can be 2-D or 3-D
p0 = np.array([0,0])
v0 = np.array([0.5,0])

p1 = np.array([1,1])
v1 = np.array([0, 0.5])

p02 = np.array([1,1])
v02 = np.array([0,0.5])

p12 = np.array([2,0])
v12 = np.array([0, 0.5])

# Create a callabla curve
curve = create_hermite_curve(p0,v0,p1,v1)
curve2 = create_hermite_curve(p02,v02,p12,v12)

# e.g. t = 0.5
print(f"Curve at t = 0.5: {curve(0.5)}")

# plot the curve
t_values = np.linspace(0,1,100)

# generate trajectory points
curve_points = np.asarray([curve(t) for t in t_values])
curve_points2 = np.asarray([curve2(t) for t in t_values])

plt.plot(curve_points[:,0], curve_points[:,1])
#plt.scatter(p0[0],p0[1],color='red')
#plt.scatter(p1[0],p1[1],color='red')

plt.plot(curve_points2[:,0], curve_points2[:,1])
#plt.scatter(p02[0],p02[1],color='red')
#plt.scatter(p12[0],p12[1],color='red')


# plot the tangents
#ARROW_PROPORTION = 0.3  # represented magnitude proportion
#plt.arrow(p0[0],p0[1],ARROW_PROPORTION*v0[0],ARROW_PROPORTION*v0[1],color='red', length_includes_head=True, head_width=0.03*np.linalg.norm(v0), head_length=0.05*np.linalg.norm(v0))
#plt.arrow(p1[0],p1[1],ARROW_PROPORTION*v1[0],ARROW_PROPORTION*v1[1],color='red', length_includes_head=True, head_width=0.03*np.linalg.norm(v1), head_length=0.05*np.linalg.norm(v1))

plt.axis('equal')
plt.show()
