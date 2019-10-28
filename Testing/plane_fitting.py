import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

N_POINTS = 10
TARGET_X_SLOPE = 2
TARGET_y_SLOPE = 3
TARGET_OFFSET  = 5
EXTENTS = 5
NOISE = 5


def project_point(x, y, z, a, b, c):
	d = -1.0

	t =  -(a*x + b*y + c*z + d )/(a*a + b*b + c*c)

	x_prime = a*t + x
	y_prime = b*t + y
	z_prime = c*t + z

	return (x_prime, y_prime, z_prime)


def calc_plane_bis(x, y, z):
	# https://stackoverflow.com/questions/17836880/orthogonal-projection-with-numpy
    a = np.column_stack((x, y, z))
    return np.linalg.lstsq(a, np.ones_like(x))[0]


def project_points(x, y, z, a, b, c):
	"""
	Projects the points with coordinates x, y, z onto the plane
	defined by a*x + b*y + c*z = 1
	"""
	vector_norm = a*a + b*b + c*c
	normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
	point_in_plane = np.array([a, b, c]) / vector_norm

	points = np.column_stack((x, y, z))
	points_from_point_in_plane = points - point_in_plane
	proj_onto_normal_vector = np.dot(points_from_point_in_plane,
	                                 normal_vector)
	proj_onto_plane = (points_from_point_in_plane -
	                   proj_onto_normal_vector[:, None]*normal_vector)

	return point_in_plane + proj_onto_plane


xs = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
ys = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
zs = []
for i in range(N_POINTS):
    zs.append(xs[i]*TARGET_X_SLOPE + \
              ys[i]*TARGET_y_SLOPE + \
              TARGET_OFFSET + np.random.normal(scale=NOISE))

# plot raw data
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(xs, ys, zs, color='b')

# do fit
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

fit2 = calc_plane_bis(xs,ys,zs)


print "solution:"
print "%f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
print "errors:"
print errors
print "residual:"
print residual

x_prime = []
y_prime = []
z_prime = []

for x,y,z in zip(xs,ys,zs):
	x_,y_,z_ = project_points(x,y,z, fit2[0].item(0), fit2[1].item(0), fit2[2].item(0))
	x_prime.append(x_)
	y_prime.append(y_)
	z_prime.append(z_)

ax.scatter(x_prime, y_prime, z_prime, color='r')
# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()