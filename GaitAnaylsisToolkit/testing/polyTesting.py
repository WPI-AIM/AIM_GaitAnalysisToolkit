import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
from LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from LearningTools.Runner import GMMRunner, TPGMMRunner, TPGMMRunner_old
from random import seed
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly



def coef(b, dt):
    A = np.array([[1.0, 0, 0, 0, 0],
                  [0, 1.0, 0, 0, 0],
                  [0, 0.0, 1.0, 0, 0],
                  [1.0, dt, dt ** 2, dt ** 3, dt ** 4],
                  [0, 1.0, 2 * dt, 3 * dt ** 2, 4*dt ** 3],
                  [0, 0.0, 2 , 6 * dt , 12*dt ** 2]])

    return np.linalg.pinv(A).dot(b)


# seed random number generator
seed(1)


b = np.array([[-3], [0.0], [0.0], [-6.0], [0.0],[0.0]])
x =  [0.5, -.3, .01, -.01 ] #coef(b, 10)
# x = [ 1.07651173e+04, -4.31074513e+05,  7.74094261e+06, -8.27680899e+07,
#       5.90144167e+08, -2.97745894e+09,  1.10017021e+10, -3.03702879e+10,
#       6.32560271e+10, -9.95654312e+10,  1.17644639e+11, -1.02628004e+11,
#       6.40749897e+10, -2.70669375e+10,  6.92657070e+09, -8.10505668e+08]

fit = poly.Polynomial(x)
t = np.linspace(-10, 10, 100)
y_prime = fit(t)
hip = []
for i in range(4):
    y = y_prime + gauss(-2, 2)
    #y = np.append(y, np.flip(y))
    hip.append(y)


b = np.array([[-3], [0.0], [0.0], [-6.0], [0.0],[0.0]])
x =  [0.5, -.3, .01, -.01 ] #coef(b, 10)
fit = poly.Polynomial(x)
t = np.linspace(-10, 10, 100)
y_prime = fit(t)
knee = []
for i in range(4):
    y = y_prime + gauss(-2, 2)
    #y = np.append(y, np.flip(y))
    knee.append(y)


trajs = [hip]
trainer = TPGMMTrainer.TPGMMTrainer(demo=trajs, file_name="poly9", n_rf=15, dt=0.01, reg=1e-5, poly_degree=[5,5])
trainer.train()
runner = TPGMMRunner.TPGMMRunner("poly9.pickle")
path = np.array(runner.run())

fig, (ax1, ax2) = plt.subplots(1, 2)

for p in hip:
    ax1.plot(p)

for p in knee:
    ax2.plot(p)

ax1.plot(path[:, 0], linewidth=4)
#ax2.plot(path[:, 1], linewidth=4)
plt.show()


