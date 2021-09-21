
import numpy as np
from pyquaternion import Quaternion
import numpy as np







def displacement(v,u):

    temp = np.array([0.0,0.0,0.0])
    magU = np.linalg.norm(u)
    if v >= 0:
        temp =  2*np.arccos(v)
    elif v < 0:
        temp =  -2*np.arccos(-v)

    return temp*u/magU

if __name__ == '__main__':

    my_quaternion = Quaternion( -0.7134254 , 0.3648376, 4.7021186, 0.3648376 )
    # my_quaternion = my_quaternion.normalised
    print(Quaternion.log(my_quaternion*my_quaternion.conjugate).vector)