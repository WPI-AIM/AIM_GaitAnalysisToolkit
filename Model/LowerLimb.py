import numpy as np
from Model import Model
import matplotlib.pyplot as plt


class LowerLimb(Model.Model):

    def __init__(self, mass, height):

        super(LowerLimb, self).__init__(mass, height)
        self.make_body()

    def IK(self, toe, hip_location=[0, 0], ankle_contraint=False):

        l1 = self.lengths["thigh"]
        l2 = self.lengths["shank"]
        l3 = self.lengths["ankle"]
        l4 = self.lengths["foot_length"]

        x = toe[0] - hip_location[0] - abs(l4)
        y = toe[1] - hip_location[1] + abs(l3)

        num = x*x + y*y - l1**2 - l2**2
        dem = 2*l1*l2

        q2 = np.arctan2(-np.sqrt(1 - (num / dem)**2 ), (num / dem) )
        q2 = np.nan_to_num(q2)
        q1 = np.nan_to_num(np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1 + l2*np.cos(q2))) + 0.5*np.pi
        q3 = np.nan_to_num(2*np.pi - q1 - q2)

        return [q1, q2, q3]

    def FK(self, q, hip_location=[0,0]):

        ox = hip_location[0]
        oy = hip_location[1]
        l1 = self.lengths["thigh"]
        l2 = self.lengths["shank"]
        l3 = self.lengths["ankle"]
        l4 = self.lengths["foot_length"]
        q[0] = q[0]-0.5*np.pi

        o1y = round(oy + l1*np.sin(q[0]),2)
        o1x = round(ox + l1*np.cos(q[0]),2)

        o2y = round(o1y + l2*np.sin(q[0] + q[1]),2)
        o2x = round(o1x + l2*np.cos(q[0] + q[1]),2)

        o3y = round(o2y + l3*np.sin(q[0] + q[1] + q[2]),2)
        o3x = round(o2x + l3*np.cos(q[0] + q[1] + q[2]),2)

        o4y = round(o3y + l4 * np.sin(np.pi/2 + q[0] + q[1] + q[2]),2)
        o4x = round(o3x + l4 * np.cos(np.pi/2 + q[0] + q[1] + q[2]),2)

        x = [ox, o1x, o2x, o3x, o4x]
        y = [oy, o1y, o2y, o3y, o4y]

        return x, y

    def get_total_leg_length(self):
        return abs(self.lengths["thigh"] + self.lengths["shank"] + self.lengths["ankle"])



if __name__ == '__main__':
    model = LowerLimb(56, 1.6)

    q = [-0, 0, 0]

    x, y = model.FK(q)
    q = model.IK([x[-3], y[-3]])
    x2, y2 = model.FK(q)
    plt.autoscale(False)
    plt.axis('equal')
    plt.plot(x, y, 'o-')
    plt.plot(x2, y2, 'o-')


    plt.show()


