import numpy as np
import Model

class LowerLimb(Model.Model):

    def __init__(self, mass, height):

        super(LowerLimb, self).__init__(mass, height)
        self.make_body()

    def IK(self, toe, offset=[0, 0], ankle_contraint=False):

        l1 = -self.lengths["thigh"]
        l2 = -self.lengths["shank"]
        l3 = self.lengths["ankle"]
        l4 = self.lengths["foot_length"]

        x = toe[0] + offset[0] - l4
        y = toe[1] + offset[1] + l3

        num = x * x + y * y - l1 ** 2 - l2 ** 2
        dem = 2 * l1 * l2
        print num / dem
        q2 = np.arctan2(-np.sqrt(1 - (num / dem) * (num / dem)), (num / dem))
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        q3 = 0.5 * np.pi - q1 - q2
        return [q1, q2, q3]

    def FK(self, q, offset=np.array([0,0])):

        l1 = -self.lengths["thigh"]
        l2 = -self.lengths["shank"]
        l3 = -self.lengths["ankle"]
        l4 = self.lengths["foot_length"]

        P1 = offset + np.array([l1 * np.cos(q[0]), l1 * np.sin(q[0])])
        P2 = P1 + np.array([l2 * np.cos(q[0] + q[1]), l2 * np.sin(q[0] + q[1])])
        P3 = P2 + np.array([l3 * np.cos(q[0] + q[1] + q[2]), l3 * np.sin(q[0] + q[1] + q[2])])
        P4 = P3 + np.array([l4 * np.cos((q[0] + q[1] + q[2]) - 0.5 * np.pi), l4 * np.sin((q[0] + q[1] + q[2]) - 0.5 * np.pi)])
        joint = np.array([offset, P1, P2, P3, P4])
        return joint


