import numpy as np
import Model

class LowerLimb(Model.Model):

    def __init__(self, mass, height):

        super(LowerLimb, self).__init__(mass, height)
        self.make_body()

    def IK(self, toe, hip_location=[0,0], ankle_contraint=False):

        x = toe[0] + hip_location[0]
        y = toe[1] + hip_location[1]
        l1 = self.lengths["thigh"]
        l2 = self.lengths["shank"]
        l3 = self.lengths["ankle"]
        l4 = self.lengths["foot_length"]

        num = x*x + y*y - l1**2 - l2**2
        dem = 2*l1*l2

        q2 = np.arccos(num / dem)

        q1 = np.arctan2(y, x) - np.arctan2( l2*np.sin(q2), l1 + l2*np.cos(q2))
        q3 = np.pi - q1 - q2


        pass