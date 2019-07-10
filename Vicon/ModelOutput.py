from collections import namedtuple


class ModelOutput(object):

    def __init__(self, data, joint_name):
        print data.keys()
        self.phyics = ["Angles", "Force", "Moment", "Power"]
        self.joint_names = joint_name
        self._left_joints = {}
        self._right_joints = {}
        self._Point = namedtuple('Point', 'x y z')
        self._Newton = namedtuple('Newton', 'angle force moment power')

        for side, joint in zip(("R", "L"), (self._left_joints, self._right_joints)):
            for output in joint_name:
                angle = self._Point(data[side + output + "Angles"]["X"]["data"],
                                    data[side + output + "Angles"]["Y"]["data"],
                                    data[side + output + "Angles"]["Z"]["data"])
                force = self._Point(data[side + output + "Force"]["X"]["data"],
                                    data[side + output + "Force"]["Y"]["data"],
                                    data[side + output + "Force"]["Z"]["data"])
                moment = self._Point(data[side + output + "Moment"]["X"]["data"],
                                     data[side + output + "Moment"]["Y"]["data"],
                                     data[side + output + "Moment"]["Z"]["data"])
                power = self._Point(data[side + output + "Power"]["X"]["data"],
                                    data[side + output + "Power"]["Y"]["data"],
                                    data[side + output + "Power"]["Z"]["data"])
                joint[output] = self._Newton(angle, force, moment, power)

        # LFE
        # LFO
        # FootProgressAngles
        # LPelvisAngles
        # LTI
        # LTO
        # PEL
        # RFE
        # RFO
        # RFootProgressAngles
        # RPelvisAngles
        # RTI
        # RTO

    def get_right_joint(self, joint_name):
        """

        :param joint_name:
        :return:
        :rtype: self._Newton
        """
        return self._right_joints[joint_name]

    def get_left_joint(self, joint_name):

        return self._right_joints[joint_name]
