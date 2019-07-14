from lib.Exoskeleton.Robot import Joint


class Leg(object):
    """
    A class to handle all the joints of a leg
    """

    def __init__(self, hip, knee, ankle):
        """

        :param hip: the hip joint
        :param knee: the knee joint
        :param ankle: the ankle joint
        :type hip: Joint.Joint
        :type knee: Joint.Joint
        :type ankle: Joint.Joint
        """
        self._hip = hip
        self._knee = knee
        self._ankle = ankle
        self._CoP = []

    def calc_CoP(self):
        """
        calculate the CoP of the foot based on the FSR location
        and force
        CoP_x = sum_i(F_i * x_i)/sum_i(F_i)
        CoP_y = sum_i(F_i * y_i)/sum_i(F_i)
        :return:
        """
        fsrs = self._ankle.FSRs

        for index in xrange(len(fsrs[0].get_values())):
            total_force = 0
            centerX = 0
            centerY = 0
            for sensor in fsrs:
                total_force += sensor.get_values()
                centerX += sensor.get_values() * sensor.orientation[0]
                centerY += sensor.get_values() * sensor.orientation[1]
            cop = [centerX / total_force, centerY / total_force]
            self._CoP.append(cop)
