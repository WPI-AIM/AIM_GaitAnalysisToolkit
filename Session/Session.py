import yaml

import Trial


class Session(object):

    def __init__(self, subject_file):

        with open(subject_file, 'r') as stream:
            try:
                self._subject = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        print self._subject
        self._age = float(self._subject["Age"])

        self._mass = float(self._subject["Mass"])
        self._height = float(self._subject["Height"])
        self._subject_number = float(self._subject["subject"])
        self._leg_length = float(self._subject["LegLength"])
        self._gender = self._subject["Gender"]
        self._trials = self.seperate_trials(self._subject["trials"])


    @property
    def gender(self):
        return self._gender

    @property
    def mass(self):
        return self._mass

    @property
    def height(self):
        return self._height

    @property
    def age(self):
        return self._age

    @property
    def subject_number(self):
        return self._subject_number

    @property
    def leg_length(self):
        return self._leg_length

    @property
    def trials(self):
        return self._trials

    @gender.setter
    def gender(self, value):
        self._gender = value

    @mass.setter
    def mass(self, value):
        self._mass = value

    @height.setter
    def height(self, value):
        self._height = value

    @age.setter
    def age(self, value):
        self._age = value

    @subject_number.setter
    def subject_number(self, value):
        self._subject_number = value

    @leg_length.setter
    def leg_length(self, value):
        self._leg_length = value

    @trials.setter
    def trials(self, value):
        self._trials = value

    def seperate_trials(self, trials_names):
        trials = {}
        for key, value in trials_names.iteritems():
            trials[key] = Trial.Trial(value["output"], value["vicon"], value["dt"], value["notes"])

        return trials

    def collect_exo_imu(self, time_scale=False):
        pass

    def collect_exo_fsr(self, time_scale=False):
        pass

    def collect_exo_pots(self, time_scale=False):
        pass

    def collect_frame(self, frame, time_scale=False):
        pass

    def compare_stepping(self, frame):
        pass

    def plot_CoP(self, frame):

        trial = self.trials[0].exoskeleton

        left = [trial.get_left_leg().fsr.fsr1, trial.get_left_leg().fsr.fsr3, trial.get_left_leg().fsr.fsr3]
        right = [trial.get_right_leg().fsr.fsr1, trial.get_right_leg().fsr.fsr3, trial.get_right_leg().fsr.fsr3]
        cop_left = self.calc_CoP(left)
        cop_right = self.calc_CoP(right)

    def compate_plate_fsr(self, frame):
        trial = self.trials[0].exoskeleton
        left = [trial.get_left_leg().fsr.fsr1, trial.get_left_leg().fsr.fsr3, trial.get_left_leg().fsr.fsr3]
        right = [trial.get_right_leg().fsr.fsr1, trial.get_right_leg().fsr.fsr3, trial.get_right_leg().fsr.fsr3]
        cop_left = self.calc_CoP(left)
        cop_right = self.calc_CoP(right)

    def calc_CoP(self, fsrs):
        """
        calculate the CoP of the foot based on the FSR location
        and force
        CoP_x = sum_i(F_i * x_i)/sum_i(F_i)
        CoP_y = sum_i(F_i * y_i)/sum_i(F_i)
        :return:
        """

        total_force = 0
        centerX = 0
        centerY = 0
        CoP = []

        for sensor in fsrs:
            for val in sensor:
                total_force += sensor
                centerX += sensor * sensor.orientation[0]
                centerY += sensor * sensor.orientation[1]
            CoP.append([centerX / total_force, centerY / total_force])









if __name__ == "__main__":
    session = Session("/home/nathaniel/git/exoserver/Main/subject_1234.yaml")
    print session.mass
