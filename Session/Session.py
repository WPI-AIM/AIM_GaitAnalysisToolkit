import yaml

import Trial


class Session(object):

    def __init__(self, config_file, subject_file):

        self._config_file = config_file
        with open(subject_file, 'r') as stream:
            try:
                self._subject = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self._age = float(self._subject["Age"])
        self._mass = float(self._subject["Mass"])
        self._height = float(self._subject["Height"])
        self._subject_number = float(self._subject["subject"])
        self._leg_length = float(self._subject["LegLength"])
        self._gender = self._subject["Gender"]
        self._trials = self.seperate_trials(self._subject["trials"])
        self._trial_names = []

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
        """
        sperates the trial into the dictionaries
        :param trials_names:
        :return:
        """
        trials = {}
        for key, value in trials_names.iteritems():
            self._trial_names.append(key)
            trials[key] = Trial.Trial(self._config_file, value["output"], value["vicon"], value["dt"], value["notes"])

        return trials

    def black_list_trial_cycle(self, trial_number, black):
        """
        blacklist the trials
        :param trial_number:
        :param black:
        :return:
        """
        self._trials[trial_number].add_to_blacklist(black)

    def black_list_trial(self, trial_number):
        """
        blacklist the trials
        :param trial_number:
        :param black:
        :return:
        """
        del self._trials[trial_number]


    def collect_exo_accel(self):
        """
        collect all the accel from all the trials
        :return:
        """
        trials = {}
        for key, value in self.trials.iteritems():
            trials[key] = self.trials[key].get_accel()
        return trials

    def collect_exo_CoP(self):
        """
        collect all the accel from all the trials
        :return:
        """
        trials = {}
        for key, value in self.trials.iteritems():
            trials[key] = self.trials[key].get_CoP()
        return trials

    def collect_exo_pots(self):
        pass

    def collect_frame(self):
        pass

    def see_trial(self, number):
        pass
# if __name__ == "__main__":
#     session = Session("/home/nathaniel/git/exoserver/Main/subject_1234.yaml")
#     print session.mass
