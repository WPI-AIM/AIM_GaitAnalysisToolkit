import yaml


class Session(object):

    def __init__(self, subject_file):
        print "salkdjflsakjflsa"
        with open(subject_file, 'r') as stream:
            try:
                self._subject = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print "fialed"
                print(exc)

        print self._subject
        self._age = float(self._subject["Age"])

        self._mass = float(self._subject["Mass"])
        self._height = float(self._subject["Height"])
        self._subject_number = float(self._subject["subject"])
        self._leg_length = float(self._subject["LegLength"])
        self._gender = self._subject["Gender"]
        self._trial = self._subject['trials']

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


if __name__ == "__main__":
    session = Session("/home/nathaniel/git/exoserver/Main/subject_1234.yaml")
