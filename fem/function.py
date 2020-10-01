import numpy as np


class Function:

    def value(self, p):
        """
        :param p: the point to evaluate the function in.
        :return:
        """
        raise NotImplementedError()

    def value_list(self, points):
        """
        :param points: a list of points to evaluate the function in
        :return:
        """
        values = []
        for p in points:
            values.append(self.value(p))
        return np.array(values)
