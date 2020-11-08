import numpy as np

from fem.plotting import plot_solution


class Function:

    def value(self, p):
        """
        :param p: the point to evaluate the function in.
        :return:
        """
        raise NotImplementedError()

    def gradient(self, p, value=None):
        """
        :param p: the point to evaluate the gradient of the function in.
        :param value:
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

    def plot(self, points, triangles, plot_func=None):
        values = self.value_list(points)
        dim = len(values[0])

        for d in range(dim):
            func_values = values[:, d]
            if plot_func is None:
                ax = plot_solution(points, func_values, triangles, latex=True)
            else:
                ax = plot_func(points, func_values, triangles)
            ax.set_title("Component " + str(d + 1))
