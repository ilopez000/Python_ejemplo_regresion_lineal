import numpy as np
from sklearn.linear_model import LinearRegression
from point import Point

class PointList:
    def __init__(self, points=[]):
        self.points = points

    def add_point(self, point):
        self.points.append(point)

    def create_linear_regression(self):
        X = np.array([[p.x for p in self.points]]).T
        Y = np.array([[p.y for p in self.points]]).T
        model = LinearRegression().fit(X, Y)
        return model