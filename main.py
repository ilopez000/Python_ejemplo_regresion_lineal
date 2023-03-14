import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from point import Point
from point_list import PointList

# Generamos 100 puntos con coordenadas (x, y) tal que y = 2*x + 3 con un poco de ruido
points = [Point(x, 2*x + 3 + np.random.normal(0, 0.5)) for x in np.random.uniform(0, 10, size=100)]

# Creamos una lista de puntos y agregamos los 100 puntos generados
point_list = PointList()
for p in points:
    point_list.add_point(p)

# Creamos una regresión lineal con los puntos
model = point_list.create_linear_regression()

# Mostramos los puntos y la línea de regresión en un gráfico
plt.scatter([p.x for p in point_list.points], [p.y for p in point_list.points])
plt.plot([0, 10], [model.intercept_, model.coef_[0][0] * 10 + model.intercept_], color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
print(model.intercept_)
print(model.coef_[0][0])