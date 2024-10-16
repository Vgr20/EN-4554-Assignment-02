import numpy as np

a = np.array([[1],[2],[3],[4],[5]])
b = np.array([4,5,6])

print("Shape of a: ", a.shape)
print("Shape of b: ", b.shape)

c = np.broadcast(a, b)
print("Shape of c: ", c.shape)
print("Value of c: ", c)
