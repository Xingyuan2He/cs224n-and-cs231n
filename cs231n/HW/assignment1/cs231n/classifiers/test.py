import numpy as np
a = np.array([[1, 2], [3, 4]])
# print(a)
b = np.array([[1], [1]])
# a[:, 0] += b
print(a.sum([1]))
