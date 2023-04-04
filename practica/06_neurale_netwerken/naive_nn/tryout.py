import numpy as np
import math
# # input array
# x = np.array([[ 1., 2., 3.], [ 4., 5., 6.], [ 7., 8., 9.]])

# # random boolean mask for which values will be changed
# mask = np.random.randint(0,2,size=x.shape).astype(np.bool)
# print(mask)
# # random matrix the same shape of your data
# r = np.random.rand(*x.shape)*np.max(x)
# print(r)
# # use your mask to replace values in your input array
# x[mask] = r[mask]
# print(x)

x = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.]])

r = np.random.randint(0, 9)

print(r)

x[0][r] = 0.5

print(x)