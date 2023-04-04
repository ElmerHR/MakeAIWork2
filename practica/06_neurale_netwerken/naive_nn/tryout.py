import numpy as np
import math
label = np.array([1, 0])
y_hat = np.array([0.81, 0.19])
cel = -np.sum(label * np.log(y_hat))
print(cel)

a = np.array([1, 2])
b = np.array([5, 6])

c = np.concatenate((a,b), axis=0)
print(c)