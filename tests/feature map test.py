import numpy as np

ar1 = np.arange(1, 10).reshape(3, 3)
ar2 = np.random.choice([0, 1], (3, 3), p = [0.2, 0.8])
print(ar1)
print(ar2)
print(np.multiply(ar1, ar2))