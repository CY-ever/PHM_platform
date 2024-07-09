import numpy as np
a = [2]
print(a)

n=3
b=a*np.ones(n, dtype=int)
b=np.reshape(b, (n, 1))
print(b)