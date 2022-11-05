import numpy as np

a = np.arange(40).reshape((5,8))       
# shuffle along axis 1
b = np.random.shuffle(a)
# shuffle along axis 0
np.random.shuffle(a)