from scipy.ndimage import correlate, convolve
import numpy as np

a = [
     [[1, 2, 3, 4],
      [5, 6, 7, 8],
      [10, 11, 12, 13]],
     [[1, 2, 3, 4],
      [8, 7, 6, 5],
      [10, 11, 12, 13]],
     [[4, 5, 3, 1],
      [4, 7, 7, 3],
      [16, 1, 13, 18]]
     ]

a2 = [[1, 2, 3, 4],
      [5, 6, 7, 8],
      [10, 11, 12, 13]]

k = np.array([
     [[0, 2, 0],
      [2, 2, 2],
      [0, 2, 0]],
     [[1, 2, 1],
      [2, 2, 2],
      [1, 2, 1]],
     [[1, 5, 1],
      [5, 5, 5],
      [1, 5, 1]]
     ])

k2 = np.array([[0, 2],
      [2, 2]])

g = np.array([[1, 2, 4], [5, 2, 0], [3, 1, 6]])

j = np.array([1, 2])

b = correlate(a2, k2, mode='constant', cval=0.0)
c = g + 5

row, col = np.indices((g.shape[0], g.shape[1]))
row_bool = np.where(row % 2 == 0, True, False)
col_bool = np.where(col % 1 == 0, True, False)
output_y = int(np.ceil(k.shape[1] / 2))
output_x = int(np.ceil(k.shape[2] / 1))
stride = row_bool & col_bool
k = np.reshape(k[:, stride], (k.shape[0], output_y, output_x))

s = np.indices((5, 1))
print(s)


