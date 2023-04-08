import numpy as np
 
# initializing points in
# numpy arrays
point1 = np.array((1, 2, 4))
point2 = np.array((1, 1, 1))
 
# calculating Euclidean distance
# using linalg.norm()
dist = np.linalg.norm(point1 - point2)
 
# printing Euclidean distance
print(dist)
