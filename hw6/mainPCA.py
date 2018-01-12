#!/usr/env/bin python
import numpy as np
from skimage import io
import sys

target = sys.argv[2]
dataPath = sys.argv[1]

#--------------------reading data--------------------
print("[Status] Reading Data\n")
X = []
pic = None
for o in range(415):
  picNum = '/%d.jpg'%o
  pic = io.imread(dataPath + picNum)
  X.append(pic.flatten())

pics_matrix = np.array(X)
mu = np.mean(pics_matrix, axis=0)

#------------------------eigen faces-----------------------
print("[Status] Eigen Faces\n")

X = pics_matrix - mu
faceRepresentNum = 4
print("[Status] SVD computing......\n")
print("\t[Info] Face Size: %d\n"%faceRepresentNum)
eigen_faces, sigma, v = np.linalg.svd(X.T, full_matrices=False)
eigen_faces = eigen_faces.T[:faceRepresentNum]


#------------------------Reconstruct-----------------------
print("[Status] Reconstruct\n")

target = io.imread(target).reshape(600*600*3)
target = target-mu

weights = np.dot(target, eigen_faces.T)
pics = np.dot(weights, eigen_faces)
pics += mu.flatten()

M = pics
M -= np.min(M)
M /= np.max(M)
M = (M*255).astype(np.uint8)
io.imsave('./reconstruction.jpg', M.reshape(600, 600, 3))

