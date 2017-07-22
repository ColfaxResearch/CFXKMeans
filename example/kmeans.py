import sys
import math
import numpy as np
from sklearn.datasets import fetch_mldata
import time
import cfxkmeans

if len(sys.argv) < 3:
  print("usage: "+sys.argv[0]+" {k} {precision (f|d)} ")
  exit()
k=int(sys.argv[1])
mnist = fetch_mldata('MNIST original')
data = None
if sys.argv[2] == "d":
  data = mnist.data.astype(np.float64)
else:
  data = mnist.data.astype(np.float32)
stride = int(math.ceil(data.shape[0]/k)+1)
init = data[::stride,:]

cfxkm = cfxkmeans.KMeans(k, init).fit(data)

print(cfxkm.labels_)

