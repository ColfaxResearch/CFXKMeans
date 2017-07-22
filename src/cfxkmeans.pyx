# distutils: language = c++
import math
import cython
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from 'cfxkmeans.h' namespace "CFXKMeans":
  cdef cppclass ALGORITHM:
    pass

cdef extern from 'cfxkmeans.h' namespace "CFXKMeans::ALGORITHM":
  cdef ALGORITHM HAMERLY

cdef extern from 'cfxkmeans.h' namespace "CFXKMeans":
  void fit(int k, int n_features, long n_vectors, float  *data, float  *centroids, int *assignment, ALGORITHM alg)
  void fit(int k, int n_features, long n_vectors, double *data, double *centroids, int *assignment, ALGORITHM alg)

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_dp(np.ndarray[double, ndim=2, mode='c'] data        not None,
           np.ndarray[double, ndim=2, mode='c'] centroids   not None, 
           np.ndarray[int, ndim=1, mode='c']     assignment not None):
  fit(centroids.shape[0], centroids.shape[1], data.shape[0], &data[0,0], &centroids[0,0], &assignment[0], HAMERLY)

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_sp(np.ndarray[float, ndim=2, mode='c'] data       not None,
           np.ndarray[float, ndim=2, mode='c'] centroids  not None, 
           np.ndarray[int, ndim=1, mode='c']   assignment not None):
  fit(centroids.shape[0], centroids.shape[1], data.shape[0], &data[0,0], &centroids[0,0], &assignment[0], HAMERLY)

class KMeans(object):
  centroids_ = None 
  labels_ = None 
  type_ = None
  def __init__(self, k, init):
    if type(init) is not np.ndarray:
      raise AttributeError("init must be an ndarray")
    else:
      if init.dtype == np.float32 or init.dtype == np.float64:
        self.type_    = init.dtype
        #making a copy because centroids is modified
        self.centroids_=np.copy(init)
      else:
        raise AttributeError("init must be float32 or float64")

  def __dealloc__(self):
    del self.centroids_

  def fit(self, data):
    if type(data) is not np.ndarray:
      raise AttributeError("data must be an ndarray")
    elif data.dtype != self.type_: 
      raise AttributeError("data must be the same type as init")
    else:
      self.labels_=np.empty((data.shape[0]), dtype=np.int32, order='C');
      if data.flags['C_CONTIGUOUS']:
        if self.type_ == np.float32:
          fit_sp(data, self.centroids_, self.labels_)
        else:
          fit_dp(data, self.centroids_, self.labels_)
      else:
        #making a copy because we want contiguous w/o modifying data
        tempdata = np.copy(data);
        if self.type_ == np.float32:
          fit_sp(tempdata, self.centroids_, self.labels_)
        else:
          fit_dp(tempdata, self.centroids_, self.labels_)
    return self
