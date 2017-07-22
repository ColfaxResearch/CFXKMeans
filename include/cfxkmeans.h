#ifndef CFXKMEANS_H
#define CFXKMEANS_H

#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <iostream>
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif 

// CFXKMeans packages
#include "../include/cfxkmeans-hamerly.h"

namespace CFXKMeans {
  // Supported algorithms
  enum class ALGORITHM { HAMERLY = 1 };

  // Class for C++ API
  template <class T>
  class KMeans {
    public:
      KMeans(int k, int n_features, T *initial_centroids, long ldc);
      KMeans(int k, int n_features, T *initial_centroids, long ldc, ALGORITHM alg);
      ~KMeans();
      void fit(long n_vectors, T *data);
      T   * getCentroids()   {return centroids_ ;}
      int * getAssignment()  {return assignment_ ;}
      int   getK()           {return k_;}
      int   getNumFeatures() {return n_features_;}
      long  getNumSamples()  {return n_vectors_;}
  
    private:
      // Input data 
      int k_;
      int n_features_;
      long n_vectors_;
      // centroids and assignment 
      T *centroids_;
      int *assignment_;

      // Algorithm to use
      CFXKMeans::ALGORITHM alg_;
 
      // util functions
      T eucl_dist_sq(T *point1, T *point2);
      void initializeScratchData(int *centroid_counter, T *sample_vector_sum, T *data);
  };
  
  // Functional versions of fit functions
  void fit(int k, int n_features, long n_vectors, float  *data, float  *centroids, int *assignment, CFXKMeans::ALGORITHM alg);
  void fit(int k, int n_features, long n_vectors, double *data, double *centroids, int *assignment, CFXKMeans::ALGORITHM alg);
}
#endif
