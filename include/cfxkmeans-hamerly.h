#ifndef CFXKMEANSFUNC_H
#define CFXKMEANSFUNC_H

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif 

namespace CFXKMeansHamerly {
  // Main clustering function, fit()
  void fit(int k, int n_features, long n_vectors, float *data, float *centroids, int *assignment);
  void fit(int k, int n_features, long n_vectors, double *data, double *centroids, int *assignment);

  // Helper functions
  void initializeScratchData(const int k, const int n_features, const long n_vectors, int *centroid_counter, float *member_vector_sum, float *data);
  void initializeScratchData(const int k, const int n_features, const long n_vectors, int *centroid_counter, double *member_vector_sum, double *data);
  float  eucl_dist_sq(const int n_features, float *point1, float *point2);
  double eucl_dist_sq(const int n_features, double *point1, double *point2);
  void findClosestCentroidToCentroid(const int k, const int n_features, float *centroids, float *min_dist_arr);
  void findClosestCentroidToCentroid(const int k, const int n_features, double *centroids, double *min_dist_arr);
}
#endif
