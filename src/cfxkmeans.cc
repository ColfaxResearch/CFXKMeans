#include "../include/cfxkmeans.h"
template <class T>
CFXKMeans::KMeans<T>::KMeans(int k, int n_features, T *initial_centroids, long ldc, CFXKMeans::ALGORITHM alg) {
  alg_ = alg;
  assert(k>1);
  k_ = k;
  n_features_ = n_features;
  centroids_   =   (T *) _mm_malloc(sizeof(T)*k_*n_features_, 64);
#pragma omp parallel for 
  for(int j = 0; j < k_; j++) {
    for(int f = 0; f < n_features_; f++) 
      centroids_[j*n_features_+f]=initial_centroids[j*n_features_+ f];
  }
}

template <class T>
CFXKMeans::KMeans<T>::KMeans(int k, int n_features, T *initial_centroids, long ldc) {
  KMeans(k, n_features, initial_centroids, ldc, CFXKMeans::ALGORITHM::HAMERLY);
}

template <class T>
CFXKMeans::KMeans<T>::~KMeans() {
  _mm_free(centroids_);
  if(assignment_)
    _mm_free(assignment_);
}

template <class T>
void CFXKMeans::KMeans<T>::fit(long n_vectors, T *data) {
  if(assignment_)
    _mm_free(assignment_);
  assignment_   =   (int *) _mm_malloc(sizeof(int)*n_vectors, 64);
  CFXKMeans::fit(k_, n_features_, n_vectors, data, centroids_, assignment_, alg_);
}

// Instantiate floating point types.
template class CFXKMeans::KMeans<float>;
template class CFXKMeans::KMeans<double>;

void CFXKMeans::fit(int k, int n_features, long n_vectors, float *data, float *centroids, int *assignment, CFXKMeans::ALGORITHM alg) {
  switch(alg) {
    case CFXKMeans::ALGORITHM::HAMERLY: 
      CFXKMeansHamerly::fit(k, n_features, n_vectors, data, centroids, assignment);
      break;
    default: 
      std::cerr << "Unknown/Unsupported algorithm type. Using HAMERLY instead." << std::endl;
      CFXKMeansHamerly::fit(k, n_features, n_vectors, data, centroids, assignment);
      break;
  }
}

void CFXKMeans::fit(int k, int n_features, long n_vectors, double *data, double *centroids, int *assignment, CFXKMeans::ALGORITHM alg) {
  switch(alg) {
    case CFXKMeans::ALGORITHM::HAMERLY: 
      CFXKMeansHamerly::fit(k, n_features, n_vectors, data, centroids, assignment);
      break;
    default: 
      std::cerr << "Unknown/Unsupported algorithm type. Using HAMERLY instead." << std::endl;
      CFXKMeansHamerly::fit(k, n_features, n_vectors, data, centroids, assignment);
      break;
  }

}


