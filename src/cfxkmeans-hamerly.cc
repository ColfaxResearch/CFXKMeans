#include "../include/cfxkmeans-hamerly.h"

// This should be defined in Makefile, but just in case
#ifndef DTYPE
#define DTYPE float
#endif

DTYPE CFXKMeansHamerly::eucl_dist_sq(const int n_features, DTYPE *point1, DTYPE *point2) {
  DTYPE dist = 0.0; 
#pragma omp simd reduction(+: dist)
  for(int f=0; f<n_features; f++) {
     dist+=(point1[f]-point2[f])*
            (point1[f]-point2[f]);
  }
  return dist;
}

void CFXKMeansHamerly::initializeScratchData(const int k, const int n_features, const long n_vectors, int *centroid_counter, DTYPE *member_vector_sum, DTYPE *data) {
  for(int i = 0; i < k; i++) {
    centroid_counter[i]=0;
    for(int f = 0; f < n_features; f++) {
      member_vector_sum[i*n_features+f] = 0.0;
    }
  }
  centroid_counter[0]=n_vectors;
#pragma omp parallel 
  {
    DTYPE vec_sum_thread[n_features];
    for(int f = 0; f < n_features; f++) 
      vec_sum_thread[f] = 0.0;
#pragma omp for
    for(int i = 0; i < n_vectors; i++) {
      for(int f = 0; f < n_features; f++) {
        vec_sum_thread[f] += data[i*n_features+f];
      }
    }
    for(int f = 0; f < n_features; f++) 
#pragma omp atomic
      member_vector_sum[f] += vec_sum_thread[f];
  }
}

// Find the closest cluster centroid for each cluster centroid centroid
void CFXKMeansHamerly::findClosestCentroidToCentroid(const int k, const int n_features, DTYPE *centroids, DTYPE *min_dist_arr) {
    for(int i = 0; i < k; i++)
      min_dist_arr[i] = INFINITY;
    DTYPE centroid_dist_arr[k*k];

#pragma omp parallel for 
    for(int i = 0; i < k; i++) {
      for(int j = i+1; j < k; j++) {
        DTYPE dist = 0.0;
#pragma omp simd reduction(+: dist)
        for(int f=0; f<n_features; f++) {
           dist+=(centroids[i*n_features+f]-centroids[j*n_features+f])*
                  (centroids[i*n_features+f]-centroids[j*n_features+f]);
        }
        centroid_dist_arr[i*k+j]=dist*0.25f; 
      }
    }

    for(int i = 0; i < k; i++) {
      for(int j = i+1; j < k; j++) {
        if(min_dist_arr[i]>centroid_dist_arr[i*k+j]) {
          min_dist_arr[i]=centroid_dist_arr[i*k+j];
        }
        if(min_dist_arr[j]>centroid_dist_arr[i*k+j]) {
          min_dist_arr[j]=centroid_dist_arr[i*k+j];
        }
      }
      min_dist_arr[i]=std::sqrt(min_dist_arr[i]);
    }

}

void CFXKMeansHamerly::fit(const int k, const int n_features, const long n_vectors, DTYPE *data, DTYPE *centroids, int *assignment) {
  // Allocating scratch data
  DTYPE* member_vector_sum = (DTYPE *) _mm_malloc(sizeof(DTYPE)*k*n_features, 64);
  DTYPE* upper_bounds      = (DTYPE *) _mm_malloc(sizeof(DTYPE)*n_vectors, 64);
  DTYPE* lower_bounds      = (DTYPE *) _mm_malloc(sizeof(DTYPE)*n_vectors, 64);

  // Initializing scratch data, and clearing assignment
#pragma omp parallel for
  for(int i = 0; i < n_vectors; i++) {
    assignment  [i]=0;
    upper_bounds[i]=INFINITY;
    lower_bounds[i]=0.0;
  }
  int centroid_counter[k];
  initializeScratchData(k, n_features, n_vectors, centroid_counter, member_vector_sum, data);

  // Thread private scratch storages. On heap beacuse we could run out of stack for KNL
  const int n_threads=omp_get_max_threads();
  int   *delta_centroid_counter_glob   = (int *) _mm_malloc(sizeof(int)*n_threads*k, 64);
  DTYPE *delta_member_vector_sum_glob  = (DTYPE *) _mm_malloc(sizeof(DTYPE)*n_threads*k*n_features, 64);

  // Keeps track of whether a centroid moved (if none moved, that is convergence)
  volatile bool converged = false;

  // Main Loop
  while(!converged) {
    converged = true;

    DTYPE min_dist_arr[k];
    findClosestCentroidToCentroid(k, n_features, centroids, min_dist_arr);

    bool update_delta_centroid[n_threads*k]; 
#pragma omp parallel reduction(&: converged)
    {
      const int thread_id=omp_get_thread_num();
      // Initialize the scratch data
      for(int i = 0; i < k; i++) {
        update_delta_centroid[thread_id*k+i] = false;
        delta_centroid_counter_glob[thread_id*k+i]=0;
        for(int f = 0; f < n_features; f++) {
          delta_member_vector_sum_glob[thread_id*k*n_features+i*n_features+f] = 0.0;
        }
      }

      // Step 1: Compute Assignment for each feature vector
#pragma omp for schedule(guided,10) 
      for(long i = 0; i < n_vectors; i++) {
        // Checking bounds to see if it is possible to skip distance computations
        const DTYPE lower_lim = (lower_bounds[i] > min_dist_arr[assignment[i]]) ? lower_bounds[i] : min_dist_arr[assignment[i]];
        if(upper_bounds[i] <= lower_lim)
          continue;
        upper_bounds[i] = std::sqrt(eucl_dist_sq(n_features, &data[i*n_features], &centroids[assignment[i]*n_features]));
        if(upper_bounds[i] > lower_lim) {
          // Bounds check failed. Need distance computation.
          
          // First computing distances. Separating this part into a function unfortunately lowered performance
          DTYPE dist_arr[k]; 
          for(int j=0; j < k; j++)  
            dist_arr[j] = 0.0;
          const int kp_ = k - k%8;
          for(int jj=0; jj < kp_; jj+=8) {
            // SIMD array reduction breaks for GCC 6.3, so manually done
            DTYPE dist0 = 0.0, dist1 = 0.0, dist2 = 0.0, dist3 = 0.0, dist4 = 0.0, dist5 = 0.0, dist6 = 0.0, dist7 = 0.0;
#pragma omp simd reduction(+: dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7) 
            for(int f=0; f<n_features; f++) {
              // unrolled by 8
              dist0+=(data[i*n_features+f]-centroids[jj*n_features+f])*
                      (data[i*n_features+f]-centroids[jj*n_features+f]);
              dist1+=(data[i*n_features+f]-centroids[(jj+1)*n_features+f])*
                      (data[i*n_features+f]-centroids[(jj+1)*n_features+f]);
              dist2+=(data[i*n_features+f]-centroids[(jj+2)*n_features+f])*
                      (data[i*n_features+f]-centroids[(jj+2)*n_features+f]);
              dist3+=(data[i*n_features+f]-centroids[(jj+3)*n_features+f])*
                      (data[i*n_features+f]-centroids[(jj+3)*n_features+f]);
              dist4+=(data[i*n_features+f]-centroids[(jj+4)*n_features+f])*
                      (data[i*n_features+f]-centroids[(jj+4)*n_features+f]);
              dist5+=(data[i*n_features+f]-centroids[(jj+5)*n_features+f])*
                      (data[i*n_features+f]-centroids[(jj+5)*n_features+f]);
              dist6+=(data[i*n_features+f]-centroids[(jj+6)*n_features+f])*
                      (data[i*n_features+f]-centroids[(jj+6)*n_features+f]);
              dist7+=(data[i*n_features+f]-centroids[(jj+7)*n_features+f])*
                      (data[i*n_features+f]-centroids[(jj+7)*n_features+f]);
            }
              dist_arr[jj+0]+=dist0; dist_arr[jj+1]+=dist1; dist_arr[jj+2]+=dist2; dist_arr[jj+3]+=dist3;
              dist_arr[jj+4]+=dist4; dist_arr[jj+5]+=dist5; dist_arr[jj+6]+=dist6; dist_arr[jj+7]+=dist7;
          }
          // Remainder case
          for(int j=kp_; j < k; j++)  {
            DTYPE dist = 0.0; 
#pragma omp simd reduction(+:dist)
            for(int f=0; f<n_features; f++) 
                dist+=(data[i*n_features+f]-centroids[j*n_features+f])*
                              (data[i*n_features+f]-centroids[j*n_features+f]);
            dist_arr[j] = dist; 
          }

          // Finding the smallest two entries in dist_arr
          DTYPE min = INFINITY, second_min = INFINITY;
          int min_index;
          for(int j=0; j < k; j++) {
            if(min > dist_arr[j]) {
              second_min = min;
              min = dist_arr[j];
              min_index = j;
            } else if(second_min > dist_arr[j]) {
              second_min = dist_arr[j];
            }
          }

          // Check if assignmnt failed. If yes, update various data 
          if(min_index != assignment[i]) {
            // These two are hear to avoid unnecessary copies in reduction.
            update_delta_centroid[thread_id*k+min_index] = true;
            update_delta_centroid[thread_id*k+assignment[i]] = true;
            // update member count
            delta_centroid_counter_glob[thread_id*k+min_index]++;
            delta_centroid_counter_glob[thread_id*k+assignment[i]]--;
            // update member vector sum
            for(int f = 0; f < n_features; f++) {
              delta_member_vector_sum_glob[thread_id*k*n_features+min_index*n_features+f]+=data[i*n_features+f];
              delta_member_vector_sum_glob[thread_id*k*n_features+assignment[i]*n_features+f]-=data[i*n_features+f];
            }
            // Assignment changed, so continue with while iterations
            converged = false;
            
            assignment[i] = min_index;
            upper_bounds[i] = std::sqrt(min);
          }
          lower_bounds[i] = std::sqrt(second_min);
        }
      }

      // Reduction
#pragma omp for
      for(int i = 0; i < k; i++) {
        for(int t = 0; t < n_threads; t++) {
          if(update_delta_centroid[t*k+i]) {
            centroid_counter[i]+=delta_centroid_counter_glob[t*k+i];
            for(int f = 0; f < n_features; f++) 
              member_vector_sum[i*n_features+f]+=delta_member_vector_sum_glob[t*k*n_features+i*n_features+f];
          }
        }
      }
    }
    
    // Step 2: Compute new centroids
    
    // Contains how much each vector moved
    DTYPE delta_arr[k];
    for(int j = 0; j < k; j++) {
      const DTYPE inv_count=1.0/((DTYPE) centroid_counter[j]);
      DTYPE delta_sq=0.0;
      // Computing the average of each cluster for the new centroid
      for(int f = 0; f < n_features; f++) {
        const DTYPE normalized_vec_sum =member_vector_sum[j*n_features+f]*inv_count; 
        delta_sq += (centroids[j*n_features+f]-normalized_vec_sum)*
                     (centroids[j*n_features+f]-normalized_vec_sum); 
        centroids[j*n_features+f]=normalized_vec_sum;
      }
      delta_arr[j] = std::sqrt(delta_sq);
    }

    // Computing maximum delta to update the lower_bound with. 
    DTYPE max_delta=delta_arr[0];
    for(int j = 1; j < k; j++) {
      if(max_delta < delta_arr[j])
        max_delta=delta_arr[j];
    }
    // Update upper and lower bounds
#pragma omp parallel for
    for(int i = 0; i < n_vectors; i++) {
      upper_bounds[i]+=delta_arr[assignment[i]];
      lower_bounds[i]-=max_delta;
    }
  }

  // Freeing scratch data
  _mm_free(upper_bounds);
  _mm_free(lower_bounds);
  _mm_free(member_vector_sum);
  _mm_free(delta_centroid_counter_glob);
  _mm_free(delta_member_vector_sum_glob);
}




