
__kernel void set_array_to_constant(__global int* array, int num_elements, int val) {
  // There is no need to touch this kernel
  if (get_global_id(0) < num_elements) array[get_global_id(0)] = val;
}

__kernel void compute_histogram(__global int* histogram,    // accumulate histogram here
                                __global const float* img,  // input image
                                int width,                  // image width
                                int height,                 // image height
                                int pitch,                  // image pitch
                                int num_hist_bins           // number of histogram bins
                                ) {
  int2 GID = (int2)(get_global_id(0), get_global_id(1));
  if (GID.x < width && GID.y < height) {
    float pix = img[GID.y * pitch + GID.x] * (float)num_hist_bins;
    int bin = min(num_hist_bins - 1, max(0, (int)pix));
    atomic_inc(&histogram[bin]);
  }
}

__kernel void compute_histogram_local_memory(__global int* histogram,    // accumulate histogram here
                                             __global const float* img,  // input image
                                             int width,                  // image width
                                             int height,                 // image height
                                             int pitch,                  // image pitch
                                             int num_hist_bins,          // number of histogram bins
                                             __local int* local_hist) {
  int2 GID = (int2)(get_global_id(0), get_global_id(1));
  int flat_id = get_local_id(1) * get_local_size(0) + get_local_id(0);

  if (flat_id < num_hist_bins) local_hist[flat_id] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (GID.x < width && GID.y < height) {
    float pix = img[GID.y * pitch + GID.x] * (float)num_hist_bins;
    int bin = min(num_hist_bins - 1, max(0, (int)pix));
    atomic_inc(&local_hist[bin]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (flat_id < num_hist_bins) atomic_add(&histogram[flat_id], local_hist[flat_id]);
}
