
//Each thread load exactly one halo pixel
//Thus, we assume that the halo size is not larger than the 
//dimension of the work-group in the direction of the kernel

//to efficiently reduce the memory transfer overhead of the global memory
// (each pixel is lodaded multiple times at high overlaps)
// one work-item will compute RESULT_STEPS pixels

//for unrolling loops, these values have to be known at compile time

/* These macros will be defined dynamically during building the program

#define KERNEL_RADIUS 2

//horizontal kernel
#define H_GROUPSIZE_X		32
#define H_GROUPSIZE_Y		4
#define H_RESULT_STEPS		2

//vertical kernel
#define V_GROUPSIZE_X		32
#define V_GROUPSIZE_Y		16
#define V_RESULT_STEPS		3

*/

#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Horizontal convolution filter

/*
c_Kernel stores 2 * KERNEL_RADIUS + 1 weights, use these during the convolution
*/

// require matching work-group size
__kernel __attribute__((reqd_work_group_size(H_GROUPSIZE_X, H_GROUPSIZE_Y, 1))) void ConvHorizontal(__global float* d_Dst, __global const float* d_Src,
                                                                                                    __constant float* c_Kernel, int Width, int Pitch) {
  // The size of the local memory: one value for each work-item.
  // We even load unused pixels to the halo area, to keep the code and local memory access simple.
  // Since these loads are coalesced, they introduce no overhead, except for slightly redundant local memory allocation.
  // Each work-item loads H_RESULT_STEPS values + 2 halo values
  __local float tile[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];

  const int2 LID = (int2)(get_local_id(0), get_local_id(1));
  const int group = get_group_id(0);
  // Pixel coordinate of the work items 1st (left most) pixel in the region of processed pixels
  const int2 base = (int2)(H_RESULT_STEPS * H_GROUPSIZE_X * group + LID.x, get_global_id(1));


  // Initialize the halo's left border
  // Here we are loading the redundant memory, since branching would split the warp into a fraction that does, and one that does not load data.
  // Since the warp will load in a coalesced manner, it doesn't yield any benefit but makes the code harder to read...
  float elem = 0;
  if (group != 0) elem = d_Src[base.y * Pitch + base.x - H_GROUPSIZE_X];
  tile[LID.y][LID.x] = elem;

// Load region of processed pixels and halo's right border
#pragma unroll
  for (int i = 0; i <= H_RESULT_STEPS; ++i) {
    int off = i * H_GROUPSIZE_X;
    float elem = 0;
    if (base.x + off < Width) elem = d_Src[base.y * Pitch + base.x + off];
    tile[LID.y][LID.x + off + H_GROUPSIZE_X] = elem;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

// Process each pixel and write back
#pragma unroll
  for (int i = 0; i < H_RESULT_STEPS; ++i) {
    int off = i * H_GROUPSIZE_X;

    // Convolve
    float out = 0;
#pragma unroll
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k) out += c_Kernel[KERNEL_RADIUS - k] * tile[LID.y][LID.x + off + H_GROUPSIZE_X + k];

    if (base.x + off < Width) d_Dst[base.y * Pitch + base.x + off] = out;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertical convolution filter

// require matching work-group size
__kernel __attribute__((reqd_work_group_size(V_GROUPSIZE_X, V_GROUPSIZE_Y, 1))) void ConvVertical(__global float* d_Dst, __global const float* d_Src,
                                                                                                  __constant float* c_Kernel, int Height, int Pitch) {
  __local float tile[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];

  const int2 LID = (int2)(get_local_id(0), get_local_id(1));
  const int group = get_group_id(1);
  // Pixel coordinate of the work items 1st (top most) pixel in the region of processed pixels
  const int2 base = (int2)(get_global_id(0), V_RESULT_STEPS * V_GROUPSIZE_Y * group + LID.y);


  // Initialize the halo's upper border
  // Here not loading the redundant memory has a beneficial effect, that some warps dont need to do anything and imediately continue to the barrier
  float elem = 0;
  if (group != 0 && LID.y >= V_GROUPSIZE_Y - KERNEL_RADIUS) elem = d_Src[(base.y - V_GROUPSIZE_Y) * Pitch + base.x];
  tile[LID.y][LID.x] = elem;

// Load region of processed pixels
#pragma unroll
  for (int i = 0; i < V_RESULT_STEPS; ++i) {
    int off = i * V_GROUPSIZE_Y;
    elem = 0;
    // Load, if the pixel lies inside the image bounds
    if (base.y + off < Height) elem = d_Src[(base.y + off) * Pitch + base.x];
    tile[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem;
  }

  // Unpacking the last iteration that loads the halo's bottom border, so that we can ommit loading the redundant memory on this side, too.
  int off = V_RESULT_STEPS * V_GROUPSIZE_Y;
  elem = 0;
  // Load, if the pixel lies inside the image bounds
  if (base.y + off < Height && LID.y <= KERNEL_RADIUS) elem = d_Src[(base.y + off) * Pitch + base.x];
  tile[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem;


  barrier(CLK_LOCAL_MEM_FENCE);

// Process pixes and write back
#pragma unroll
  for (int i = 0; i < V_RESULT_STEPS; ++i) {
    int off = i * V_GROUPSIZE_Y;

    // Convolve
    float out = 0;
#pragma unroll
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k) out += c_Kernel[KERNEL_RADIUS - k] * tile[LID.y + off + V_GROUPSIZE_Y + k][LID.x];

    if (base.y + off < Height) d_Dst[(base.y + off) * Pitch + base.x] = out;
  }
}
