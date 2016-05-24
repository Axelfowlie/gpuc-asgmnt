
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#define DEPTH_THRESHOLD	0.025f
#define NORM_THRESHOLD	0.9f

// These functions define discontinuities
bool IsNormalDiscontinuity(float4 n1, float4 n2){
	return fabs(dot(n1, n2)) < NORM_THRESHOLD;
}

bool IsDepthDiscontinuity(float d1, float d2){
	return fabs(d1 - d2) > DEPTH_THRESHOLD;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////
// Horizontal convolution filter

// require matching work-group size
__kernel __attribute__((reqd_work_group_size(H_GROUPSIZE_X, H_GROUPSIZE_Y, 1))) void DiscontinuityHorizontal(__global int* d_Disc,
                                                                                                             __global const float4* d_NormDepth, int Width,
                                                                                                             int Height, int Pitch) {
  // The size of the local memory: one value for each work-item.
  // We even load unused pixels to the halo area, to keep the code and local memory access simple.
  // Since these loads are coalesced, they introduce no overhead, except for slightly redundant local memory allocation.
  // Each work-item loads H_RESULT_STEPS values + 2 halo values
  // We split the float4 (normal + depth) into an array of float3 and float to avoid bank conflicts.

  __local float tileNormX[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];
  __local float tileNormY[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];
  __local float tileNormZ[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];
  __local float tileDepth[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];

  const int2 LID = (int2)(get_local_id(0), get_local_id(1));
  const int group = get_group_id(0);
  // Pixel coordinate of the work items 1st (left most) pixel in the region of processed pixels
  const int2 base = (int2)(H_RESULT_STEPS * H_GROUPSIZE_X * group + LID.x, get_global_id(1));


  // Initialize the halo's left border
  // Here we are loading the redundant memory, since branching would split the warp into a fraction that does, and one that does not load data.
  // Since the warp will load in a coalesced manner, it doesn't yield any benefit but makes the code harder to read...
  float4 elem = (float4)(0, 0, 0, 0);
  if (group != 0) elem = d_NormDepth[base.y * Pitch + base.x - H_GROUPSIZE_X];
  tileNormX[LID.y][LID.x] = elem.x;
  tileNormY[LID.y][LID.x] = elem.y;
  tileNormZ[LID.y][LID.x] = elem.z;
  tileDepth[LID.y][LID.x] = elem.w;

// Load region of processed pixels and halo's right border
#pragma unroll
  for (int i = 0; i <= H_RESULT_STEPS; ++i) {
    int off = i * H_GROUPSIZE_X;
    float4 elem = (float4)(0, 0, 0, 0);
    if (base.x + off < Width) elem = d_NormDepth[base.y * Pitch + base.x + off];
    tileNormX[LID.y][LID.x + off + H_GROUPSIZE_X] = elem.x;
    tileNormY[LID.y][LID.x + off + H_GROUPSIZE_X] = elem.y;
    tileNormZ[LID.y][LID.x + off + H_GROUPSIZE_X] = elem.z;
    tileDepth[LID.y][LID.x + off + H_GROUPSIZE_X] = elem.w;
  }

  barrier(CLK_LOCAL_MEM_FENCE);


// Process pixel and write back
#pragma unroll
  for (int i = 0; i < H_RESULT_STEPS; ++i) {
    // offset in global memory and the tile
    int off = i * H_GROUPSIZE_X;
    int off_tile = off + H_GROUPSIZE_X;

    int flag = 0;

    // Depth and normal of the center pixel in the current result step
    float myD = tileDepth[LID.y][LID.x + off_tile];
    float4 myN = (float4)(tileNormX[LID.y][LID.x + off_tile], tileNormY[LID.y][LID.x + off_tile], tileNormZ[LID.y][LID.x + off_tile], 0);

    // Left neighbour
    float otherD = tileDepth[LID.y][LID.x + off_tile - 1];
    float4 otherN = (float4)(tileNormX[LID.y][LID.x + off_tile - 1], tileNormY[LID.y][LID.x + off_tile - 1], tileNormZ[LID.y][LID.x + off_tile - 1], 0);
    if (IsDepthDiscontinuity(myD, otherD) || IsNormalDiscontinuity(myN, otherN)) flag |= 1;

    // Right neighbour
    otherD = tileDepth[LID.y][LID.x + off_tile + 1];
    otherN = (float4)(tileNormX[LID.y][LID.x + off_tile + 1], tileNormY[LID.y][LID.x + off_tile + 1], tileNormZ[LID.y][LID.x + off_tile + 1], 0);
    if (IsDepthDiscontinuity(myD, otherD) || IsNormalDiscontinuity(myN, otherN)) flag |= 2;

    // Write back result
    if (base.x + off < Width) d_Disc[base.y * Pitch + base.x + off] = flag;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertical convolution filter

// require matching work-group size
__kernel
    __attribute__((reqd_work_group_size(V_GROUPSIZE_X, V_GROUPSIZE_Y, 1))) void DiscontinuityVertical(__global int* d_Disc, __global const float4* d_NormDepth,
                                                                                                      int Width, int Height, int Pitch) {
  __local float tileNormX[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];
  __local float tileNormY[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];
  __local float tileNormZ[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];
  __local float tileDepth[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];

  const int2 LID = (int2)(get_local_id(0), get_local_id(1));
  const int group = get_group_id(1);
  // Pixel coordinate of the work items 1st (top most) pixel in the region of processed pixels
  const int2 base = (int2)(get_global_id(0), V_RESULT_STEPS * V_GROUPSIZE_Y * group + LID.y);


  // Initialize the halo's left border
  // Here we are loading the redundant memory, since branching would split the warp into a fraction that does, and one that does not load data.
  // Since the warp will load in a coalesced manner, it doesn't yield any benefit but makes the code harder to read...
  float4 elem = (float4)(0, 0, 0, 0);
  if (group != 0 && LID.y >= V_GROUPSIZE_Y - KERNEL_RADIUS) elem = d_NormDepth[(base.y - V_GROUPSIZE_Y) * Pitch + base.x];
  tileNormX[LID.y][LID.x] = elem.x;
  tileNormY[LID.y][LID.x] = elem.y;
  tileNormZ[LID.y][LID.x] = elem.z;
  tileDepth[LID.y][LID.x] = elem.w;

// Load region of processed pixels and halo's right border
#pragma unroll
  for (int i = 0; i < V_RESULT_STEPS; ++i) {
    int off = i * V_GROUPSIZE_Y;
    float4 elem = (float4)(0, 0, 0, 0);
    if (base.y + off < Height) elem = d_NormDepth[(base.y + off) * Pitch + base.x];
    tileNormX[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem.x;
    tileNormY[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem.y;
    tileNormZ[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem.z;
    tileDepth[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem.w;
  }

  // Unpacking the last iteration that loads the halo's bottom border, so that we can ommit loading the redundant memory on this side, too.
  int off = V_RESULT_STEPS * V_GROUPSIZE_Y;
  elem = (float4)(0, 0, 0, 0);
  // Load, if the pixel lies inside the image bounds
  if (base.y + off < Height && LID.y <= KERNEL_RADIUS) elem = d_NormDepth[(base.y + off) * Pitch + base.x];
  tileNormX[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem.x;
  tileNormY[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem.y;
  tileNormZ[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem.z;
  tileDepth[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem.w;

  barrier(CLK_LOCAL_MEM_FENCE);

// Process pixel and write back
#pragma unroll
  for (int i = 0; i < V_RESULT_STEPS; ++i) {
    // Y-offset into global memory and the tile
    int off = i * V_GROUPSIZE_Y;
    int off_tile = off + V_GROUPSIZE_Y;

    int flag = 0;

    // Depth and normal of the center pixel in the current result step
    float myD = tileDepth[LID.y + off_tile][LID.x];
    float4 myN = (float4)(tileNormX[LID.y + off_tile][LID.x], tileNormY[LID.y + off_tile][LID.x], tileNormZ[LID.y + off_tile][LID.x], 0);

    // Left neighbour
    float otherD = tileDepth[LID.y + off_tile - 1][LID.x];
    float4 otherN = (float4)(tileNormX[LID.y + off_tile - 1][LID.x], tileNormY[LID.y + off_tile - 1][LID.x], tileNormZ[LID.y + off_tile - 1][LID.x], 0);
    if (IsDepthDiscontinuity(myD, otherD) || IsNormalDiscontinuity(myN, otherN)) flag |= 4;

    // right neighbour
    otherD = tileDepth[LID.y + off_tile + 1][LID.x];
    otherN = (float4)(tileNormX[LID.y + off_tile + 1][LID.x], tileNormY[LID.y + off_tile + 1][LID.x], tileNormZ[LID.y + off_tile + 1][LID.x], 0);
    if (IsDepthDiscontinuity(myD, otherD) || IsNormalDiscontinuity(myN, otherN)) flag |= 8;


    // Write back, is not and doesn't need to be atomic
    if (base.y + off < Height) d_Disc[(base.y + off) * Pitch + base.x] |= flag;
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////
// Horizontal convolution filter

// require matching work-group size
__kernel __attribute__((reqd_work_group_size(H_GROUPSIZE_X, H_GROUPSIZE_Y, 1))) void ConvHorizontal(__global float* d_Dst, __global const float* d_Src,
                                                                                                    __global const int* d_Disc, __constant float* c_Kernel,
                                                                                                    int Width, int Height, int Pitch) {
  // The size of the local memory: one value for each work-item.
  // We even load unused pixels to the halo area, to keep the code and local memory access simple.
  // Since these loads are coalesced, they introduce no overhead, except for slightly redundant local memory allocation.
  // Each work-item loads H_RESULT_STEPS values + 2 halo values
  __local float tile[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];
  __local int disc[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];

  const int2 LID = (int2)(get_local_id(0), get_local_id(1));
  const int group = get_group_id(0);
  // Pixel coordinate of the work items 1st (left most) pixel in the region of processed pixels
  const int2 base = (int2)(H_RESULT_STEPS * H_GROUPSIZE_X * group + LID.x, get_global_id(1));


  // Initialize the halo's left border
  // Here we are loading the redundant memory, since branching would split the warp into a fraction that does, and one that does not load data.
  // Since the warp will load in a coalesced manner, it doesn't yield any benefit but makes the code harder to read...
  float elem = 0;
  int dc = 0;
  if (group != 0) {
    elem = d_Src[base.y * Pitch + base.x - H_GROUPSIZE_X];
    dc = d_Disc[base.y * Pitch + base.x - H_GROUPSIZE_X];
  }
  tile[LID.y][LID.x] = elem;
  disc[LID.y][LID.x] = dc;

// Load region of processed pixels and halo's right border
#pragma unroll
  for (int i = 0; i <= H_RESULT_STEPS; ++i) {
    int off = i * H_GROUPSIZE_X;
    float elem = 0;
    int dc = 0;
    if (base.x + off < Width) {
      elem = d_Src[base.y * Pitch + base.x + off];
      dc = d_Disc[base.y * Pitch + base.x + off];
    }
    tile[LID.y][LID.x + off + H_GROUPSIZE_X] = elem;
    disc[LID.y][LID.x + off + H_GROUPSIZE_X] = dc;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

// Process each pixel and write back
#pragma unroll
  for (int i = 0; i < H_RESULT_STEPS; ++i) {
    int off = i * H_GROUPSIZE_X;

    // Center pixel
    float out = c_Kernel[KERNEL_RADIUS] * tile[LID.y][LID.x + off + H_GROUPSIZE_X];
    float weight = c_Kernel[KERNEL_RADIUS];

    // Walk towards left
    for (int k = 0; k < KERNEL_RADIUS;) {
      int flag = disc[LID.y][LID.x + off + H_GROUPSIZE_X - k];
      if ((flag & 0x0001) != 0 || base.x - k <= 0) break;

      ++k;

      float w = c_Kernel[KERNEL_RADIUS - k];
      out += w * tile[LID.y][LID.x + off + H_GROUPSIZE_X - k];
      weight += w;
    }
    // Walk towards right
    for (int k = 0; k < KERNEL_RADIUS; ) {
      int flag = disc[LID.y][LID.x + off + H_GROUPSIZE_X + k];
      if ((flag & 0x0002) != 0 || base.x + k > Width) break;

      ++k;

      float w = c_Kernel[KERNEL_RADIUS + k];
      out += w * tile[LID.y][LID.x + off + H_GROUPSIZE_X + k];
      weight += w;
    }

    // Normalize
    if (weight != 0) out /= weight;
    else out = 0;

    if (base.x + off < Width) d_Dst[base.y * Pitch + base.x + off] = out;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertical convolution filter



// require matching work-group size
__kernel __attribute__((reqd_work_group_size(V_GROUPSIZE_X, V_GROUPSIZE_Y, 1))) void ConvVertical(__global float* d_Dst, __global const float* d_Src,
                                                                                                  __global const int* d_Disc, __constant float* c_Kernel,
                                                                                                  int Width, int Height, int Pitch) {
  __local float tile[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];
  __local int disc[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];

  const int2 LID = (int2)(get_local_id(0), get_local_id(1));
  const int group = get_group_id(1);
  // Pixel coordinate of the work items 1st (top most) pixel in the region of processed pixels
  const int2 base = (int2)(get_global_id(0), V_RESULT_STEPS * V_GROUPSIZE_Y * group + LID.y);


  // Initialize the halo's upper border
  // Here not loading the redundant memory has a beneficial effect, that some warps dont need to do anything and imediately continue to the barrier
  float elem = 0;
  int dc = 0;
  if (group != 0 && LID.y >= V_GROUPSIZE_Y - KERNEL_RADIUS) {
    elem = d_Src[(base.y - V_GROUPSIZE_Y) * Pitch + base.x];
    dc = d_Disc[(base.y - V_GROUPSIZE_Y) * Pitch + base.x];
  }
  tile[LID.y][LID.x] = elem;
  disc[LID.y][LID.x] = dc;

// Load region of processed pixels
#pragma unroll
  for (int i = 0; i <= V_RESULT_STEPS; ++i) {
    int off = i * V_GROUPSIZE_Y;
    elem = 0;
    dc = 0;
    // Load, if the pixel lies inside the image bounds
    if (base.y + off < Height) {
      elem = d_Src[(base.y + off) * Pitch + base.x];
      dc = d_Disc[(base.y + off) * Pitch + base.x];
    }
    tile[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem;
    disc[LID.y + off + V_GROUPSIZE_Y][LID.x] = dc;
  }

  // Unpacking the last iteration that loads the halo's bottom border, so that we can ommit loading the redundant memory on this side, too.
  int off = V_RESULT_STEPS * V_GROUPSIZE_Y;
  elem = 0;
  dc = 0;
  // Load, if the pixel lies inside the image bounds
  if (base.y + off < Height && LID.y <= KERNEL_RADIUS) {
    elem = d_Src[(base.y + off) * Pitch + base.x];
    dc = d_Disc[(base.y + off) * Pitch + base.x];
  }
  tile[LID.y + off + V_GROUPSIZE_Y][LID.x] = elem;
  disc[LID.y + off + V_GROUPSIZE_Y][LID.x] = dc;


  barrier(CLK_LOCAL_MEM_FENCE);

// Process pixes and write back
#pragma unroll
  for (int i = 0; i < V_RESULT_STEPS; ++i) {
    int off = i * V_GROUPSIZE_Y;

    // Center pixel
    float out = c_Kernel[KERNEL_RADIUS] * tile[LID.y + off + V_GROUPSIZE_Y][LID.x];
    float weight = c_Kernel[KERNEL_RADIUS];

    // Walk towards top
    for (int k = 0; k < KERNEL_RADIUS; ) {
      int flag = disc[LID.y + off + V_GROUPSIZE_Y - k][LID.x];
      if ((flag & 0x0004) != 0 || base.y - k <= 0) break;

      ++k;

      float w = c_Kernel[KERNEL_RADIUS - k];
      out += w * tile[LID.y + off + V_GROUPSIZE_Y - k][LID.x];
      weight += w;
    }
    // Walk towards bottom
    for (int k = 0; k < KERNEL_RADIUS; ) {
      int flag = disc[LID.y + off + V_GROUPSIZE_Y + k][LID.x];
      if ((flag & 0x0008) != 0 || base.y + k > Height) break;

      ++k;

      float w = c_Kernel[KERNEL_RADIUS + k];
      out += w * tile[LID.y + off + V_GROUPSIZE_Y + k][LID.x];
      weight += w;
    }

    // Normalize
    if (weight != 0) out /= weight;
    else out = 0;

    if (base.y + off < Height) d_Dst[(base.y + off) * Pitch + base.x] = out;
  }
}

