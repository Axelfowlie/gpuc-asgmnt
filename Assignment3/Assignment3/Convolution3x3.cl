/*
We assume a 3x3 (radius: 1) convolution kernel, which is not separable.
Each work-group will process a (TILE_X x TILE_Y) tile of the image.
For coalescing, TILE_X should be multiple of 16.

Instead of examining the image border for each kernel, we recommend to pad the image
to be the multiple of the given tile-size.
*/

// should be multiple of 32 on Fermi and 16 on pre-Fermi...
#define TILE_X 32

#define TILE_Y 16

// d_Dst is the convolution of d_Src with the kernel c_Kernel
// c_Kernel is assumed to be a float[11] array of the 3x3 convolution constants, one multiplier (for normalization) and an offset (in this order!)
// With & Height are the image dimensions (should be multiple of the tile size)
__kernel
    __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1))) void Convolution(__global float* d_Dst, __global const float* d_Src, __constant float* c_Kernel,
                                                                              uint Width,  // Use width to check for image bounds
                                                                              uint Height,
                                                                              uint Pitch  // Use pitch for offsetting between lines
                                                                              ) {
  // OpenCL allows to allocate the local memory from 'inside' the kernel (without using the clSetKernelArg() call)
  // in a similar way to standard C.
  // the size of the local memory necessary for the convolution is the tile size + the halo area
  __local float tile[TILE_Y + 2][TILE_X + 2];

  int2 GID = (int2)(get_global_id(0), get_global_id(1));
  int2 LID = (int2)(get_local_id(0), get_local_id(1));



  // Load main tile into local memory
  if (GID.x < Width && GID.y < Height) tile[LID.y + 1][LID.x + 1] = d_Src[GID.y * Pitch + GID.x];


  // Load the halo into local memory
  // Loads edges and corners individually.
  // For each edge (top/bottom row, left/right column) and corner a separate warp will load the data.
  // This is ensured with the "if (LID.y == ???)" statement, because tile width is 32.
  // Rows are accessing memory coalesced and every work item writes to a different memory bank.
  // Columns are special:
  // The tile in local mamory has with 32+2, since there are 32 banks data lements in the same column
  // reside on banks that have one spare bank in between (e.g. (0,0) on bank 0, (0,1) on bank 2, etc...)
  // When loading 16 consecutive elements on the same comlumn we can load up to 16 elements without bank confilcts.
  // NOTE: This only works if TILE_X == 2*TILE_Y
  //
  // NOTE: Loading the columns using the first/last work item in each row in the work group was slower, most likely
  // due to serialization of if-statments.


  // This is the pixel coor of the top left corner pixel of the MAIN tile (not the top left halo) IN GLOBAL COORDINATES OF THE IMAGE
  int2 g_topleft = (int2)(get_group_id(0) * TILE_X, get_group_id(1) * TILE_Y);
  // This is the pixel coor of the bottom right corner pixel of the MAIN tile (not the halo) IN LOCAL COORDINATES OF THE PROCESSED TILE
  // We use the pixel coor and not the size, since then we can always use e.g. tile[l_botright.y + 2][x] when accessing the bottom row of the halo.
  int2 l_botright = (int2)(TILE_X - 1, TILE_Y - 1);
  // The value is usually the size of the work group minus 1 for x and y, although this is not the case for tiles at the bottom/right edge of the image.
  // Then we have to take care about the real image width height...
  if (g_topleft.x + l_botright.x >= Width) l_botright.x = Width % TILE_X - 1;
  if (g_topleft.y + l_botright.y >= Height) l_botright.y = Height % TILE_Y - 1;


  // Top row of halo
  // This if is taken by the whole warp or none of the work items inside
  if (LID.y == 0) {
    // Init elememt with 0
    float elem = 0;
    // Load pixel from source, if we are not in the very first row of the image
    if (g_topleft.y != 0) elem = d_Src[(g_topleft.y - 1) * Pitch + GID.x];
    // Write either 0 (first row of image) or the source value (any other row) into local memory
    tile[0][LID.x + 1] = elem;
  }

  // Bottom row of halo
  // This if is taken by the whole warp or none of the work items inside
  if (LID.y == 1) {
    float elem = 0;
    if (g_topleft.y + l_botright.y != Height) elem = d_Src[(g_topleft.y + l_botright.y + 1) * Pitch + GID.x];
    tile[l_botright.y + 2][LID.x + 1] = elem;
  }

  // Left column of halo
  if (LID.y == 2 && LID.x <= l_botright.y) {
    float elem = 0;
    if (g_topleft.x != 0) elem = d_Src[(g_topleft.y + LID.x) * Pitch + g_topleft.x - 1];
    tile[LID.x + 1][0] = elem;
  }

  // Right column of halo
  if (LID.y == 3 && LID.x <= l_botright.y) {
    float elem = 0;
    if (g_topleft.x + l_botright.x != Width) elem = d_Src[(g_topleft.y + LID.x) * Pitch + g_topleft.x + l_botright.x + 1];
    tile[LID.x + 1][l_botright.x + 2] = elem;
  }


  // Top left corner
  if (LID.y == 4 && LID.x == 0) {
    float elem = 0;
    if (g_topleft.x != 0 && g_topleft.y != 0) elem = d_Src[(g_topleft.y - 1) * Pitch + g_topleft.x - 1];
    tile[0][0] = elem;
  }
  // Bottom left corner
  if (LID.y == 5 && LID.x == 0) {
    float elem = 0;
    if (g_topleft.x != 0 && g_topleft.y + l_botright.y != Height) elem = d_Src[(g_topleft.y + l_botright.y + 1) * Pitch + g_topleft.x - 1];
    tile[l_botright.y + 2][0] = elem;
  }

  // Top right corner
  if (LID.y == 5 && LID.x == 0) {
    float elem = 0;
    if (g_topleft.x + l_botright.x != Width && g_topleft.y != 0) elem = d_Src[(g_topleft.y - 1) * Pitch + g_topleft.x + l_botright.x + 1];
    tile[0][l_botright.x + 2] = elem;
  }
  // Bottom left corner
  if (LID.y == 5 && LID.x == 0) {
    float elem = 0;
    if (g_topleft.x + l_botright.x != Width && g_topleft.y + l_botright.y != Height) elem = d_Src[(g_topleft.y + l_botright.y + 1) * Pitch + g_topleft.x + l_botright.x + 1];
    tile[l_botright.y + 2][l_botright.x + 2] = elem;
  }


  //// Left column of halo (including top left and bottom left corners)
  //// The corners are included, so that we can ommit some ifs that check for the same bounds when doing it separately later on
  //// We do not use vertically aligned work items to read the column, since they have to write to local memory with same banks (in column)
  //// Each warp has to serialize the left and right column and corners respectively
  //// So in the end there are up to 6 serialized memory accesses for a warp in worst case... (main block + top/bot row + left col + left corner + right col +
  //// right corner)
  // if (LID.x == 0) {
  //  // If this is the left most column in the picture, initialize the halo to zero
  //  if (GID.x == 0) {
  //    // The edge
  //    tile[LID.y + 1][0] = 0;
  //    // Top right corner
  //    if (LID.y == 0) tile[0][0] = 0;
  //    // Bottom right corner
  //    if (LID.y == TILE_Y - 1 || GID.y == Height) tile[LID.y + 2][0] = 0;
  //  } else {
  //    // We are somewhere inside the image, so load the pixels from the left column
  //    tile[LID.y + 1][0] = d_Src[GID.y * Pitch + GID.x - 1];
  //    // Top left corner needs special case, since it may be on the top most row
  //    float elem = 0;
  //    if (LID.y == 0) {
  //      if (GID.y != 0) elem = d_Src[(GID.y - 1) * Pitch + GID.x - 1];
  //      tile[0][0] = elem;
  //    }
  //    // Bottom left corner is spacial case on the bottom most row...
  //    elem = 0;
  //    if (LID.y == TILE_Y - 1 || GID.y == Height) {
  //      if (GID.y != Height) elem = d_Src[(GID.y + 1) * Pitch + GID.x - 1];
  //      tile[LID.y + 2][0] = elem;
  //    }
  //  }
  //}
  // if (LID.x == TILE_X - 1 || GID.x == Width) {
  //  // If this is the right most column in the picture, initialize the halo to zero
  //  if (GID.x == Width) {
  //    // The edge
  //    tile[LID.y + 1][LID.x + 2] = 0;
  //    // Top right corner
  //    if (LID.y == 0) tile[0][LID.x + 2] = 0;
  //    // Bottom right corner
  //    if (LID.y == TILE_Y - 1 || GID.y == Height) tile[LID.y + 2][LID.x + 2] = 0;
  //  } else {
  //    // We are somewhere inside the image, so load the pixels from the right column
  //    tile[LID.y + 1][LID.x + 2] = d_Src[GID.y * Pitch + GID.x + 1];
  //    // Top left corner needs special case, since it may be on the top most row
  //    float elem = 0;
  //    if (LID.y == 0) {
  //      if (GID.y != 0) elem = d_Src[(GID.y - 1) * Pitch + GID.x + 1];
  //      tile[0][LID.x + 2] = elem;
  //    }
  //    // Bottom left corner is spacial case on the bottom most row...
  //    elem = 0;
  //    if (LID.y == TILE_Y - 1 || GID.y == Height) {
  //      if (GID.y != Height) elem = d_Src[(GID.y + 1) * Pitch + GID.x + 1];
  //      tile[LID.y + 2][LID.x + 2] = elem;
  //    }
  //  }
  //}

  barrier(CLK_LOCAL_MEM_FENCE);


  float out = tile[LID.y][LID.x] * c_Kernel[0] + 
              tile[LID.y][LID.x + 1] * c_Kernel[1] + 
              tile[LID.y][LID.x + 2] * c_Kernel[2] +
              tile[LID.y + 1][LID.x] * c_Kernel[3] + 
              tile[LID.y + 1][LID.x + 1] * c_Kernel[4] + 
              tile[LID.y + 1][LID.x + 2] * c_Kernel[5] +
              tile[LID.y + 2][LID.x] * c_Kernel[6] + 
              tile[LID.y + 2][LID.x + 1] * c_Kernel[7] + 
              tile[LID.y + 2][LID.x + 2] * c_Kernel[8];
  out *= c_Kernel[9];
  out += c_Kernel[10];



  if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = out;
  // top row
  //if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = tile[LID.y][LID.x+1];
  // bottom row
  //if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = tile[LID.y+2][LID.x+1];
  // left col
  //if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = tile[LID.y+1][LID.x];
  // right col
  //if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = tile[LID.y+1][LID.x + 2];
  // top left
  //if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = tile[LID.y][LID.x];
  // bottom left
  //if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = tile[LID.y+2][LID.x];
  // top right
  //if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = tile[LID.y][LID.x+2];
  // top right
  //if (GID.x < Width && GID.y < Height) d_Dst[GID.y * Pitch + GID.x] = tile[LID.y+2][LID.x+2];
}
