
// Rotate the matrix CLOCKWISE

//naive implementation: move the elements of the matrix directly to their destinations
//this will cause unaligned memory accessed which - as we will see - should be avoided on the GPU

__kernel void MatrixRotNaive(__global const float* M, __global float* MR, uint SizeX, uint SizeY)
{
  int2 GID;
  GID.x = get_global_id(0);
  GID.y = get_global_id(1);

  if (GID.x < SizeX && GID.y < SizeY)
    MR[ GID.x * SizeY + (SizeY - GID.y - 1) ] = M[ GID.y * SizeX + GID.x ];
}

//this kernel does the same thing, however, the local memory is used to
//transform a small chunk of the matrix locally
//then write it back after synchronization in a coalesced access pattern

__kernel void MatrixRotOptimized(__global const float* M, __global float* MR, uint SizeX, uint SizeY,
							__local float* block)
{
  int2 GID = (int2)(get_global_id(0), get_global_id(1));

  int2 LID = (int2)(get_local_id(0), get_local_id(1));
  int2 LS = (int2)(get_local_size(0), get_local_size(1));


  // only load existing matrix elements
  if (GID.x < SizeX && GID.y < SizeY) 
    block[LID.y * LS.x + LID.x] = M[GID.y * SizeX + GID.x];

  barrier(CLK_LOCAL_MEM_FENCE);

  // Enumerate elements in the tile from top left to bottom right
  int LocalElemNb = LID.y * LS.x + LID.x;

  // Use this enumerartion to find the (x,y) coordinate of the work in the rotated tile
  int2 LIDrot;
  LIDrot.y = LocalElemNb / LS.y;
  LIDrot.x = LocalElemNb - LIDrot.y * LS.y;  // basically modulo but faster...

  // Offset of the x coordinate for the write position is the ramainder of unused elements in the (vertically) last tile.
  int xoff = get_global_size(1) - SizeY;
  // Tile ID in the rotated matrix
  int2 TileIDRot = (int2)(get_num_groups(1) - get_group_id(1) - 1, get_group_id(0));
  // With the tile ID and the (x,y) of the rotated tile we can compute the element position in the matrix globally.
  int2 ElemPos;
  ElemPos.x = TileIDRot.x * LS.y + LIDrot.x - xoff;
  ElemPos.y = TileIDRot.y * LS.x + LIDrot.y;

  // Finaly write back the matrix element
  //
  // Use the rotated (x,y) coordinate and rotate it BACK.
  // Then we found the correct (x,y) to index into the (not rotated) tile and can read back from local memory
  // RotateBack(x,y) = (y, LS.y - x - 1) = 90 degree ccw rotation 
  if (ElemPos.x >= 0 && ElemPos.x < SizeX && ElemPos.y < SizeY)
    MR[ElemPos.y * SizeY + ElemPos.x] = block[(LS.y - LIDrot.x - 1) * LS.x + LIDrot.y];
}
 
