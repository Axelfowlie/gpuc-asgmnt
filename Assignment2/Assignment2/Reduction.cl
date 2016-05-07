
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint stride, uint N) {
  int pos = get_global_id(0) * stride * 2;

  uint right = 0;

  if (pos + stride < N) right = array[pos + stride];
  if (pos < N) array[pos] += right;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride, uint N) {
  int pos = get_global_id(0);

  uint right = 0;

  if (pos + stride < N) right = array[pos + stride];
  if (pos < N) array[pos] += right;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock) {
  // Reduce a block length of <local_size * 2> elements
  // This means the right and left elements are <local_size> elements away from each other
  int stride = get_local_size(0);
  // We compute the position of the left element by its local id and the group id multiplied by the doubled local size (=stride between groups)
  int Grp = get_group_id(0);
  int LID = get_local_id(0);
  int pos = LID + Grp * stride * 2;

  // By default copy a zero into the localBlock and add the right and left element, if they are inside the array's range
  // This yields the benefit, that these zero elements are then the identity and we do not need to care about
  // special cases, like out of range elements, in the local reduction.
  uint elem = 0;
  if (pos < N) elem += inArray[pos];
  if (pos + stride < N) elem += inArray[pos + stride];
  localBlock[LID] = elem;

  // Wait for the local block to be written completely
  barrier(CLK_LOCAL_MEM_FENCE);


  // Reduction of the localBlock
  stride >>= 1;
  while (stride > 0) {
    // We do need to check that we do not address elements outside the range of the localBlock.
    // But we do NOT have to take care of elements in the localBlock that do not have a valid element copied from before.
    if (LID < stride) localBlock[LID] += localBlock[LID + stride];
    stride >>= 1;

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Let the first thread in the group write back the result of the local reduction
  if (LID == 0) outArray[Grp] = localBlock[0];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock) {
  // Reduce a block length of <local_size * 2> elements
  // This means the right and left elements are <local_size> elements away from each other
  int stride = get_local_size(0);
  // We compute the position of the left element by its local id and the group id multiplied by the doubled local size (=stride between groups)
  int Grp = get_group_id(0);
  int LID = get_local_id(0);
  int pos = LID + Grp * stride * 2;

  // By default copy a zero into the localBlock and add the right and left element, if they are inside the array's range
  // This yields the benefit, that these zero elements are then the identity and we do not need to care about
  // special cases, like out of range elements, in the local reduction.
  uint elem = 0;
  if (pos < N) elem += inArray[pos];
  if (pos + stride < N) elem += inArray[pos + stride];
  localBlock[LID] = elem;

  // Wait for the local block to be written completely
  barrier(CLK_LOCAL_MEM_FENCE);


  // Reduction of the localBlock
  stride >>= 1;
  while (stride > 64) {
    // We do NOT have to take care of elements in the localBlock that do not have a valid element copied from before.
    // But we DO need to check that we do not address elements outside the range of the localBlock. Or maybe we don't since we wont use them anyway...
    //if (LID < stride) 
    localBlock[LID] += localBlock[LID + stride];
    stride >>= 1;

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  localBlock[LID] += localBlock[LID + 32];
  localBlock[LID] += localBlock[LID + 16];
  localBlock[LID] += localBlock[LID + 8];
  localBlock[LID] += localBlock[LID + 4];
  localBlock[LID] += localBlock[LID + 2];

  // Let the first thread in the group write back the result of the local reduction
  if (LID == 0) outArray[Grp] = localBlock[0] + localBlock[1];
}
