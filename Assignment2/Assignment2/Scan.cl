


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset) {
  int GID = get_global_id(0);

  if (GID >= N) return;

  uint elem = inArray[GID];
  if (offset <= GID) elem += inArray[GID - offset];
  outArray[GID] = elem;
}



// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
#define NUM_BANKS 32
#define NUM_BANKS_LOG 5
#define SIMD_GROUP_SIZE 32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
// Simple and one padding element after NUM_BANKS of "normal" elements
#define OFFSET(A) ((A) + (A) / NUM_BANKS)
#else
#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock) {
  // LocalID, number of work items in group and number of processed elements in group
  int LID = get_local_id(0);
  int sizeLocal = get_local_size(0);
  int sizeBlock = sizeLocal * 2;
  // Position of the element in the source data
  int GID = get_group_id(0) * sizeBlock + LID;

  // Copy into local memory
  localBlock[OFFSET(LID)] = array[GID];
  localBlock[OFFSET(LID + sizeLocal)] = array[GID + sizeLocal];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Up sweep
  int stride = 1;
  while (stride < sizeBlock) {
    // Position of the left element, it is shifted about [stride - 1] to the right to compensate,
    // that the elements are reduced towards the right side of the array
    int left = LID * stride * 2 + stride - 1;
    if (left + stride < sizeBlock) localBlock[OFFSET(left + stride)] += localBlock[OFFSET(left)];
    stride <<= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Set the last element int the block to zero
  if (LID == 0) localBlock[OFFSET(sizeBlock - 1)] = 0;
  stride >>= 1;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Down sweep
  while (stride > 0) {
    // Right element of the down sweep
    // The LID gets inverted, so that the work item with the hightes ID gets to write the element with the highest index.
    int right = sizeBlock - (sizeLocal - LID - 1) * stride * 2 - 1;
    int left = right - stride;
    // The way the right element is indexed, we only have to prevent work items not to touch elements with negative indices
    if (left >= 0) {
      uint tmp = localBlock[OFFSET(left)];
      localBlock[OFFSET(left)] = localBlock[OFFSET(right)];
      localBlock[OFFSET(right)] += tmp;
    }

    stride >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write back to main array and the higher level array
  array[GID] += localBlock[OFFSET(LID)];
  array[GID + sizeLocal] += localBlock[OFFSET(LID + sizeLocal)];
  // The work item that wrote the last element in the array, writes the one in the higher level too.
  if (LID == get_local_size(0) - 1) higherLevelArray[get_group_id(0)] = array[GID + sizeLocal];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array, __local uint* localBlock) {
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array) {
  // The group size in the Add and in the Scan is the same but the number of elements in each block is double the group size,
  // so we have to account for this in the indexing
  // The first block of [local_size*2] elements does not need to be modified
  array[get_global_id(0) + get_local_size(0) * 2] += higherLevelArray[get_group_id(0) / 2];
}
