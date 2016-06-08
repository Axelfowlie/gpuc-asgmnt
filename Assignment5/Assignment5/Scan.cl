


#define NUM_BANKS 32
#define NUM_BANKS_LOG 5

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
// Simple and one padding element after NUM_BANKS of "normal" elements
#define OFFSET(A) ((A) + (A) / NUM_BANKS)
#else
#define OFFSET(A) (A)
#endif

__kernel void Scan(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock) {
  // LocalID, number of work items in group and number of processed elements in group
  int LID = get_local_id(0);
  int sizeLocal = get_local_size(0);
  int sizeBlock = sizeLocal * 2;
  // Position of the element in the source data
  int GID = get_group_id(0) * sizeBlock + LID;

  // Copy into local memory
  // Save the right element for later, when we write the reduced result into the higher level array.
  // We only need this in one work item, but saving it here prevents having to load a second time from global memory later on.
  // We would have to load from global memory again, because the down sweep kills the last element...
  localBlock[OFFSET(LID)] = array[GID];
  uint last = localBlock[OFFSET(LID + sizeLocal)] = array[GID + sizeLocal];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Up sweep
  int stride = 1;
  while (stride < sizeBlock) {
    // Position of the left element.
    // The position is shifted about [stride - 1] to the right to compensate, that the elements are reduced towards the right side of the array
    //    ...
    //    0L    0R    2L
    //  / |   / |   / |
    // 0L 0R 1L 1R 2L 2R
    //
    // This way the work item with LID==0 always reduces the left most element in the tree
    int left = LID * stride * 2 + stride - 1;
    // Since the work items with lower IDs reduce elements with lower index we can only check for right index < block size
    if (left + stride < sizeBlock) localBlock[OFFSET(left + stride)] += localBlock[OFFSET(left)];

    // Increment stride to next power of 2 and sync memory
    stride <<= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Set the last element int the block to zero
  if (LID == 0) localBlock[OFFSET(sizeBlock - 1)] = 0;
  stride >>= 1;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Down sweep
  while (stride > 0) {
    // Turn the index calculation from above around, to ensure the same property, that
    // to work item with LID==0 always reduces the left most element in the tree
    // The turnaround is needed because we start with stride = sizeBlock/2 and make it smaller each iteration.
    int left = sizeBlock - (LID * stride * 2 + stride) - 1;
    int right = left + stride;

    // The condition to swap the elements is equally simple as above,
    // In an iteration all work items, that index elements outside the memory, have left indices < 0 since the stride is too large...
    if (left >= 0) {
      uint tmp = localBlock[OFFSET(left)];
      localBlock[OFFSET(left)] = localBlock[OFFSET(right)];
      localBlock[OFFSET(right)] += tmp;
    }

    // Decrease stride and sync memory
    stride >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }



  // Write back to main array and the higher level array
  array[GID] = localBlock[OFFSET(LID)];
  array[GID + sizeLocal] = localBlock[OFFSET(LID + sizeLocal)];
  // Write back the result of the reduction. Write with the last work item, since this sis the one with the correct value in last
  if (LID == get_local_size(0) - 1) higherLevelArray[get_group_id(0)] = last + localBlock[OFFSET(LID + sizeLocal)];
}

__kernel void ScanAdd(__global uint* higherLevelArray, __global uint* outArray) {
  // The group size in the Add and in the Scan is the same but the number of elements in each block is double the group size,
  // so we have to account for this in the indexing
  // The first block of [local_size*2] elements does not need to be modified
  //
  // We are allowed to doe the +1 for the higher level index, since we assured that it exists during buffer allocation.
  // Also, there is a valid value in this location, since the value in the very highest level (which is possybly garbage) is never used
  // as well as all the other values make sense, because they are exclusive prefix sums...
  outArray[get_global_id(0) + get_local_size(0) * 2] += higherLevelArray[get_group_id(0) / 2 + 1];
}
