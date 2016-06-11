

// For each key write a flag into flagnotset and flagset
// If all 1 bits in mask are also set in key, writes a 1 into flagset and a 0 into flagnotset
// If at least one set bit in mask is not set in key, writes 1 and 0 the other way round.
__kernel void SelectBitflag(__global uint* flagnotset, __global uint* flagset, const __global uint* keys, uint mask, uint N) {
  int GID = get_global_id(0);

  // Write flag not set for all items out of range
  if (GID >= N) {
    flagnotset[GID] = 0;
    flagset[GID] = 0;
    return;
  }

  // By default thinkt the bit in mask is not set in key
  uint rflagnotset = 1;
  uint rflagset = 0;

  // If he bit is set in key, swap the results
  if ((keys[GID] & mask) == mask) {
    rflagnotset = 0;
    rflagset = 1;
  }

  // Write the results to global memory
  flagnotset[GID] = rflagnotset;
  flagset[GID] = rflagset;
}



__kernel void ReorderKeys(__global uint* keys, __global uint* permutation, const __global uint* indexzerobits, const __global uint* indexonebits, const __global uint* offset) {
}
