

__kernel void SetBitflag(__global uint* bitflags, const __global uint* keys, uint mask, uint flagset, uint flagnotset, uint N) {
  int GID = get_global_id(0);

  if (GID >= N) {
    outArray[GID] = 0;
    return;
  }

  if ((inArray[GID] & mask) == mask)
    outArray[GID] = flagset;
  else
    outArray[GID] = flagnotset;
}

