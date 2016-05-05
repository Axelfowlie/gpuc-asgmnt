
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint stride, uint N) {
  int pos = get_global_id(0) * stride * 2;

  uint right = 0;

  if (pos + stride < N) right = array[pos + stride];
  if (pos < N) array[pos] += right;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride) {
  // TO DO: Kernel implementation
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock) {
  // TO DO: Kernel implementation
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock) {
  // TO DO: Kernel implementation
}