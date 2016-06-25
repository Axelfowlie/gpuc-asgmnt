


__kernel void ReduceAABBmin(__global float4* AABBs, uint N, uint stride) {
  int pos = get_global_id(0);
  if (pos >= N) return;

  float4 minv = AABBs[pos];
  if (pos + stride < N) minv = min(minv, AABBs[pos + stride]);

  AABBs[pos] = minv;
}

__kernel void ReduceAABBmax(__global float4* AABBs, uint N, uint stride) {
  int pos = get_global_id(0);
  if (pos >= N) return;

  float4 minv = AABBs[pos];
  if (pos + stride < N) minv = max(minv, AABBs[pos + stride]);

  AABBs[pos] = minv;
}


__kernel void MortonCodeAABB(__global float4* AABBout, __global float4* AABBmin, __global float4* AABBmax) {
  if (get_global_id(0) == 0) AABBout[0] = AABBmin[0];
  if (get_global_id(0) == 1) AABBout[1] = AABBmax[0];
}


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
uint expandBits(uint v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}


__constant float4 f1024 = (float4)(1024.0f, 1024.0f, 1024.0f, 1024.0f);
__constant float4 f1023 = (float4)(1023.0f, 1023.0f, 1023.0f, 1023.0f);
__constant float4 f0 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint morton3D(float4 v) {
  v = min(max(v * f1024, f0), f1023);
  uint x = expandBits((uint)(v.x));
  uint y = expandBits((uint)(v.y));
  uint z = expandBits((uint)(v.z));
  return x * 4 + y * 2 + z;
}

__kernel void MortonCodes(__global uint* codes, const __global float4* positions, const __global float4* AABB, uint N) {
  int GID = get_global_id(0);
  if (GID >= N) return;

  float4 minAABB = AABB[0];
  float4 maxAABB = AABB[1];
  float4 sizeAABB = maxAABB - minAABB;

  float4 pos = (positions[GID] - minAABB) / sizeAABB;

  codes[GID] = morton3D(pos);
}

