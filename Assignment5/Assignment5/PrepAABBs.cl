
__kernel void AdvancePositions(__global float4* positions, __global float4* velocities, uint N) {
  int GID = get_global_id(0);
  if (GID >= N) return;

  float4 p = positions[GID];
  float pw = p.w;
  float4 v = velocities[GID];

  p += v;
  p.w = pw;

  if (p.x > 5.1) v.x = -v.x;
  if (p.y > 5.1) v.y = -v.y;
  if (p.z > 5.1) v.z = -v.z;
  if (p.x < -5.1) v.x = -v.x;
  if (p.y < -5.1) v.y = -v.y;
  if (p.z < -5.1) v.z = -v.z;

  positions[GID] = p;
  velocities[GID] = v;
}



__kernel void CreateLeafAABBs(const __global float4* positions, __global float4* AABBmin, __global float4* AABBmax, uint N, uint offset) {
  int GID = get_global_id(0);
  if (GID >= N) return;

  float4 pos = positions[GID];

  float4 size = (float4)(pos.w, pos.w, pos.w, pos.w) / 2;

  AABBmin[offset + GID] = pos - size;
  AABBmax[offset + GID] = pos + size;
}
