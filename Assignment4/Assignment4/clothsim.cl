#define DAMPING 0.02f

#define G_GRAVITY (float4)(0.f, -9.81f, 0.f, 0.f)
#define G_WIND (float4)(0.f, 0.f, sin(simulationTime), 0.f)

#define WEIGHT_ORTHO	0.138f
#define WEIGHT_DIAG		0.097f
#define WEIGHT_ORTHO_2 0.069f
#define WEIGHT_DIAG_2 0.048f


#define ROOT_OF_2 1.4142135f
#define DOUBLE_ROOT_OF_2 2.8284271f



///////////////////////////////////////////////////////////////////////////////
// The integration kernel
// Input data:
// width and height - the dimensions of the particle grid
// d_pos - the most recent position of the cloth particle while...
// d_prevPos - ...contains the position from the previous iteration.
// elapsedTime      - contains the elapsed time since the previous invocation of the kernel,
// prevElapsedTime  - contains the previous time step.
// simulationTime   - contains the time elapsed since the start of the simulation (useful for wind)
// All time values are given in seconds.
//
// Output data:
// d_prevPos - Input data from d_pos must be copied to this array
// d_pos     - Updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void Integrate(unsigned int width, unsigned int height, __global float4* d_pos, __global float4* d_prevPos, float elapsedTime, float prevElapsedTime,
                        float simulationTime) {
  // Make sure the work-item does not map outside the cloth
  if (get_global_id(0) >= width || get_global_id(1) >= height) return;

  unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
  // This is just to keep every 8th particle of the first row attached to the bar
  if (particleID > width - 1 || (particleID & (7)) != 0) {

    float4 pPrev = d_prevPos[particleID];
    float4 pNow = d_pos[particleID];

    // If this is missing everything goes to hell, because d_prevPos is not initialized...
    // The better practice would be to do this in the host code, 
    // but I don't know about copying a gl vertex buffer into a cl buffer, so I took the lazy way out...
    if (prevElapsedTime == 0.0f) pPrev = pNow;

    // From wikipedia...
    // We use gravity and wind, who's strength is increasing with simulation time.
    float4 pNext = 2 * pNow - pPrev + elapsedTime * elapsedTime * (1-DAMPING) * (G_GRAVITY + G_WIND * simulationTime);

    // The next previous position is the current position...
    d_prevPos[particleID] = pNow;
    // and the next current position is the next position.
    d_pos[particleID] = pNext;
  }
}


// Bit masks for taring of springs
// Each spring of a particle is mapped by one bit in a 4byte uint
// Luckily there are exactly 32 springs for each particle, so we can do something like this for all springs in the 5x5 region
// (in Hex)
// 0001   --   0002   --   0004
//  --   0100  1000  0200   --
// 0008  2000   --   4000  0010
//  --   0400  8000  0800   --
// 0020   --   0040   --   0080

#define BEND_TOP_LEFT 0x0001
#define BEND_TOP 0x0002
#define BEND_TOP_RIGHT 0x0004
#define BEND_LEFT 0x0008
#define BEND_RIGHT 0x0010
#define BEND_BOT_LEFT 0x0020
#define BEND_BOT 0x0040
#define BEND_BOT_RIGHT 0x0080
#define SHEAR_TOP_LEFT 0x0100
#define SHEAR_TOP_RIGHT 0x0200
#define SHEAR_BOT_LEFT 0x0400
#define SHEAR_BOT_RIGHT 0x0800
#define STRUCTURE_TOP 0x1000
#define STRUCTURE_LEFT 0x2000
#define STRUCTURE_RIGHT 0x4000
#define STRUCTURE_BOT 0x8000

#define TILE_X 16
#define TILE_Y 16
#define HALOSIZE 2

#define TILE_X_H 20
#define TILE_Y_H 20

#define USE_LOCAL_MEM
#define ENABLE_TARE



///////////////////////////////////////////////////////////////////////////////
// Input data:
// pos1 and pos2 - The positions of two particles
// restDistance  - the distance between the given particles at rest
//
// Return data:
// correction vector for particle 1
///////////////////////////////////////////////////////////////////////////////
float4 SatisfyConstraint(uint* springFlag, uint springMask, float4 pos1, float4 pos2, float restDistance) {
  float4 toNeighbor = pos2 - pos1;

#ifdef ENABLE_TARE
  // If the spring is inactive or if the length of the spring is too large we simply return 0 as correction vector.
  if ((*springFlag & springMask) == 0 || length(toNeighbor) > restDistance * 5.5) {
    // This might be the first time the spring was stretched too far, so we kill it by setting the corresponding spring bit to 0
    *springFlag &= !springMask;
    return (float4)(0, 0, 0, 0);
  } else
    // The spring is active and not stretched too far, so we can correct the position as usual...
    return (toNeighbor - normalize(toNeighbor) * restDistance);
#else
  return (toNeighbor - normalize(toNeighbor) * restDistance);
#endif
}


#ifdef USE_LOCAL_MEM
///////////////////////////////////////////////////////////////////////////////
// Input data:
// springFlag - pointer to the spring flag indicating what springs are active
// mypos - particle position of the center particle
// restDistance  - the distance between the given particles at rest
// GID - XY-coor of the center particle in global memory
// LID - XY-coor of the center particle in local memory
// width - Width of the cloth
// height - Height of the cloth
// tile - local memory with the particle positions
//
// Return data:
// correction vector for the center particle
// springFlag - the new springFlag (may change in case a soring breaks)
///////////////////////////////////////////////////////////////////////////////
float4 SatisfyConstraintAll(uint* springFlag, float4 mypos, float restDistance, int2 GID, int2 LID, int width, int height, __local float4* tile) {
  float4 corr = (float4)(0, 0, 0, 0);

  // Scaled restDistance for shear and bend constraints.
  float rdDiag = restDistance * ROOT_OF_2;
  float rdDiag2 = restDistance * DOUBLE_ROOT_OF_2;
  float rd2 = restDistance * 2;

  // Start with particle in the upper left of the tile. Then compute correction for for each row...
  int2 otherGID = GID + (int2)(-2, -2);
  if (otherGID.y >= 0) {
    if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, BEND_TOP_LEFT, mypos, tile[(LID.y - 2) * TILE_X_H + LID.x - 2], rdDiag2) * WEIGHT_DIAG_2;

    otherGID.x += 2;
    corr += SatisfyConstraint(springFlag, BEND_TOP, mypos, tile[(LID.y - 2) * TILE_X_H + LID.x], rd2) * WEIGHT_ORTHO_2;

    otherGID.x += 2;
    if (otherGID.x < width) corr += SatisfyConstraint(springFlag, BEND_TOP_RIGHT, mypos, tile[(LID.y - 2) * TILE_X_H + LID.x + 2], rdDiag2) * WEIGHT_DIAG_2;
  }

  // Second row
  otherGID = GID + (int2)(-1, -1);
  if (otherGID.y >= 0) {
    if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, SHEAR_TOP_LEFT, mypos, tile[(LID.y - 1) * TILE_X_H + LID.x - 1], rdDiag) * WEIGHT_DIAG;

    ++otherGID.x;
    corr += SatisfyConstraint(springFlag, STRUCTURE_TOP, mypos, tile[(LID.y - 1) * TILE_X_H + LID.x], restDistance) * WEIGHT_ORTHO;

    ++otherGID.x;
    if (otherGID.x < width) corr += SatisfyConstraint(springFlag, SHEAR_TOP_RIGHT, mypos, tile[(LID.y - 1) * TILE_X_H + LID.x + 1], rdDiag) * WEIGHT_DIAG;
  }


  // Central row
  otherGID = GID + (int2)(-2, 0);
  if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, BEND_LEFT, mypos, tile[LID.y * TILE_X_H + LID.x - 2], rd2) * WEIGHT_ORTHO_2;

  ++otherGID.x;
  if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, STRUCTURE_LEFT, mypos, tile[LID.y * TILE_X_H + LID.x - 1], restDistance) * WEIGHT_ORTHO;

  otherGID.x += 2;
  if (otherGID.x < width) corr += SatisfyConstraint(springFlag, STRUCTURE_RIGHT, mypos, tile[LID.y * TILE_X_H + LID.x + 1], restDistance) * WEIGHT_ORTHO;

  ++otherGID.x;
  if (otherGID.x < width) corr += SatisfyConstraint(springFlag, BEND_RIGHT, mypos, tile[LID.y * TILE_X_H + LID.x + 2], rd2) * WEIGHT_ORTHO_2;


  // Third row
  otherGID = GID + (int2)(-1, 1);
  if (otherGID.y < height) {
    if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, SHEAR_BOT_LEFT, mypos, tile[(LID.y + 1) * TILE_X_H + LID.x - 1], rdDiag) * WEIGHT_DIAG;

    ++otherGID.x;
    corr += SatisfyConstraint(springFlag, STRUCTURE_BOT, mypos, tile[(LID.y + 1) * TILE_X_H + LID.x], restDistance) * WEIGHT_ORTHO;

    ++otherGID.x;
    if (otherGID.x < width) corr += SatisfyConstraint(springFlag, SHEAR_BOT_RIGHT, mypos, tile[(LID.y + 1) * TILE_X_H + LID.x + 1], rdDiag) * WEIGHT_DIAG;
  }

  //// Last row
  otherGID = GID + (int2)(-2, 2);
  if (otherGID.y < height) {
    if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, BEND_BOT_LEFT, mypos, tile[(LID.y + 2) * TILE_X_H + LID.x - 2], rdDiag2) * WEIGHT_DIAG_2;

    otherGID.x += 2;
    corr += SatisfyConstraint(springFlag, BEND_BOT, mypos, tile[(LID.y + 2) * TILE_X_H + LID.x], rd2) * WEIGHT_ORTHO_2;

    otherGID.x += 2;
    if (otherGID.x < width) corr += SatisfyConstraint(springFlag, BEND_BOT_RIGHT, mypos, tile[(LID.y + 2) * TILE_X_H + LID.x + 2], rdDiag2) * WEIGHT_DIAG_2;
  }

  return corr;
}
#else
float4 SatisfyConstraintAll(uint* springFlag, float4 mypos, float restDistance, int2 GID, int width, int height, __global float4* pos) {
  float4 corr = (float4)(0, 0, 0, 0);

  float rdDiag = restDistance * ROOT_OF_2;
  float rdDiag2 = restDistance * DOUBLE_ROOT_OF_2;
  float rd2 = restDistance * 2;

  // Start with particle in the upper left of the tile. Then compute correction for for each row...
  int2 otherGID = GID + (int2)(-2, -2);
  if (otherGID.y >= 0) {
    if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, BEND_TOP_LEFT, mypos, pos[otherGID.y * width + otherGID.x], rdDiag2) * WEIGHT_DIAG_2;

    otherGID.x += 2;
    corr += SatisfyConstraint(springFlag, BEND_LEFT, mypos, pos[otherGID.y * width + otherGID.x], rd2) * WEIGHT_ORTHO_2;

    otherGID.x += 2;
    if (otherGID.x < width) corr += SatisfyConstraint(springFlag, BEND_TOP_RIGHT, mypos, pos[otherGID.y * width + otherGID.x], rdDiag2) * WEIGHT_DIAG_2;
  }

  // Second row
  otherGID = GID + (int2)(-1, -1);
  if (otherGID.y >= 0) {
    if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, SHEAR_TOP_LEFT, mypos, pos[otherGID.y * width + otherGID.x], rdDiag) * WEIGHT_DIAG;

    ++otherGID.x;
    corr += SatisfyConstraint(springFlag, STRUCTURE_TOP, mypos, pos[otherGID.y * width + otherGID.x], restDistance) * WEIGHT_ORTHO;

    ++otherGID.x;
    if (otherGID.x < width) corr += SatisfyConstraint(springFlag, SHEAR_TOP_RIGHT, mypos, pos[otherGID.y * width + otherGID.x], rdDiag) * WEIGHT_DIAG;
  }


  // Central row
  otherGID = GID + (int2)(-2, 0);
  if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, BEND_LEFT, mypos, pos[otherGID.y * width + otherGID.x], rd2) * WEIGHT_ORTHO_2;

  ++otherGID.x;
  if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, STRUCTURE_LEFT, mypos, pos[otherGID.y * width + otherGID.x], restDistance) * WEIGHT_ORTHO;

  otherGID.x += 2;
  if (otherGID.x < width) corr += SatisfyConstraint(springFlag, STRUCTURE_RIGHT, mypos, pos[otherGID.y * width + otherGID.x], restDistance) * WEIGHT_ORTHO;

  ++otherGID.x;
  if (otherGID.x < width) corr += SatisfyConstraint(springFlag, BEND_LEFT, mypos, pos[otherGID.y * width + otherGID.x], rd2) * WEIGHT_ORTHO_2;


  // Third row
  otherGID = GID + (int2)(-1, 1);
  if (otherGID.y < height) {
    if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, SHEAR_BOT_LEFT, mypos, pos[otherGID.y * width + otherGID.x], rdDiag) * WEIGHT_DIAG;

    ++otherGID.x;
    corr += SatisfyConstraint(springFlag, STRUCTURE_BOT, mypos, pos[otherGID.y * width + otherGID.x], restDistance) * WEIGHT_ORTHO;

    ++otherGID.x;
    if (otherGID.x < width) corr += SatisfyConstraint(springFlag, SHEAR_BOT_RIGHT, mypos, pos[otherGID.y * width + otherGID.x], rdDiag) * WEIGHT_DIAG;
  }

  // Last row
  otherGID = GID + (int2)(-2, 2);
  if (otherGID.y < height) {
    if (otherGID.x >= 0) corr += SatisfyConstraint(springFlag, BEND_BOT_LEFT, mypos, pos[otherGID.y * width + otherGID.x], rdDiag2) * WEIGHT_DIAG_2;

    otherGID.x += 2;
    corr += SatisfyConstraint(springFlag, BEND_BOT, mypos, pos[otherGID.y * width + otherGID.x], rd2) * WEIGHT_ORTHO_2;

    otherGID.x += 2;
    if (otherGID.x < width) corr += SatisfyConstraint(springFlag, BEND_BOT_RIGHT, mypos, pos[otherGID.y * width + otherGID.x], rdDiag2) * WEIGHT_DIAG_2;
  }

  return corr;
}
#endif


void loadLocalTile(uint width, uint height, __local float4* tile, __global float4 const* d_posIn) {
  // Flattened ID of the work item in the group, and total number of work items in the group
  int flatID = get_local_id(1) * get_local_size(0) + get_local_id(0);
  int numItems = get_local_size(0) * get_local_size(1);

  // XY-coor that corresponds to the (0, 0) element in the local tile (includes halo)
  // This coor can contain negative values, in case the work group id is 0 (on left/upper bound)
  int2 tileUpperLeft = (int2)(get_group_id(0) * get_local_size(0) - 2, get_group_id(1) * get_local_size(1) - 2);

  // Every work item starts loading the element at the position of its flattened id
  // We keep loading elements until the flat id exceeds the length of the local memory
  int currentFlatID = flatID;
  while (currentFlatID < TILE_Y_H * TILE_X_H) {
    // Find the XY-coor that corresponds to the flat id when indexing with the width and height of the local tile (including halo)
    int2 coorTile;
    coorTile.x = currentFlatID % (TILE_X_H);
    coorTile.y = currentFlatID / (TILE_X_H);

    // The coor of the current element in global memory, is the coor in the tile + the corresponding (0,0) coor in global memory
    int2 loadGlobal = tileUpperLeft + coorTile;
    // Load only if inside bounds
    if (loadGlobal.x >= 0 && loadGlobal.y >= 0 && loadGlobal.x < width && loadGlobal.y < height)
      tile[coorTile.y *TILE_X_H + coorTile.x] = d_posIn[loadGlobal.y * width + loadGlobal.x];

    // We loaded exactly the number of elements as there are work items in the group
    // Since we compute all indices from the flattened ID of the tile we can just increment like this
    currentFlatID += numItems;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
}


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// restDistance     - the distance between two orthogonally neighboring particles at rest
// d_posIn          - the input positions
//
// Output data:
// d_posOut - new positions must be written here
///////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1))) __kernel
    void SatisfyConstraints(unsigned int width, unsigned int height, float restDistance, float4 spherePos, float sphereRad, __global uint* d_springFlag, __global float4* d_posOut,
                            __global float4 const* d_posIn) {
  int2 GID = (int2)(get_global_id(0), get_global_id(1));
  int particleID = GID.y * width + GID.x;
  float4 mypos;
  uint springFlag = d_springFlag[particleID];
  float4 corr = (float4)(0, 0, 0, 0);


#ifdef USE_LOCAL_MEM
  // Local memory to store the positions
  __local float4 tile[TILE_Y_H][TILE_X_H];

  // Load positions into local memory
  loadLocalTile(width, height, tile, d_posIn);

  // Discard everything thats outside bounds
  if (GID.x >= width || GID.y >= height) return;

  // Local coor in the tile including halo that corresponds to the "real" local ID
  int2 LID = (int2)(get_local_id(0) + 2, get_local_id(1) + 2);
  mypos = tile[LID.y][LID.x];

  corr = SatisfyConstraintAll(&springFlag, mypos, restDistance, GID, LID, width, height, tile);

#else


  if (GID.x >= width || GID.y >= height) return;
  mypos = d_posIn[particleID];
  corr = SatisfyConstraintAll(&springFlag, mypos, restDistance, GID, width, height, d_posIn);
#endif

  // Clamp to restDistance / 2, so that we don't accidentally 
  //    - blow up the simulation, 
  //    - melt any hardware, 
  //    - initiate the zombie apocalypse 
  //    - or cause World War III 
  if (length(corr) > restDistance * 0.5f) corr = normalize(corr) * restDistance * 0.5f;

  // Only modify the particles not connected to the bar
  if (particleID > width - 1 || (particleID & (7)) != 0) mypos += corr;


  // Do the collision detection right here, so that we can save the overhead of launching a second kernel.
  float4 CtoP = mypos - spherePos;
  float d = length(CtoP);
  if (d < sphereRad) mypos = spherePos + CtoP / d * sphereRad;

  // Write back the corrected position and new spring Flag
  d_posOut[particleID] = mypos;
  d_springFlag[particleID] = springFlag;
}


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// d_pos            - the input positions
// spherePos        - The position of the sphere (xyz)
// sphereRad        - The radius of the sphere
//
// Output data:
// d_pos            - The updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void CheckCollisions(unsigned int width, unsigned int height, __global float4* d_pos, float4 spherePos, float sphereRad) {
  int2 GID = (int2)(get_global_id(0), get_global_id(1));
  if (GID.x >= width || GID.y >= height) return;

  int particleID = GID.y * width + GID.x;
  float4 mypos = d_pos[particleID];

  // Distance d of the particle to the center of the sphere.
  float4 CtoP = mypos - spherePos;
  float d = length(CtoP);
  // If d is smaller than the radius the particle is inside the sphere
  // Normally I would multiply an epsilon to the right summand but I saw that the sphere is rendered with radius * 0.005...
  if (d < sphereRad) d_pos[particleID] = spherePos + CtoP / d * sphereRad;
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this function!
///////////////////////////////////////////////////////////////////////////////
float4 CalcTriangleNormal(float4 p1, float4 p2, float4 p3) {
  float4 v1 = p2 - p1;
  float4 v2 = p3 - p1;

  return cross(v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this kernel!
///////////////////////////////////////////////////////////////////////////////
__kernel void ComputeNormals(unsigned int width, unsigned int height, __global float4* d_pos, __global float4* d_normal) {
  int particleID = get_global_id(0) + get_global_id(1) * width;
  float4 normal = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  int minX, maxX, minY, maxY, cntX, cntY;
  minX = max((int)(0), (int)(get_global_id(0) - 1));
  maxX = min((int)(width - 1), (int)(get_global_id(0) + 1));
    minY = max( (int)(0), (int)(get_global_id(1)-1));
    maxY = min( (int)(height-1), (int)(get_global_id(1)+1));
    
    for( cntX = minX; cntX < maxX; ++cntX) {
        for( cntY = minY; cntY < maxY; ++cntY) {
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY+1)],
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
        }
    }
    d_normal[particleID] = normalize( normal);
}
