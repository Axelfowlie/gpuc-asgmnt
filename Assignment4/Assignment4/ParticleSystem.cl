

float4 cross3(float4 a, float4 b){
	float4 c;
	c.x = a.y * b.z - b.y * a.z;
	c.y = a.z * b.x - b.z * a.x;
	c.z = a.x * b.y - b.x * a.y;
	c.w = 0.f;
	return c;
}

float dot3(float4 a, float4 b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
}


#define EPSILON 0.001f

// This function expects two points defining a ray (x0 and x1)
// and three vertices stored in v1, v2, and v3 (the last component is not used)
// it returns true if an intersection is found and sets the isectT and isectN
// with the intersection ray parameter and the normal at the intersection point.
bool LineTriangleIntersection(float4 x0, float4 x1, float4 v1, float4 v2, float4 v3, float* isectT, float4* isectN) {
  float4 dir = x1 - x0;
  dir.w = 0.f;

  float4 e1 = v2 - v1;
  float4 e2 = v3 - v1;
  e1.w = 0.f;
  e2.w = 0.f;

  float4 s1 = cross3(dir, e2);
  float divisor = dot3(s1, e1);
  if (divisor == 0.f) return false;
  float invDivisor = 1.f / divisor;

  // Compute first barycentric coordinate
  float4 d = x0 - v1;
  float b1 = dot3(d, s1) * invDivisor;
  if (b1 < -EPSILON || b1 > 1.f + EPSILON) return false;

  // Compute second barycentric coordinate
  float4 s2 = cross3(d, e1);
  float b2 = dot3(dir, s2) * invDivisor;
  if (b2 < -EPSILON || b1 + b2 > 1.f + EPSILON) return false;

  // Compute _t_ to intersection point
  float t = dot3(e2, s2) * invDivisor;
  if (t < -EPSILON || t > 1.f + EPSILON) return false;

  // Store the closest found intersection so far
  *isectT = t;
  *isectN = cross3(e1, e2);
  *isectN = normalize(*isectN);
  return true;
}


bool CheckCollisions(
    float4 x0, float4 x1, __global float4* gTriangleSoup,
    __local float4* lTriangleCache,  // The cache should hold as many vertices as the number of threads (therefore the number of triangles is nThreads/3)
    uint nTriangles, float* t, float4* n) {
  // Set t to a length that is always larger than any t we can possibly find in LineTriangleIntersection
  // which is in this case the squared length of the ray from x0 to x1.
  *t = 1.0f;
  // Initially do not think a collision appeared
  bool res = false;

//#define NO_LOCAL_MEM 1
#ifdef NO_LOCAL_MEM
  // Loop over all triangles
  for (int i = 0; i < nTriangles; ++i) {
    float4 v0 = gTriangleSoup[3 * i];
    float4 v1 = gTriangleSoup[3 * i + 1];
    float4 v2 = gTriangleSoup[3 * i + 2];

    // Perform test for each triangle
    float thist;
    float4 thisn;
    if (LineTriangleIntersection(x0, x1, v0, v1, v2, &thist, &thisn))
      // Update global t and n, only if the test was positive AND the returned thist is smaller
      if (thist < *t) {
        *t = thist;
        *n = thisn;
        res = true;
      }
  }

#else
  // The local size must be a multiple of 3!!! (else we'd have to do the special case and no one wants to do those...)
  int LID = get_local_id(0);
  int chunksize = get_local_size(0);
  int triperchunk = chunksize / 3;

  // We can only load <triperchunk> triangles into the local memory at a time
  // <chunkstart> is the index of the vertex where the currently processed chunk of triangles starts
  for (int chunkstart = 0; chunkstart < nTriangles * 3; chunkstart += chunksize) {
    // Load into the local memory
    if (chunkstart + LID < nTriangles * 3) lTriangleCache[LID] = gTriangleSoup[chunkstart + LID];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Similar to NO_LOCAL_MEM expect to only loop over the current chunk
    for (int i = 0; i < triperchunk; ++i) {
      float thist;
      float4 thisn;
      if (LineTriangleIntersection(x0, x1, lTriangleCache[3 * i], lTriangleCache[3 * i + 1], lTriangleCache[3 * i + 2], &thist, &thisn))
        if (thist < *t) {
          *t = thist;
          *n = thisn;
          res = true;
        }
    }
  }
#endif

  return res;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is the integration kernel. Implement the missing functionality
//
// Input data:
// gAlive         - Field of flag indicating whether the particle with that index is alive (!= 0) or dead (0). You will have to modify this
// gForceField    - 3D texture with the  force field
// sampler        - 3D texture sampler for the force field (see usage below)
// nParticles     - Number of input particles
// nTriangles     - Number of triangles in the scene (for collision detection)
// lTriangleCache - Local memory cache to be used during collision detection for the triangles
// gTriangleSoup  - The triangles in the scene (layout see the description of CheckCollisions())
// gPosLife       - Position (xyz) and remaining lifetime (w) of a particle
// gVelMass       - Velocity vector (xyz) and the mass (w) of a particle
// dT             - The timestep for the integration (the has to be subtracted from the remaining lifetime of each particle)
//
// Output data:
// gAlive   - Updated alive flags
// gPosLife - Updated position and lifetime
// gVelMass - Updated position and mass
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Integrate(__global uint* gAlive, __read_only image3d_t gForceField, sampler_t sampler, uint nParticles, uint nTriangles,
                        __local float4* lTriangleCache, __global float4* gTriangleSoup, __global float4* gPosLife, __global float4* gVelMass, float dT) {
  float4 gGravity = (float4)(0.f, -9.81f, 0.f, 0.f);

  // Verlet Velocity Integration
  float4 x0 = gPosLife[get_global_id(0)];
  float4 v0 = gVelMass[get_global_id(0)];

  float mass = v0.w;
  float life = x0.w;

  // Acceleration at time t is Gravity + Force(x0)/mass
  float4 lookUp = x0;
  lookUp.w = 0.f;
  float4 F = read_imagef(gForceField, sampler, lookUp);
  float4 a0 = gGravity + F / mass;

  // Integrate new position at time t + dT
  float4 x1 = x0 + v0 * dT + 0.5f * a0 * dT * dT;

  // Acceleration at time t+dT is Gravity + Force(x1)/mass
  // THIS IS ONLY TRUE IF THE FORCE FEELD IS STATIC, however I do not know where to get the force at time t + dT if not...
  lookUp = x1;
  lookUp.w = 0.f;
  F = read_imagef(gForceField, sampler, lookUp);
  float4 a1 = gGravity + F / mass;

  // Integrate the velocity
  float4 v1 = v0 + 0.5f * (a0 + a1) * dT;


  float t;
  float4 n;
  bool collision = CheckCollisions(x0, x1, gTriangleSoup, lTriangleCache, nTriangles, &t, &n);
  if (collision) {
    // If a collision happened, stop the particle at the surface intersection point and flip the velocity
    x1 = x0 + t * (x1 - x0) + 0.00001f * n;
    v1 = 0.4f * (2.0f * n - v1);
  }



  // We loose life but no mass
  x1.w = life - dT;
  v1.w = mass;

  gPosLife[get_global_id(0)] = x1;
  gVelMass[get_global_id(0)] = v1;


  // Kill the particle if its life is < 0.0 by setting the corresponding flag in gAlive to 0.
  if (x1.w < 0.0f)
    gAlive[get_global_id(0)] = 0;
  else
    gAlive[get_global_id(0)] = 1;

  // Spawn new particle if collision with too much velocity
  float vl = dot3(v0, n);
  if (collision && vl > 1.0f) {
    float4 newposl = x1;
    float4 newvelm = n * sqrt(vl) * 3;
    newposl.w = 10.0f;
    newvelm.w = 1.0f;

    // Do not spawn new particle, if theres no room for it...
    if (get_global_id(0) < nParticles) {
      gPosLife[get_global_id(0) + nParticles] = newposl;
      gVelMass[get_global_id(0) + nParticles] = newvelm;
      gAlive[get_global_id(0) + nParticles] = 1;
    }
  } else
    gAlive[get_global_id(0) + nParticles] = 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Clear(__global float4* gPosLife, __global float4* gVelMass) {
  uint GID = get_global_id(0);
  gPosLife[GID] = 0.f;
  gVelMass[GID] = 0.f;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reorganize(__global uint* gAlive, __global uint* gRank, __global float4* gPosLifeIn, __global float4* gVelMassIn, __global float4* gPosLifeOut,
                         __global float4* gVelMassOut, uint nParticles) {
  // Re-order the particles according to the gRank obtained from the parallel prefix sum
  // ADD YOUR CODE HERE

  int GID = get_global_id(0);


  uint alive = gAlive[GID];

  if (alive == 1) {
    uint rk = gRank[GID];

    if (rk < nParticles * 2) {
      gPosLifeOut[rk] = gPosLifeIn[GID];
      gVelMassOut[rk] = gVelMassIn[GID];
    }
  }
}
