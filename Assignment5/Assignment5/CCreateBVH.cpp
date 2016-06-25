/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CCreateBVH.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"

#ifdef min // these macros are defined under windows, but collide with our math utility
#	undef min
#endif
#ifdef max
#	undef max
#endif

#include "HLSLEx.h"

#include <set>
#include <bitset>
#include <string>
#include <algorithm>
#include <string.h>
#include <iomanip>

#include <sstream>

#include "CL/cl_gl.h"

#define NUM_FORCE_LINES 4096
#define NUM_BANKS 32

using namespace std;
using namespace hlsl;

#ifdef _MSC_VER
// we would like to use fopen...
#pragma warning(disable : 4996)
// unreferenced local parameter: as some code is missing from the startup kit, we don't want the compiler complaining about these
#pragma warning(disable : 4100)
#endif

///////////////////////////////////////////////////////////////////////////////
// CCreateBVH

CCreateBVH::CCreateBVH(const std::string& CollisionMeshPath, size_t NElems, size_t scanWorksize, size_t LocalWorkSize[3])
    : m_nElements(NElems), m_CollisionMeshPath(CollisionMeshPath) {
  m_RotateX = 0;
  m_RotateY = 0;
  m_TranslateZ = -1.5f;

  // Work size for scan
  m_ScanLocalWorkSize[0] = scanWorksize;
  m_ScanLocalWorkSize[1] = 1;
  m_ScanLocalWorkSize[2] = 1;

  // Compute the number of levels, and their buffer sizes for the scan.
  size_t N = m_nElements;
  while (N > 0) {
    // Buffer gets created in InitResources, but we can determine the buffer size.
    // Every level needs to be a multiple of 2*scanWorksize, because then we have enough spare elements to store all the local PPS
    m_clScanLevels.push_back(pair<cl_mem, size_t>(nullptr, CLUtil::GetGlobalWorkSize(N, 2 * scanWorksize)));
    N /= 2 * scanWorksize;
  }
  m_clScanLevels.push_back(pair<cl_mem, size_t>(nullptr, scanWorksize * 2));


  for (int i = 0; i < 255; i++) m_KeyboardMask[i] = false;
}

CCreateBVH::~CCreateBVH() {
  ReleaseResources();
}

bool CCreateBVH::InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue) {
  (void)CommandQueue;


  if (!InitGL()) return false;


  srand(time(0));

  cl_int clError;
  string programCode;

  //
  //
  //
  //
  // ########################################################
  //  ### PREP KERNELS (CREATE AABBs, ADVANCE POSITIONS) ###
  // ########################################################
  // Buffer for center positions with radius of leaf node AABBs
  m_clPositions = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_float4) * m_nElements, NULL, &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create positions buffer.");
  // Initialize with random values
  std::vector<cl_float4> initpos(m_nElements);
  for (auto& p : initpos) {
    p.s[0] = (float(rand()) / float(RAND_MAX) * 10.0f - 5.0f);
    p.s[1] = (float(rand()) / float(RAND_MAX) * 10.0f - 5.0f);
    p.s[2] = (float(rand()) / float(RAND_MAX) * 10.0f - 5.0f);
    p.s[3] = (float(rand()) / float(RAND_MAX) * 0.1 + 0.1f);
  }
  V_RETURN_FALSE_CL(clEnqueueWriteBuffer(CommandQueue, m_clPositions, CL_FALSE, 0, m_nElements * sizeof(cl_float4), initpos.data(), 0, NULL, NULL),
                    "Error writing positions to cl memory!");


  // Buffer for velocity of each leaf node AABB center point
  m_clVelocities = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_float4) * m_nElements, NULL, &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create positions buffer.");
  // Initialize with random values
  std::vector<cl_float4> initvel(m_nElements);
  for (auto& p : initvel) {
    p.s[0] = (float(rand()) / float(RAND_MAX) * 0.0025f - 0.00125f);
    p.s[1] = (float(rand()) / float(RAND_MAX) * 0.0025f - 0.00125f);
    p.s[2] = (float(rand()) / float(RAND_MAX) * 0.0025f - 0.00125f);
  }
  V_RETURN_FALSE_CL(clEnqueueWriteBuffer(CommandQueue, m_clVelocities, CL_FALSE, 0, m_nElements * sizeof(cl_float4), initvel.data(), 0, NULL, NULL),
                    "Error writing positions to cl memory!");



  // Create buffers for AABB leafs and inner nodes from open gl buffers
  m_clAABBLeaves[0] = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE, m_glAABBLeafBuf[0], &clError);
  m_clAABBLeaves[1] = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE, m_glAABBLeafBuf[1], &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create AABBLeaves buffer.");
  m_clAABBNodes[0] = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE, m_glAABBNodeBuf[0], &clError);
  m_clAABBNodes[1] = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE, m_glAABBNodeBuf[1], &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create AABBNodes buffer.");

  m_clMortonAABB = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_float4) * 2, NULL, &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create MortonAABB buffer.");

  // Create buffers for internal node's children and parents indices
  m_clNodeChildren = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE, m_glNodeChildrenBuf, &clError);
  m_clNodeParents = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE, m_glNodeParentsBuf, &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create internal nodes buffer.");



  // Kernel for building the leaf node AABBs from the center position and radius
  CLUtil::LoadProgramSourceToMemory("PrepAABBs.cl", programCode);
  m_PrepAABBsProgram = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
  if (!m_PrepAABBsProgram) return false;

  m_AdvancePositionsKernel = clCreateKernel(m_PrepAABBsProgram, "AdvancePositions", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create AdvancePositions kernel.");
  m_CreateLeafAABBsKernel = clCreateKernel(m_PrepAABBsProgram, "CreateLeafAABBs", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create CreateLeafAAPPs kernel.");


  // Kernel for calculating morton codes
  CLUtil::LoadProgramSourceToMemory("MortonCodes.cl", programCode);
  m_MortonCodesProgram = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
  if (!m_MortonCodesProgram) return false;

  m_ReduceAABBminKernel = clCreateKernel(m_MortonCodesProgram, "ReduceAABBmin", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create ReduceAABBkernelmin kernel.");
  m_ReduceAABBmaxKernel = clCreateKernel(m_MortonCodesProgram, "ReduceAABBmax", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create ReduceAABBkernelmax kernel.");

  m_MortonCodeAABBKernel = clCreateKernel(m_MortonCodesProgram, "MortonCodeAABB", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create MortonCodeAABB kernel.");
  m_MortonCodesKernel = clCreateKernel(m_MortonCodesProgram, "MortonCodes", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create MortonCodes kernel.");


  // Kernel for calculating internal nodes
  CLUtil::LoadProgramSourceToMemory("BVHNodes.cl", programCode);
  m_BVHNodesProgram = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
  if (!m_BVHNodesProgram) return false;

  m_NodesHierarchyKernel = clCreateKernel(m_BVHNodesProgram, "NodesHierarchy", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create NodesHierarchy kernel.");

  //
  //
  //
  //
  // ####################
  //  ### RADIX SORT ###
  // ####################
  // Levels for radix sort scan
  // Do not create a buffer for the first level, since we will create separate buffers as inputs
  for (size_t l = 1; l < m_clScanLevels.size(); ++l) {
    m_clScanLevels[l].first = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[l].second, NULL, &clError);
    V_RETURN_FALSE_CL(clError, "Error allocating device arrays");
  }

  // Scan kernels
  CLUtil::LoadProgramSourceToMemory("Scan.cl", programCode);
  m_ScanProgram = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
  if (!m_ScanProgram) return false;

  m_ScanKernel = clCreateKernel(m_ScanProgram, "Scan", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create Scan kernel.");
  m_ScanAddKernel = clCreateKernel(m_ScanProgram, "ScanAdd", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create ScanAdd kernel.");



  // Morton code for each bounding volume
  m_clMortonCodes = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  m_clSortPermutation = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  // Ping-pong buffers for reorder
  m_clRadixKeysPong = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  m_clRadixPermutationPong = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  // Radix bit buffers
  // Buffers that holds the flags where the bit of the current radix is zero/one
  m_clRadixZeroBit = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  m_clRadixOneBit = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  // Temp buffer for the permute kernel to reorder positions and velocities
  m_clPermuteTemp = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_float4) * m_nElements, NULL, &clError);


  // Sort kernels
  CLUtil::LoadProgramSourceToMemory("RadixSort.cl", programCode);
  m_RadixSortProgram = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
  if (!m_ScanProgram) return false;

  m_SelectBitflagKernel = clCreateKernel(m_RadixSortProgram, "SelectBitflag", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create SelectBitflag kernel.");
  m_ReorderKeysKernel = clCreateKernel(m_RadixSortProgram, "ReorderKeys", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create ReorderKeys kernel.");
  m_PermutationIdentityKernel = clCreateKernel(m_RadixSortProgram, "PermutationIdentity", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create PermutationIdentity kernel.");
  m_PermuteKernel = clCreateKernel(m_RadixSortProgram, "Permute", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create Permute kernel.");


  return true;
}

bool CCreateBVH::InitGL() {
  // Load mesh
  float4x4 M = float4x4(1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.5f, 0.f, 0.5f, 0.f);

  m_pMesh = CTriMesh::LoadFromObj(m_CollisionMeshPath, M);

  m_pMesh->GetVertexBuffer();
  if (!m_pMesh) {
    cout << "Failed to load mesh." << endl;
    return false;
  }
  if (!m_pMesh->CreateGLResources()) {
    cout << "Failed to create mesh OpenGL resources" << endl;
    return false;
  }



  // Create buffers for leaf node AABBs
  glGenBuffers(2, m_glAABBLeafBuf);
  glBindBuffer(GL_ARRAY_BUFFER, m_glAABBLeafBuf[0]);
  glBufferData(GL_ARRAY_BUFFER, m_nElements * sizeof(cl_float4), NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, m_glAABBLeafBuf[1]);
  glBufferData(GL_ARRAY_BUFFER, m_nElements * sizeof(cl_float4), NULL, GL_DYNAMIC_DRAW);
  CHECK_FOR_OGL_ERROR();

  // Create texture buffers
  glGenTextures(2, m_glAABBLeafTB);
  glBindTexture(GL_TEXTURE_BUFFER, m_glAABBLeafTB[0]);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F_ARB, m_glAABBLeafBuf[0]);
  glBindTexture(GL_TEXTURE_BUFFER, m_glAABBLeafTB[1]);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F_ARB, m_glAABBLeafBuf[1]);
  glBindBuffer(GL_TEXTURE_BUFFER, 0);
  glBindTexture(GL_TEXTURE_BUFFER, 0);
  CHECK_FOR_OGL_ERROR();


  // Create buffers for inner node AABBs
  glGenBuffers(2, m_glAABBNodeBuf);
  glBindBuffer(GL_ARRAY_BUFFER, m_glAABBNodeBuf[0]);
  glBufferData(GL_ARRAY_BUFFER, m_nElements * sizeof(cl_float4), NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, m_glAABBNodeBuf[1]);
  glBufferData(GL_ARRAY_BUFFER, m_nElements * sizeof(cl_float4), NULL, GL_DYNAMIC_DRAW);
  CHECK_FOR_OGL_ERROR();

  // Create texture buffers
  glGenTextures(2, m_glAABBNodeTB);
  glBindTexture(GL_TEXTURE_BUFFER, m_glAABBNodeTB[0]);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F_ARB, m_glAABBNodeBuf[0]);
  glBindTexture(GL_TEXTURE_BUFFER, m_glAABBNodeTB[1]);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F_ARB, m_glAABBNodeBuf[1]);
  glBindBuffer(GL_TEXTURE_BUFFER, 0);
  glBindTexture(GL_TEXTURE_BUFFER, 0);
  CHECK_FOR_OGL_ERROR();



  // Create buffer for internal node children indices
  // For each internal node (m_nElements - 1) left and right child index
  glGenBuffers(1, &m_glNodeChildrenBuf);
  glBindBuffer(GL_ARRAY_BUFFER, m_glNodeChildrenBuf);
  glBufferData(GL_ARRAY_BUFFER, m_nElements * sizeof(cl_uint2), NULL, GL_DYNAMIC_DRAW);
  CHECK_FOR_OGL_ERROR();

  // Create texture buffer
  glGenTextures(1, &m_glNodeChildrenTB);
  glBindTexture(GL_TEXTURE_BUFFER, m_glNodeChildrenTB);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32UI, m_glNodeChildrenTB);
  glBindBuffer(GL_TEXTURE_BUFFER, 0);
  glBindTexture(GL_TEXTURE_BUFFER, 0);
  CHECK_FOR_OGL_ERROR();

  // Create buffer for node parent indices
  // For each internal node (m_nElements - 1) + each leaf node (m_nElements) parent index
  glGenBuffers(1, &m_glNodeParentsBuf);
  glBindBuffer(GL_ARRAY_BUFFER, m_glNodeParentsBuf);
  glBufferData(GL_ARRAY_BUFFER, 2 * m_nElements * sizeof(cl_uint), NULL, GL_DYNAMIC_DRAW);
  CHECK_FOR_OGL_ERROR();

  // Create texture buffer
  glGenTextures(1, &m_glNodeParentsTB);
  glBindTexture(GL_TEXTURE_BUFFER, m_glNodeParentsTB);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, m_glNodeParentsTB);
  glBindBuffer(GL_TEXTURE_BUFFER, 0);
  glBindTexture(GL_TEXTURE_BUFFER, 0);
  CHECK_FOR_OGL_ERROR();



  // shader programs
  m_VSMesh = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
  m_PSMesh = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);

  if (!CreateShaderFromFile("mesh.vert", m_VSMesh)) return false;

  if (!CreateShaderFromFile("mesh.frag", m_PSMesh)) return false;

  m_ProgRenderMesh = glCreateProgramObjectARB();
  glAttachObjectARB(m_ProgRenderMesh, m_VSMesh);
  glAttachObjectARB(m_ProgRenderMesh, m_PSMesh);
  if (!LinkGLSLProgram(m_ProgRenderMesh)) return false;

  m_VSParticles = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
  m_PSParticles = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);

  if (!CreateShaderFromFile("particles.vert", m_VSParticles)) return false;
  if (!CreateShaderFromFile("particles.frag", m_PSParticles)) return false;

  m_ProgRenderParticles = glCreateProgramObjectARB();
  glAttachObjectARB(m_ProgRenderParticles, m_VSParticles);
  glAttachObjectARB(m_ProgRenderParticles, m_PSParticles);
  if (!LinkGLSLProgram(m_ProgRenderParticles)) return false;

  GLint tboSampler = glGetUniformLocationARB(m_ProgRenderParticles, "AABBmin");
  glUseProgramObjectARB(m_ProgRenderParticles);
  glUniform1i(tboSampler, 0);
  tboSampler = glGetUniformLocationARB(m_ProgRenderParticles, "AABBmax");
  glUseProgramObjectARB(m_ProgRenderParticles);
  glUniform1i(tboSampler, 1);
  CHECK_FOR_OGL_ERROR();

  aabbminmax = glGetUniformLocationARB(m_ProgRenderParticles, "aabbminmax");


  // set the modelview matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0f, 0.0f, m_TranslateZ);
  glRotatef(m_RotateY, 0.0f, 1.0f, 0.0f);
  glRotatef(m_RotateX, 1.0f, 0.0f, 0.0f);
  glTranslatef(-0.5, -0.5, -0.5);

  return true;
}

void CCreateBVH::ReleaseResources() {
  if (m_pMesh) {
    m_pMesh->ReleaseGLResources();
    delete m_pMesh;
    m_pMesh = NULL;
  }

  // Device resources

  for (auto& level : m_clScanLevels) SAFE_RELEASE_MEMOBJECT(level.first);

  SAFE_RELEASE_PROGRAM(m_MortonCodesProgram);
  SAFE_RELEASE_KERNEL(m_ReduceAABBminKernel);
  SAFE_RELEASE_KERNEL(m_ReduceAABBmaxKernel);

  SAFE_RELEASE_PROGRAM(m_PrepAABBsProgram);
  SAFE_RELEASE_KERNEL(m_AdvancePositionsKernel);
  SAFE_RELEASE_KERNEL(m_CreateLeafAABBsKernel);

  SAFE_RELEASE_KERNEL(m_SelectBitflagKernel);
  SAFE_RELEASE_KERNEL(m_ReorderKeysKernel);
  SAFE_RELEASE_KERNEL(m_PermutationIdentityKernel);
  SAFE_RELEASE_KERNEL(m_PermuteKernel);
  SAFE_RELEASE_PROGRAM(m_RadixSortProgram);

  SAFE_RELEASE_KERNEL(m_ScanKernel);
  SAFE_RELEASE_KERNEL(m_ScanAddKernel);
  SAFE_RELEASE_PROGRAM(m_ScanProgram);


  SAFE_RELEASE_GL_SHADER(m_PSParticles);
  SAFE_RELEASE_GL_SHADER(m_VSParticles);
  SAFE_RELEASE_GL_SHADER(m_VSMesh);
  SAFE_RELEASE_GL_SHADER(m_PSMesh);

  SAFE_RELEASE_GL_PROGRAM(m_ProgRenderParticles);
  SAFE_RELEASE_GL_PROGRAM(m_ProgRenderMesh);
}

void CCreateBVH::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]) {
  // We use the one specified in the constructor
  (void)LocalWorkSize;


  glFinish();
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clAABBLeaves[0], 0, NULL, NULL), "Error acquiring OpenGL buffer.");
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clAABBLeaves[1], 0, NULL, NULL), "Error acquiring OpenGL buffer.");

  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clNodeChildren, 0, NULL, NULL), "Error acquiring OpenGL buffer.");
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clNodeParents, 0, NULL, NULL), "Error acquiring OpenGL buffer.");

  // DO THE CL STUFF HERE

  // move the center points
  AdvancePositions(Context, CommandQueue, m_clPositions, m_clVelocities);

  // Creates leaf node AABBs, reduces them to morton AABB, calculates Morton Codes from the morton AABB
  MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
  // Sort morton codes and create permutation array
  RadixSort(Context, CommandQueue, m_clMortonCodes, m_clSortPermutation);
  // Permute the center positions and velocities using the permutation from radix sort
  Permute(Context, CommandQueue, &m_clPositions, m_clSortPermutation);
  Permute(Context, CommandQueue, &m_clVelocities, m_clSortPermutation);

  // Create leaf node AABBs again (their memory was used as temp memory during the morton AABB reduction)
  CreateLeafAABBs(Context, CommandQueue, m_clAABBLeaves, m_clPositions);



  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clAABBLeaves[0], 0, NULL, NULL), "Error releasing OpenGL buffer.");
  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clAABBLeaves[1], 0, NULL, NULL), "Error releasing OpenGL buffer.");

  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clNodeChildren, 0, NULL, NULL), "Error releasing OpenGL buffer.");
  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clNodeParents, 0, NULL, NULL), "Error releasing OpenGL buffer.");

  clFinish(CommandQueue);
}



unsigned int expandBits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}
unsigned int morton3D(float x, float y, float z) {
  x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
  y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
  z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
  unsigned int xx = expandBits((unsigned int)x);
  unsigned int yy = expandBits((unsigned int)y);
  unsigned int zz = expandBits((unsigned int)z);
  return xx * 4 + yy * 2 + zz;
}

void nodehierarchy(std::vector<cl_uint2>& children, std::vector<cl_uint>& parents, size_t i, size_t left, size_t right,
                   const std::vector<cl_uint>& mortoncodes) {
  cl_uint firstCode = mortoncodes[left];
  cl_uint lastCode = mortoncodes[right];

  int split = left;
  // Split range of identical codes in the middle
  if (firstCode == lastCode)
    split = (left + right) >> 1;
  else {
    // count leading zeros of XOR of the two codes, corresponds to number of leading bits that are not the same
    int referencePrefix = __builtin_clz(firstCode ^ lastCode);

    // Binary search for the highest code, that has a longer common prefix than the reference prefix
    // Start with the first element in the range and stride of the whole range
    int step = right - left;

    do {
      // decrease step size exponentially, and calculate the potential split position
      step = (step + 1) >> 1;
      int newSplit = split + step;

      if (newSplit < right) {
        cl_uint splitCode = mortoncodes[newSplit];
        int splitPrefix = __builtin_clz(firstCode ^ splitCode);
        // A larger prefix than referencePrefix means, that we still haven't advanced far enough towards the right
        if (splitPrefix > referencePrefix) split = newSplit;
      }
    } while (step > 1);
  }


  int leftChild = split;
  if (leftChild == left) leftChild += mortoncodes.size();

  int rightChild = split + 1;
  if (rightChild == right) rightChild += mortoncodes.size();

  // Set the child nodes in this internal node
  children[i].s[0] = leftChild;
  children[i].s[1] = rightChild;
  // Set the index of the parent in the two child nodes
  parents[leftChild] = i;
  parents[rightChild] = i;

  if (leftChild < mortoncodes.size()) nodehierarchy(children, parents, leftChild, left, leftChild, mortoncodes);
  if (rightChild < mortoncodes.size()) nodehierarchy(children, parents, rightChild, rightChild, right, mortoncodes);
}

void CCreateBVH::TestPerformance(cl_context Context, cl_command_queue CommandQueue) {
  auto print_info = [&](const string& name, double timecpu, double timegpu, bool correct) {
    cout << "\n############";
    for (auto c : name) cout << '#';
    cout << "\n ###  ";

    if (correct) {
#ifndef WIN32
      cout << "\033[1;32m";
#endif
      cout << name;
#ifndef WIN32
      cout << "\033[0m";
#endif
    } else {
#ifndef WIN32
      cout << "\033[1;31m";
#endif
      cout << name;
#ifndef WIN32
      cout << "\033[0m";
#endif
    }


    cout << "  ### \n############";
    for (auto c : name) cout << '#';
    cout << "\n\nExecution time (ms):" << endl
         << "\tCPU: " << timecpu << endl
         << "\tGPU: " << timegpu << endl;
  };

  cout << "Running performance test..." << endl
       << endl;
  glFinish();
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clAABBLeaves[0], 0, NULL, NULL), "Error acquiring OpenGL buffer.");
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clAABBLeaves[1], 0, NULL, NULL), "Error acquiring OpenGL buffer.");
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clNodeChildren, 0, NULL, NULL), "Error acquiring OpenGL buffer.");
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clNodeParents, 0, NULL, NULL), "Error acquiring OpenGL buffer.");

  CTimer timer;
  double timecpu, timegpu;
  bool res = true;

  std::vector<cl_float4> positions(m_nElements);
  std::vector<cl_float4> leafaabbmin(m_nElements);
  std::vector<cl_float4> leafaabbmin_cpu(m_nElements);
  std::vector<cl_float4> leafaabbmax(m_nElements);
  std::vector<cl_float4> leafaabbmax_cpu(m_nElements);
  std::vector<cl_float4> mortonaabb(2);
  std::vector<cl_float4> mortonaabb_cpu(2);

  std::vector<cl_uint> mortoncodes(m_nElements);
  std::vector<cl_uint> mortoncodes_cpu(m_nElements);

  std::vector<cl_uint> zeroflags(m_nElements);
  std::vector<cl_uint> oneflags(m_nElements);
  std::vector<cl_uint> zeroflags_cpu(m_nElements);
  std::vector<cl_uint> oneflags_cpu(m_nElements);
  std::vector<cl_uint> mortoncodessort(m_nElements);
  std::vector<cl_uint> mortoncodessort_cpu(m_nElements);
  std::vector<cl_uint> permutation(m_nElements);
  std::vector<cl_uint> permutation_cpu(m_nElements);
  std::vector<cl_float4> positionssort(m_nElements);
  std::vector<cl_float4> positionssort_cpu(m_nElements);

  std::vector<cl_uint2> childnodes(m_nElements);
  std::vector<cl_uint2> childnodes_cpu(m_nElements);
  std::vector<cl_uint> parentnodes(2 * m_nElements);
  std::vector<cl_uint> parentnodes_cpu(2 * m_nElements);

  // #############
  // CREATE LEAF NODE AABBs
  // #############
  if (res) {
    // Read input positions from gpu
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clPositions, CL_TRUE, 0, m_nElements * sizeof(cl_float4), positions.data(), 0, NULL, NULL),
                "Error reading data from device!");
    // Time CPU
    clFinish(CommandQueue);
    timer.Start();
    for (size_t i = 0; i < m_nElements; ++i) {
      leafaabbmin_cpu[i].s[0] = positions[i].s[0] - positions[i].s[3] / 2;
      leafaabbmin_cpu[i].s[1] = positions[i].s[1] - positions[i].s[3] / 2;
      leafaabbmin_cpu[i].s[2] = positions[i].s[2] - positions[i].s[3] / 2;
      leafaabbmin_cpu[i].s[3] = positions[i].s[3] - positions[i].s[3] / 2;
      leafaabbmax_cpu[i].s[0] = positions[i].s[0] + positions[i].s[3] / 2;
      leafaabbmax_cpu[i].s[1] = positions[i].s[1] + positions[i].s[3] / 2;
      leafaabbmax_cpu[i].s[2] = positions[i].s[2] + positions[i].s[3] / 2;
      leafaabbmax_cpu[i].s[3] = positions[i].s[3] + positions[i].s[3] / 2;
    }
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    // Time for GPU
    timer.Start();
    for (size_t i = 0; i < 100; ++i) CreateLeafAABBs(Context, CommandQueue, m_clAABBLeaves, m_clPositions);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;

    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clAABBLeaves[0], CL_TRUE, 0, m_nElements * sizeof(cl_float4), leafaabbmin.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clAABBLeaves[1], CL_TRUE, 0, m_nElements * sizeof(cl_float4), leafaabbmax.data(), 0, NULL, NULL),
                "Error reading data from device!");
    res = memcmp(leafaabbmin.data(), leafaabbmin_cpu.data(), leafaabbmin.size() * sizeof(cl_float4)) == 0 &&
          memcmp(leafaabbmax.data(), leafaabbmax_cpu.data(), leafaabbmax.size() * sizeof(cl_float4)) == 0;
    print_info("CREATE LEAF NODE AABB", timecpu, timegpu, res);
  }


  // #############
  // MORTON CODE AABB
  // #############
  if (res) {
    // input AABBs from gpu
    CreateLeafAABBs(Context, CommandQueue, m_clAABBLeaves, m_clPositions);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clAABBLeaves[0], CL_TRUE, 0, m_nElements * sizeof(cl_float4), leafaabbmin.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clAABBLeaves[1], CL_TRUE, 0, m_nElements * sizeof(cl_float4), leafaabbmax.data(), 0, NULL, NULL),
                "Error reading data from device!");
    // Time CPU
    timer.Start();
    mortonaabb_cpu[0].s[0] = leafaabbmin[0].s[0];
    mortonaabb_cpu[0].s[1] = leafaabbmin[0].s[1];
    mortonaabb_cpu[0].s[2] = leafaabbmin[0].s[2];
    mortonaabb_cpu[0].s[3] = leafaabbmin[0].s[3];
    mortonaabb_cpu[1].s[0] = leafaabbmax[0].s[0];
    mortonaabb_cpu[1].s[1] = leafaabbmax[0].s[1];
    mortonaabb_cpu[1].s[2] = leafaabbmax[0].s[2];
    mortonaabb_cpu[1].s[3] = leafaabbmax[0].s[3];
    for (size_t i = 1; i < m_nElements; ++i) {
      mortonaabb_cpu[0].s[0] = std::min(mortonaabb_cpu[0].s[0], leafaabbmin[i].s[0]);
      mortonaabb_cpu[0].s[1] = std::min(mortonaabb_cpu[0].s[1], leafaabbmin[i].s[1]);
      mortonaabb_cpu[0].s[2] = std::min(mortonaabb_cpu[0].s[2], leafaabbmin[i].s[2]);
      mortonaabb_cpu[0].s[3] = std::min(mortonaabb_cpu[0].s[3], leafaabbmin[i].s[3]);

      mortonaabb_cpu[1].s[0] = std::max(mortonaabb_cpu[1].s[0], leafaabbmax[i].s[0]);
      mortonaabb_cpu[1].s[1] = std::max(mortonaabb_cpu[1].s[1], leafaabbmax[i].s[1]);
      mortonaabb_cpu[1].s[2] = std::max(mortonaabb_cpu[1].s[2], leafaabbmax[i].s[2]);
      mortonaabb_cpu[1].s[3] = std::max(mortonaabb_cpu[1].s[3], leafaabbmax[i].s[3]);
    }
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    // Time for GPU
    timer.Start();
    for (size_t i = 0; i < 100; ++i) MortonCodeAABB(Context, CommandQueue, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;

    // Verify results
    MortonCodeAABB(Context, CommandQueue, m_clMortonAABB, m_clAABBLeaves, m_clPositions);

    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonAABB, CL_TRUE, 0, sizeof(cl_float4) * 2, mortonaabb.data(), 0, NULL, NULL),
                "Error reading data from device!");

    res = memcmp(mortonaabb.data(), mortonaabb_cpu.data(), mortonaabb.size() * sizeof(cl_float4)) == 0;
    print_info("MORTON CODE AABB", timecpu, timegpu, res);
  }


  // #############
  // MORTON CODES
  // #############
  if (res) {
    // Input morton AABB from gpu
    CreateLeafAABBs(Context, CommandQueue, m_clAABBLeaves, m_clPositions);
    MortonCodeAABB(Context, CommandQueue, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonAABB, CL_TRUE, 0, sizeof(cl_float4) * 2, mortonaabb.data(), 0, NULL, NULL),
                "Error reading data from device!");

    // Time CPU
    timer.Start();
    cl_float4 size;
    size.s[0] = mortonaabb[1].s[0] - mortonaabb[0].s[0];
    size.s[1] = mortonaabb[1].s[1] - mortonaabb[0].s[1];
    size.s[2] = mortonaabb[1].s[2] - mortonaabb[0].s[2];
    for (size_t i = 0; i < m_nElements; ++i) {
      cl_float4 p = positions[i];
      p.s[0] = (p.s[0] - mortonaabb[0].s[0]) / size.s[0];
      p.s[1] = (p.s[1] - mortonaabb[0].s[1]) / size.s[1];
      p.s[2] = (p.s[2] - mortonaabb[0].s[2]) / size.s[2];

      mortoncodes_cpu[i] = morton3D(p.s[0], p.s[1], p.s[2]);
    }
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    // Time for GPU
    timer.Start();
    for (size_t i = 0; i < 100; ++i) MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;



    // Verify results
    size.s[0] = mortonaabb[1].s[0] - mortonaabb[0].s[0];
    size.s[1] = mortonaabb[1].s[1] - mortonaabb[0].s[1];
    size.s[2] = mortonaabb[1].s[2] - mortonaabb[0].s[2];
    size.s[3] = mortonaabb[1].s[3] - mortonaabb[0].s[3];
    for (size_t i = 0; i < m_nElements; ++i) {
      cl_float4 p = positions[i];
      p.s[0] = (p.s[0] - mortonaabb[0].s[0]) / size.s[0];
      p.s[1] = (p.s[1] - mortonaabb[0].s[1]) / size.s[1];
      p.s[2] = (p.s[2] - mortonaabb[0].s[2]) / size.s[2];

      if (p.s[0] < 0 || p.s[0] > 1 || p.s[1] < 0 || p.s[1] > 1 || p.s[2] < 0 || p.s[2] > 1) {
        cout << "PANIC" << endl;
        return;
      }
    }


    // Verify results
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);

    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonCodes, CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodes.data(), 0, NULL, NULL),
                "Error reading data from device!");

    res = memcmp(mortoncodes.data(), mortoncodes_cpu.data(), mortoncodes.size() * sizeof(cl_uint)) == 0;
    print_info("MORTON CODES", timecpu, timegpu, res);

    size_t cnt = 0;
    for (size_t i = 0; i < m_nElements; ++i) {
      if (mortoncodes[i] != mortoncodes_cpu[i]) {
        cout << i << "             " << mortoncodes[i] << "    " << mortoncodes_cpu[i] << "   >>>>>>> ERROR" << endl;
        ++cnt;
      }
    }
    res = cnt < 4;
  }


  // #############
  // PERMUTATION IDENTITY
  // #############
  if (res) {
    // Time for CPU
    timer.Start();
    for (size_t i = 0; i < m_nElements; ++i) permutation_cpu[i] = i;
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    // Time for GPU
    timer.Start();
    for (size_t i = 0; i < 100; ++i) PermutationIdentity(Context, CommandQueue, m_clSortPermutation);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;

    // Validate correctness
    PermutationIdentity(Context, CommandQueue, m_clSortPermutation);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clSortPermutation, CL_TRUE, 0, m_nElements * sizeof(cl_uint), permutation.data(), 0, NULL, NULL),
                "Error reading data from device!");

    res = memcmp(permutation.data(), permutation_cpu.data(), permutation.size() * sizeof(cl_uint)) == 0;
    print_info("PERMUTATION IDENTITY", timecpu, timegpu, res);
  }


  // #############
  // RADIX FLAGS
  // #############
  if (res) {
    // Input morton codes from gpu
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonCodes, CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodes.data(), 0, NULL, NULL),
                "Error reading data from device!");

    // Time for CPU
    timer.Start();
    for (size_t i = 0; i < m_nElements; ++i) {
      cl_uint rzero = 1;
      cl_uint rone = 0;
      if ((mortoncodes[i] & 0x0001) == 0x0001) {
        rzero = 0;
        rone = 1;
      }
      zeroflags_cpu[i] = rzero;
      oneflags_cpu[i] = rone;
    }
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    // Time for GPU
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);

    timer.Start();
    for (size_t i = 0; i < 100; ++i) SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes, 0x0001);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;

    // Validate correctness
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes, 0x0001);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixZeroBit, CL_TRUE, 0, m_nElements * sizeof(cl_uint), zeroflags.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixOneBit, CL_TRUE, 0, m_nElements * sizeof(cl_uint), oneflags.data(), 0, NULL, NULL),
                "Error reading data from device!");

    res = memcmp(zeroflags.data(), zeroflags_cpu.data(), zeroflags.size() * sizeof(cl_uint)) == 0 &&
          memcmp(oneflags.data(), oneflags_cpu.data(), oneflags.size() * sizeof(cl_uint)) == 0;
    print_info("SELECT BIT", timecpu, timegpu, res);

    // for (size_t i = 0; i < 16; ++i) {
    //  cout << std::hex << setw(4) << initkeys[i] << "   " << std::hex << setw(4) << result1_cpu[i] << "   " << std::hex << setw(4) << result1[i] << endl;
    //}
  }


  // #############
  // SCAN
  // #############
  if (res) {
    // Input flags from gpu
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes, 0x0001);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixZeroBit, CL_TRUE, 0, m_nElements * sizeof(cl_uint), zeroflags_cpu.data(), 0, NULL, NULL),
                "Error reading data from device!");

    // Time for CPU
    timer.Start();
    size_t sum = 0;
    for (size_t i = 0; i < m_nElements; ++i) {
      cl_uint tmp = zeroflags_cpu[i];
      zeroflags_cpu[i] = sum;
      sum += tmp;
    }
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();


    // Time for GPU
    // Initialize buffers for scan with result from SelectBitflag
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes, 0x0001);

    timer.Start();
    for (size_t i = 0; i < 100; ++i) Scan(Context, CommandQueue, m_clRadixZeroBit);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;


    // Validate correctness
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes, 0x0001);
    Scan(Context, CommandQueue, m_clRadixZeroBit);
    Scan(Context, CommandQueue, m_clRadixOneBit);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixZeroBit, CL_TRUE, 0, m_nElements * sizeof(cl_uint), zeroflags.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixOneBit, CL_TRUE, 0, m_nElements * sizeof(cl_uint), oneflags.data(), 0, NULL, NULL),
                "Error reading data from device!");

    // Assert the prefix sum is correct
    res = memcmp(zeroflags.data(), zeroflags_cpu.data(), zeroflags.size() * sizeof(cl_uint)) == 0;
    // Assert the two perfix sums index the new permutation from 0 to m_nElements-1
    std::vector<bool> seen(m_nElements, false);
    for (size_t i = 0; i < m_nElements; ++i) {
      cl_uint idx;
      if ((mortoncodes[i] & 0x0001) == 0x0001) {
        idx = zeroflags[m_nElements - 1] + oneflags[i];
        if ((mortoncodes[m_nElements - 1] & 0x0001) == 0) ++idx;
      } else {
        idx = zeroflags[i];
      }

      res = (idx >= 0 && idx < m_nElements && !seen[idx]);
      seen[idx] = true;
      if (!res) break;
    }

    print_info("SCAN", timecpu, timegpu, res);
  }


  // #############
  // REORDER
  // #############
  if (res) {
    // Input bit flag prefix sums from gpu
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes, 0x0001);
    Scan(Context, CommandQueue, m_clRadixZeroBit);
    Scan(Context, CommandQueue, m_clRadixOneBit);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonCodes, CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodes.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clSortPermutation, CL_TRUE, 0, m_nElements * sizeof(cl_uint), permutation.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixZeroBit, CL_TRUE, 0, m_nElements * sizeof(cl_uint), zeroflags.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixOneBit, CL_TRUE, 0, m_nElements * sizeof(cl_uint), oneflags.data(), 0, NULL, NULL),
                "Error reading data from device!");

    // Time for CPU
    // zeroflags and oneflags contains the correct result, that has been calculated in test for SCAN from the GPU
    // permutation contains the correct result, that has been calculated in test for PERMUTATION IDENTITY from the GPU
    timer.Start();
    for (size_t i = 0; i < m_nElements; ++i) {
      cl_uint idx;
      if ((mortoncodes[i] & 0x0001) == 0x0001) {
        idx = zeroflags[m_nElements - 1] + oneflags[i];
        if ((mortoncodes[m_nElements - 1] & 0x0001) == 0) ++idx;
      } else {
        idx = zeroflags[i];
      }

      mortoncodessort_cpu[idx] = mortoncodes[i];
      permutation_cpu[idx] = permutation[i];
    }
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();


    // Time for GPU
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes, 0x0001);
    Scan(Context, CommandQueue, m_clRadixZeroBit);
    Scan(Context, CommandQueue, m_clRadixOneBit);
    timer.Start();
    for (size_t i = 0; i < 100; ++i)
      ReorderKeys(Context, CommandQueue, m_clRadixKeysPong, m_clRadixPermutationPong, m_clMortonCodes, m_clSortPermutation, m_clRadixZeroBit, m_clRadixOneBit,
                  0x0001);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;


    // Validate correctness
    PermutationIdentity(Context, CommandQueue, m_clSortPermutation);
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes, 0x0001);
    Scan(Context, CommandQueue, m_clRadixZeroBit);
    Scan(Context, CommandQueue, m_clRadixOneBit);
    ReorderKeys(Context, CommandQueue, m_clRadixKeysPong, m_clRadixPermutationPong, m_clMortonCodes, m_clSortPermutation, m_clRadixZeroBit, m_clRadixOneBit,
                0x0001);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixKeysPong, CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodessort.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clRadixPermutationPong, CL_TRUE, 0, m_nElements * sizeof(cl_uint), permutation.data(), 0, NULL, NULL),
                "Error reading data from device!");


    // ASSERT that morton codes and permutation are correct,
    res = memcmp(mortoncodessort.data(), mortoncodessort_cpu.data(), mortoncodessort.size() * sizeof(cl_uint)) == 0 &&
          memcmp(permutation.data(), permutation_cpu.data(), permutation.size() * sizeof(cl_uint)) == 0;
    // ASSERT that the permutation permutes to the actual reorderd morton codes.
    for (size_t i = 0; i < m_nElements; ++i) {
      res = mortoncodessort[i] == mortoncodes[permutation[i]];
      if (!res) break;
    }

    print_info("REORDER", timecpu, timegpu, res);
  }


  // #############
  // RADIX SORT
  // #############
  if (res) {
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonCodes, CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodessort_cpu.data(), 0, NULL, NULL),
                "Error reading data from device!");
    timer.Start();
    sort(mortoncodessort_cpu.begin(), mortoncodessort_cpu.end());
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    timer.Start();
    RadixSort(Context, CommandQueue, m_clMortonCodes, m_clSortPermutation);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds();


    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonCodes, CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodessort.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clSortPermutation, CL_TRUE, 0, m_nElements * sizeof(cl_uint), permutation.data(), 0, NULL, NULL),
                "Error reading data from device!");

    // ASSERT that morton codes and permutation are correct,
    res = memcmp(mortoncodessort.data(), mortoncodessort_cpu.data(), mortoncodessort.size() * sizeof(cl_uint)) == 0;
    // ASSERT that the permutation permutes to the actual reorderd morton codes.
    for (size_t i = 0; i < m_nElements; ++i) {
      res = mortoncodessort[i] == mortoncodes[permutation[i]];
      if (!res) break;
    }
    std::set<cl_uint> ids;
    for (size_t i = 0; i < m_nElements; ++i) {
      res = 0 <= permutation[i] && permutation[i] < m_nElements && ids.find(permutation[i]) == ids.end();
      ids.insert(permutation[i]);
      if (!res) break;
    }

    print_info("RADIX SORT", timecpu, timegpu, res);
  }


  // #############
  // PERMUTE
  // #############
  if (res) {
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    RadixSort(Context, CommandQueue, m_clMortonCodes, m_clSortPermutation);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clPositions, CL_TRUE, 0, m_nElements * sizeof(cl_float4), positions.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clSortPermutation, CL_TRUE, 0, m_nElements * sizeof(cl_uint), permutation.data(), 0, NULL, NULL),
                "Error reading data from device!");

    timer.Start();
    for (size_t i = 0; i < m_nElements; ++i) positionssort_cpu[i] = positions[permutation[i]];

    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    timer.Start();
    for (size_t i = 0; i < 100; ++i) Permute(Context, CommandQueue, &m_clPositions, m_clSortPermutation);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;


    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    RadixSort(Context, CommandQueue, m_clMortonCodes, m_clSortPermutation);
    Permute(Context, CommandQueue, &m_clPositions, m_clSortPermutation);
    Permute(Context, CommandQueue, &m_clVelocities, m_clSortPermutation);

    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clPositions, CL_TRUE, 0, m_nElements * sizeof(cl_float4), positionssort.data(), 0, NULL, NULL),
                "Error reading data from device!");


    // ASSERT that morton codes and permutation are correct,
    res = memcmp(positionssort.data(), positionssort_cpu.data(), positionssort.size() * sizeof(cl_float4)) == 0;
    print_info("PERMUTE", timecpu, timegpu, res);
  }


  // #############
  // PERMUTE
  // #############
  if (res) {
    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    RadixSort(Context, CommandQueue, m_clMortonCodes, m_clSortPermutation);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonCodes, CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodes.data(), 0, NULL, NULL),
                "Error reading data from device!");

    timer.Start();
    nodehierarchy(childnodes_cpu, parentnodes_cpu, 0, 0, m_nElements - 1, mortoncodes);
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    timer.Start();
    for (size_t i = 0; i < 100; ++i) CreateHierarchy(Context, CommandQueue, m_clNodeChildren, m_clNodeParents, m_clMortonCodes);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;


    MortonCodes(Context, CommandQueue, m_clMortonCodes, m_clMortonAABB, m_clAABBLeaves, m_clPositions);
    RadixSort(Context, CommandQueue, m_clMortonCodes, m_clSortPermutation);

    CreateHierarchy(Context, CommandQueue, m_clNodeChildren, m_clNodeParents, m_clMortonCodes);

    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clNodeChildren, CL_TRUE, 0, m_nElements * sizeof(cl_uint2), childnodes.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clNodeParents, CL_TRUE, 0, 2 * m_nElements * sizeof(cl_uint), parentnodes.data(), 0, NULL, NULL),
                "Error reading data from device!");

    // children[m_nElements- 1] and parents[m_nElements - 1] are undefined, because this node does not exist
    // parents[0] is undefined, because the root has no parent node
    res = memcmp(childnodes.data(), childnodes_cpu.data(), (childnodes.size() - 1) * sizeof(cl_uint2)) == 0 &&
          memcmp(&parentnodes[1], &parentnodes_cpu[1], (m_nElements - 2) * sizeof(cl_uint)) == 0 &&
          memcmp(&parentnodes[m_nElements], &parentnodes_cpu[m_nElements], m_nElements * sizeof(cl_uint)) == 0;
    print_info("NODE HIERARCHY", timecpu, timegpu, res);
  }

  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clAABBLeaves[0], 0, NULL, NULL), "Error releasing OpenGL buffer.");
  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clAABBLeaves[1], 0, NULL, NULL), "Error releasing OpenGL buffer.");
  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clNodeChildren, 0, NULL, NULL), "Error releasing OpenGL buffer.");
  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clNodeParents, 0, NULL, NULL), "Error releasing OpenGL buffer.");

  clFinish(CommandQueue);
}


void CCreateBVH::AdvancePositions(cl_context Context, cl_command_queue CommandQueue, cl_mem positions, cl_mem velocities) {
  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_nElements, m_ScanLocalWorkSize[0]);

  clErr = clSetKernelArg(m_AdvancePositionsKernel, 0, sizeof(cl_mem), (void*)&positions);
  clErr |= clSetKernelArg(m_AdvancePositionsKernel, 1, sizeof(cl_mem), (void*)&velocities);
  clErr |= clSetKernelArg(m_AdvancePositionsKernel, 2, sizeof(cl_uint), (void*)&m_nElements);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_AdvancePositionsKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");
}

void CCreateBVH::CreateLeafAABBs(cl_context Context, cl_command_queue CommandQueue, cl_mem aabbs[2], cl_mem positions) {
  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_nElements, m_ScanLocalWorkSize[0]);

  clErr = clSetKernelArg(m_CreateLeafAABBsKernel, 0, sizeof(cl_mem), (void*)&positions);
  clErr |= clSetKernelArg(m_CreateLeafAABBsKernel, 1, sizeof(cl_mem), (void*)&aabbs[0]);
  clErr |= clSetKernelArg(m_CreateLeafAABBsKernel, 2, sizeof(cl_mem), (void*)&aabbs[1]);
  clErr |= clSetKernelArg(m_CreateLeafAABBsKernel, 3, sizeof(cl_uint), (void*)&m_nElements);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_CreateLeafAABBsKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");
}
void CCreateBVH::ReduceAABB(cl_context Context, cl_command_queue CommandQueue, cl_mem aabbs, cl_kernel Kernel) {
  cl_int clErr;
  size_t globalWorkSize;

  size_t N = m_nElements;
  while (N >= 2) {
    // The number of threads is half the number of elements in the array, that need to be reduced in this iteration
    // This is exactly the same offset to the right summand.
    unsigned int stride = N / 2 + N % 2;
    globalWorkSize = CLUtil::GetGlobalWorkSize(stride, m_ScanLocalWorkSize[0]);

    clErr = clSetKernelArg(Kernel, 0, sizeof(cl_mem), (void*)&aabbs);
    clErr |= clSetKernelArg(Kernel, 1, sizeof(cl_uint), (void*)&m_nElements);
    clErr |= clSetKernelArg(Kernel, 2, sizeof(cl_uint), (void*)&stride);
    V_RETURN_CL(clErr, "Error setting kernel arguments.");

    clErr = clEnqueueNDRangeKernel(CommandQueue, Kernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
    V_RETURN_CL(clErr, "Error when enqueuing kernel.");

    // In this iteration stride elements have been written, so the number of to be reduces elements in the next iteration is stride.
    N = stride;
  }
}
void CCreateBVH::MortonCodeAABB(cl_context Context, cl_command_queue CommandQueue, cl_mem mortonaabb, cl_mem aabbs[2], cl_mem positions) {
  CreateLeafAABBs(Context, CommandQueue, m_clAABBLeaves, positions);
  ReduceAABB(Context, CommandQueue, aabbs[0], m_ReduceAABBminKernel);
  ReduceAABB(Context, CommandQueue, aabbs[1], m_ReduceAABBmaxKernel);


  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = 2;
  size_t localWorkSize = 2;

  clErr = clSetKernelArg(m_MortonCodeAABBKernel, 0, sizeof(cl_mem), (void*)&mortonaabb);
  clErr = clSetKernelArg(m_MortonCodeAABBKernel, 1, sizeof(cl_mem), (void*)&aabbs[0]);
  clErr = clSetKernelArg(m_MortonCodeAABBKernel, 2, sizeof(cl_mem), (void*)&aabbs[1]);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_MortonCodeAABBKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");
}
void CCreateBVH::MortonCodes(cl_context Context, cl_command_queue CommandQueue, cl_mem codes, cl_mem mortonaabb, cl_mem aabbs[2], cl_mem positions) {
  MortonCodeAABB(Context, CommandQueue, mortonaabb, aabbs, positions);

  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_nElements, m_ScanLocalWorkSize[0]);

  clErr = clSetKernelArg(m_MortonCodesKernel, 0, sizeof(cl_mem), (void*)&codes);
  clErr = clSetKernelArg(m_MortonCodesKernel, 1, sizeof(cl_mem), (void*)&positions);
  clErr = clSetKernelArg(m_MortonCodesKernel, 2, sizeof(cl_mem), (void*)&mortonaabb);
  clErr = clSetKernelArg(m_MortonCodesKernel, 3, sizeof(cl_uint), (void*)&m_nElements);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_MortonCodesKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");
}

void CCreateBVH::PermutationIdentity(cl_context Context, cl_command_queue CommandQueue, cl_mem permutation) {
  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_nElements, m_ScanLocalWorkSize[0]);

  clErr = clSetKernelArg(m_PermutationIdentityKernel, 0, sizeof(cl_mem), (void*)&permutation);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_PermutationIdentityKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");
}
void CCreateBVH::SelectBitflag(cl_context Context, cl_command_queue CommandQueue, cl_mem flagnotset, cl_mem flagset, cl_mem keys, cl_uint mask) {
  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_nElements, m_ScanLocalWorkSize[0]);

  clErr = clSetKernelArg(m_SelectBitflagKernel, 0, sizeof(cl_mem), (void*)&flagnotset);
  clErr |= clSetKernelArg(m_SelectBitflagKernel, 1, sizeof(cl_mem), (void*)&flagset);
  clErr |= clSetKernelArg(m_SelectBitflagKernel, 2, sizeof(cl_mem), (void*)&keys);
  clErr |= clSetKernelArg(m_SelectBitflagKernel, 3, sizeof(cl_uint), (void*)&mask);
  clErr |= clSetKernelArg(m_SelectBitflagKernel, 4, sizeof(cl_uint), (void*)&m_nElements);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_SelectBitflagKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");
}
void CCreateBVH::Scan(cl_context Context, cl_command_queue CommandQueue, cl_mem inoutbuffer) {
  cl_int clErr;
  size_t globalWorkSize;
  size_t localMemSize = sizeof(cl_uint) * (m_ScanLocalWorkSize[0] * 2 + m_ScanLocalWorkSize[0] / NUM_BANKS);

  // Set the input as level zero
  m_clScanLevels[0].first = inoutbuffer;

  // Loop to compute the local PPS
  for (size_t level = 0; level < m_clScanLevels.size() - 1; ++level) {
    // We only need half the number of work items than elements in the array, because we load 2 elements from the start.
    globalWorkSize = CLUtil::GetGlobalWorkSize(m_clScanLevels[level].second / 2, m_ScanLocalWorkSize[0]);

    // level is the array read from, and storing the local PPS results
    // level+1 is the array storing the reduction result of each local PPS
    clErr = clSetKernelArg(m_ScanKernel, 0, sizeof(cl_mem), (void*)&m_clScanLevels[level].first);
    clErr |= clSetKernelArg(m_ScanKernel, 1, sizeof(cl_mem), (void*)&m_clScanLevels[level + 1].first);
    clErr |= clSetKernelArg(m_ScanKernel, 2, localMemSize, NULL);
    V_RETURN_CL(clErr, "Error setting scan kernel arguments.");

    clErr = clEnqueueNDRangeKernel(CommandQueue, m_ScanKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
    V_RETURN_CL(clErr, "Error when enqueuing scan kernel.");
  }

  // Loop to add the higher order PPS to the lower order ones
  for (size_t level = m_clScanLevels.size() - 2; level >= 1; --level) {
    // We need as many work items as elements (exept for the first block)
    globalWorkSize = CLUtil::GetGlobalWorkSize(m_clScanLevels[level - 1].second - m_ScanLocalWorkSize[0] * 2, m_ScanLocalWorkSize[0]);

    clErr = clSetKernelArg(m_ScanAddKernel, 0, sizeof(cl_mem), (void*)&m_clScanLevels[level].first);
    clErr |= clSetKernelArg(m_ScanAddKernel, 1, sizeof(cl_mem), (void*)&m_clScanLevels[level - 1].first);
    V_RETURN_CL(clErr, "Error setting kernel arguments.");

    clErr = clEnqueueNDRangeKernel(CommandQueue, m_ScanAddKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
    V_RETURN_CL(clErr, "Error when enqueuing kernel.");
  }

  // Results are in the input buffer again, clean up so that no weird thing happen...
  m_clScanLevels[0].first = nullptr;
}
void CCreateBVH::ReorderKeys(cl_context Context, cl_command_queue CommandQueue, cl_mem keysout, cl_mem permutationout, cl_mem keysin, cl_mem permutationin,
                             cl_mem indexzerobits, cl_mem indexonebits, cl_uint mask) {
  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_nElements, m_ScanLocalWorkSize[0]);

  clErr = clSetKernelArg(m_ReorderKeysKernel, 0, sizeof(cl_mem), (void*)&keysout);
  clErr |= clSetKernelArg(m_ReorderKeysKernel, 1, sizeof(cl_mem), (void*)&permutationout);
  clErr |= clSetKernelArg(m_ReorderKeysKernel, 2, sizeof(cl_mem), (void*)&keysin);
  clErr |= clSetKernelArg(m_ReorderKeysKernel, 3, sizeof(cl_mem), (void*)&permutationin);
  clErr |= clSetKernelArg(m_ReorderKeysKernel, 4, sizeof(cl_mem), (void*)&indexzerobits);
  clErr |= clSetKernelArg(m_ReorderKeysKernel, 5, sizeof(cl_mem), (void*)&indexonebits);
  clErr |= clSetKernelArg(m_ReorderKeysKernel, 6, sizeof(cl_uint), (void*)&mask);
  clErr |= clSetKernelArg(m_ReorderKeysKernel, 7, sizeof(cl_uint), (void*)&m_nElements);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_ReorderKeysKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");
}
void CCreateBVH::RadixSort(cl_context Context, cl_command_queue CommandQueue, cl_mem keys, cl_mem permutation) {
  PermutationIdentity(Context, CommandQueue, permutation);

  cl_mem keyspingpong[2] = {keys, m_clRadixKeysPong};
  cl_mem permutationpingpong[2] = {permutation, m_clRadixPermutationPong};

  // Initialize the first source permutation to the ascending numbers from 0 to m_nElements
  PermutationIdentity(Context, CommandQueue, permutationpingpong[0]);

  // For each bit stable sort
  for (size_t i = 0; i < 32; ++i) {
    // Start with least important bit
    cl_uint mask = 1 << i;
    // Indices for source and destination of ping-pong buffers.
    int src = i % 2;
    int dst = (i + 1) % 2;

    // Extract the flags for the current bit.
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, keyspingpong[src], mask);
    // Scan on each buffer of flags
    Scan(Context, CommandQueue, m_clRadixZeroBit);
    Scan(Context, CommandQueue, m_clRadixOneBit);
    // Reorder
    ReorderKeys(Context, CommandQueue, keyspingpong[dst], permutationpingpong[dst], keyspingpong[src], permutationpingpong[src], m_clRadixZeroBit,
                m_clRadixOneBit, mask);
  }
}
void CCreateBVH::Permute(cl_context Context, cl_command_queue CommandQueue, cl_mem* target, cl_mem permutation) {
  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_nElements, m_ScanLocalWorkSize[0]);

  clErr = clSetKernelArg(m_PermuteKernel, 0, sizeof(cl_mem), (void*)&m_clPermuteTemp);
  clErr |= clSetKernelArg(m_PermuteKernel, 1, sizeof(cl_mem), (void*)target);
  clErr |= clSetKernelArg(m_PermuteKernel, 2, sizeof(cl_mem), (void*)&permutation);
  clErr |= clSetKernelArg(m_PermuteKernel, 3, sizeof(cl_uint), (void*)&m_nElements);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_PermuteKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");

  swap(m_clPermuteTemp, *target);
}

void CCreateBVH::CreateHierarchy(cl_context Context, cl_command_queue CommandQueue, cl_mem children, cl_mem parents, cl_mem mortoncodes) {
  cl_int clErr;
  // We need as many work items as elements
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_nElements, m_ScanLocalWorkSize[0]);

  clErr = clSetKernelArg( m_NodesHierarchyKernel, 0, sizeof(cl_mem), (void*)&children);
  clErr |= clSetKernelArg(m_NodesHierarchyKernel, 1, sizeof(cl_mem), (void*)&parents);
  clErr |= clSetKernelArg(m_NodesHierarchyKernel, 2, sizeof(cl_mem), (void*)&mortoncodes);
  clErr |= clSetKernelArg(m_NodesHierarchyKernel, 3, sizeof(cl_uint), (void*)&m_nElements);
  V_RETURN_CL(clErr, "Error setting kernel arguments.");

  clErr = clEnqueueNDRangeKernel(CommandQueue, m_NodesHierarchyKernel, 1, NULL, &globalWorkSize, m_ScanLocalWorkSize, 0, NULL, NULL);
  V_RETURN_CL(clErr, "Error when enqueuing kernel.");
}

void CCreateBVH::Render() {
  glCullFace(GL_BACK);

  // enable depth test & depth write
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);

  glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT, GL_FILL);
  glPolygonMode(GL_BACK, GL_LINE);
  glUseProgramObjectARB(m_ProgRenderMesh);
  glColor4f(0, 0.5f, 0.2f, 1.0f);
  if (m_pMesh) m_pMesh->DrawGL(GL_TRIANGLES);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  // enable depth test but disable depth write
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);

  glUseProgramObjectARB(m_ProgRenderParticles);



  //
  // Set up the OpenGL state machine for using point sprites...
  //
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);


  // This is how will our point sprite's size will be modified by
  // distance from the viewer
  float quadratic[] = {1.0f, 0.0f, 0.01f};
  glPointParameterfvARB(GL_POINT_DISTANCE_ATTENUATION_ARB, quadratic);

  glPointSize(10.0f);

  // The alpha of a point is calculated to allow the fading of points
  // instead of shrinking them past a defined threshold size. The threshold
  // is defined by GL_POINT_FADE_THRESHOLD_SIZE_ARB and is not clamped to
  // the minimum and maximum point sizes.

  // Specify point sprite texture coordinate replacement mode for each
  // texture unit
  glTexEnvf(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);

  //
  // Render point sprites...
  //


  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_BUFFER, m_glAABBLeafTB[0]);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_BUFFER, m_glAABBLeafTB[1]);

  glUniform1i(aabbminmax, 1);
  glDrawArraysInstanced(GL_LINES, 0, 24, m_nElements);

  glBindTexture(GL_TEXTURE_BUFFER, 0);
}

void CCreateBVH::OnKeyboard(int Key, int Action) {
  if (Key >= 255) return;
  if (Action == GLFW_PRESS) {
    m_KeyboardMask[Key] = true;

    if (Key == GLFW_KEY_F) cout << "F" << endl;
  } else {
    m_KeyboardMask[Key] = false;
  }
}

void CCreateBVH::OnMouse(int Button, int State) {
  if (State == GLFW_PRESS) {
    m_Buttons |= 1 << Button;
  } else if (State == GLFW_RELEASE) {
    m_Buttons = 0;
  }
}

void CCreateBVH::OnMouseMove(int X, int Y) {
  int dx, dy;
  dx = X - m_PrevX;
  dy = Y - m_PrevY;

  // left button
  if (m_Buttons & 1) {
    m_RotateX += dy * 0.2f;
    m_RotateY += dx * 0.2f;
  }
  m_PrevX = X;
  m_PrevY = Y;

  // set view matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, m_TranslateZ);
  glRotatef(m_RotateY, 0.0, 1.0, 0.0);
  glRotatef(m_RotateX, 1.0, 0.0, 0.0);
  glTranslatef(-0.5, -0.5, -0.5);
}

void CCreateBVH::OnIdle(double, float ElapsedTime) {
  // move camera?
  if (m_KeyboardMask[GLFW_KEY_W]) m_TranslateZ += 10.f * ElapsedTime;
  if (m_KeyboardMask[GLFW_KEY_S]) m_TranslateZ -= 10.f * ElapsedTime;

  if (m_TranslateZ > 0) m_TranslateZ = 0;

  // set the modelview matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, m_TranslateZ);
  glRotatef(m_RotateY, 0.0, 1.0, 0.0);
  glRotatef(m_RotateX, 1.0, 0.0, 0.0);
  glTranslatef(-0.5, -0.5, -0.5);
}

void CCreateBVH::OnWindowResized(int Width, int Height) {
  // viewport
  glViewport(0, 0, Width, Height);

  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)Width / (GLfloat)Height, 0.1, 1000.0);
}


///////////////////////////////////////////////////////////////////////////////
