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

#include <string>
#include <algorithm>
#include <string.h>
#include <iomanip>


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
    : m_nElements(NElems), m_nParticles(NElems), m_CollisionMeshPath(CollisionMeshPath) {
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


  cl_int clError;
  string programCode;


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
  m_clMortonCodes[0] = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  m_clMortonCodes[1] = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  m_clSortPermutation[0] = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  m_clSortPermutation[1] = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  // Radix bit buffers
  // Buffers that holds the flags where the bit of the current radix is zero/one
  m_clRadixZeroBit = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);
  m_clRadixOneBit = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_clScanLevels[0].second, NULL, &clError);


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



  //
  //
  //
  //
  //
  //
  //
  //
  //
  //

  cl_int clError2;

  // Particle arrrays
  m_clPosLife[0] = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE, m_glPosLife[0], &clError2);
  clError = clError2;
  m_clPosLife[1] = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE, m_glPosLife[1], &clError2);
  clError |= clError2;



  // Scan kernels
  CLUtil::LoadProgramSourceToMemory("Scan.cl", programCode);
  m_ScanProgram = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
  if (!m_ScanProgram) return false;

  m_ScanKernel = clCreateKernel(m_ScanProgram, "Scan", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create Scan kernel.");
  m_ScanAddKernel = clCreateKernel(m_ScanProgram, "ScanAdd", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create ScanAdd kernel.");


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



  // CPU resources
  cl_float4* pPosLife = new cl_float4[m_nParticles * 2];
  memset(pPosLife, 0, sizeof(cl_float4) * m_nParticles * 2);

  // fill the array with some values
  for (unsigned int i = 0; i < m_nParticles; i++) {
    pPosLife[i].s[0] = (float(rand()) / float(RAND_MAX) * 0.5f + 0.25f);
    pPosLife[i].s[1] = (float(rand()) / float(RAND_MAX) * 0.5f + 0.25f);
    pPosLife[i].s[2] = (float(rand()) / float(RAND_MAX) * 0.5f + 0.25f);
    pPosLife[i].s[3] = 1.f + 5.f * (float(rand()) / float(RAND_MAX));
  }

  // Device resources
  glGenBuffers(2, m_glPosLife);
  glBindBuffer(GL_ARRAY_BUFFER, m_glPosLife[0]);
  glBufferData(GL_ARRAY_BUFFER, m_nParticles * sizeof(cl_float4) * 2, pPosLife, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, m_glPosLife[1]);
  glBufferData(GL_ARRAY_BUFFER, m_nParticles * sizeof(cl_float4) * 2, pPosLife, GL_DYNAMIC_DRAW);
  CHECK_FOR_OGL_ERROR();

  SAFE_DELETE_ARRAY(pPosLife);



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

  GLint tboSampler = glGetUniformLocationARB(m_ProgRenderParticles, "tboSampler");
  glUseProgramObjectARB(m_ProgRenderParticles);
  glUniform1i(tboSampler, 0);



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

  SAFE_RELEASE_MEMOBJECT(m_clPosLife[0]);
  SAFE_RELEASE_MEMOBJECT(m_clPosLife[1]);

  for (auto& level : m_clScanLevels) SAFE_RELEASE_MEMOBJECT(level.first);


  SAFE_RELEASE_KERNEL(m_ScanKernel);
  SAFE_RELEASE_KERNEL(m_ScanAddKernel);
  SAFE_RELEASE_PROGRAM(m_ScanProgram);


  SAFE_RELEASE_GL_BUFFER(m_glPosLife[0]);
  SAFE_RELEASE_GL_BUFFER(m_glPosLife[1]);

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
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clPosLife[0], 0, NULL, NULL), "Error acquiring OpenGL buffer.");
  //  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clVelMass[0], 0, NULL, NULL), "Error acquiring OpenGL buffer.");
  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clPosLife[1], 0, NULL, NULL), "Error acquiring OpenGL buffer.");
  //  V_RETURN_CL(clEnqueueAcquireGLObjects(CommandQueue, 1, &m_clVelMass[1], 0, NULL, NULL), "Error acquiring OpenGL buffer.");

  // DO THE CL STUFF HERE


  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clPosLife[0], 0, NULL, NULL), "Error releasing OpenGL buffer.");
  //  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clVelMass[0], 0, NULL, NULL), "Error releasing OpenGL buffer.");
  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clPosLife[1], 0, NULL, NULL), "Error releasing OpenGL buffer.");
  //  V_RETURN_CL(clEnqueueReleaseGLObjects(CommandQueue, 1, &m_clVelMass[1], 0, NULL, NULL), "Error releasing OpenGL buffer.");

  clFinish(CommandQueue);
}


void CCreateBVH::TestPerformance(cl_context Context, cl_command_queue CommandQueue) {
  auto print_info = [&](const string& name, double timecpu, double timegpu, bool correct) {
    cout << endl
         << "############";
    for (auto c : name) cout << '#';
    cout << endl
         << " ###  " << name << "  ### " << endl;
    cout << "############";
    for (auto c : name) cout << '#';

    cout << endl
         << endl;
    if (correct) {
#ifndef WIN32
      cout << "\033[1;32m";
#endif
      cout << "CORRECT RESULTS!";
#ifndef WIN32
      cout << "\033[0m";
#endif
    } else {
#ifndef WIN32
      cout << "\033[1;31m";
#endif
      cout << "INVALID RESULTS!";
#ifndef WIN32
      cout << "\033[0m";
#endif
    }
    cout << endl
         << endl
         << "Execution time (ms):" << endl
         << "\tCPU: " << timecpu << endl
         << "\tGPU: " << timegpu << endl;
  };

  cout << "Running performance test..." << endl
       << endl;

  CTimer timer;
  double timecpu, timegpu;
  bool res = true;

  std::vector<cl_uint> mortoncodes(m_nElements);
  std::vector<cl_uint> zeroflags(m_nElements);
  std::vector<cl_uint> oneflags(m_nElements);
  std::vector<cl_uint> zeroflags_cpu(m_nElements);
  std::vector<cl_uint> oneflags_cpu(m_nElements);
  std::vector<cl_uint> mortoncodessort(m_nElements);
  std::vector<cl_uint> mortoncodessort_cpu(m_nElements);
  std::vector<cl_uint> permutation(m_nElements);
  std::vector<cl_uint> permutation_cpu(m_nElements);

  for (auto& c : mortoncodes) c = rand() & 0x00FF;


  // #############
  // PERMUTATION IDENTITY
  // #############
  {
    // Time for CPU
    timer.Start();
    for (size_t i = 0; i < m_nElements; ++i) permutation_cpu[i] = i;
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    // Time for GPU
    timer.Start();
    for (size_t i = 0; i < 100; ++i) PermutationIdentity(Context, CommandQueue, m_clSortPermutation[0]);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;

    // Validate correctness
    PermutationIdentity(Context, CommandQueue, m_clSortPermutation[0]);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clSortPermutation[0], CL_TRUE, 0, m_nElements * sizeof(cl_uint), permutation.data(), 0, NULL, NULL),
                "Error reading data from device!");

    res = memcmp(permutation.data(), permutation_cpu.data(), permutation.size() * sizeof(cl_uint)) == 0;
    print_info("PERMUTATION IDENTITY", timecpu, timegpu, res);
  }


  // #############
  // RADIX FLAGS
  // #############
  {
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
    V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_clMortonCodes[0], CL_FALSE, 0, m_nElements * sizeof(cl_uint), mortoncodes.data(), 0, NULL, NULL),
                "Error writing random keys to cl memory!");

    timer.Start();
    for (size_t i = 0; i < 100; ++i) SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes[0], 0x0001);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;

    // Validate correctness
    V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_clMortonCodes[0], CL_FALSE, 0, m_nElements * sizeof(cl_uint), mortoncodes.data(), 0, NULL, NULL),
                "Error writing random keys to cl memory!");
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes[0], 0x0001);
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
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes[0], 0x0001);

    timer.Start();
    for (size_t i = 0; i < 100; ++i) Scan(Context, CommandQueue, m_clRadixZeroBit);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;


    // Validate correctness
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes[0], 0x0001);
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
    // Initialize buffers for scan with result from SelectBitflag
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes[0], 0x0001);
    timer.Start();
    for (size_t i = 0; i < 100; ++i)
      ReorderKeys(Context, CommandQueue, m_clMortonCodes[1], m_clSortPermutation[1], m_clMortonCodes[0], m_clSortPermutation[0], m_clRadixZeroBit,
                  m_clRadixOneBit, 0x0001);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds() / 100.0;


    // Validate correctness
    PermutationIdentity(Context, CommandQueue, m_clSortPermutation[0]);
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes[0], 0x0001);
    Scan(Context, CommandQueue, m_clRadixZeroBit);
    Scan(Context, CommandQueue, m_clRadixOneBit);
    ReorderKeys(Context, CommandQueue, m_clMortonCodes[1], m_clSortPermutation[1], m_clMortonCodes[0], m_clSortPermutation[0], m_clRadixZeroBit,
                m_clRadixOneBit, 0x0001);
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonCodes[1], CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodessort.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clSortPermutation[1], CL_TRUE, 0, m_nElements * sizeof(cl_uint), permutation.data(), 0, NULL, NULL),
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
  {
    for (size_t i = 0; i < m_nElements; ++i) mortoncodessort_cpu[i] = mortoncodes[i];
    timer.Start();
    sort(mortoncodessort_cpu.begin(), mortoncodessort_cpu.end());
    timer.Stop();
    timecpu = timer.GetElapsedMilliseconds();

    timer.Start();
    RadixSort(Context, CommandQueue);
    clFinish(CommandQueue);
    timer.Stop();
    timegpu = timer.GetElapsedMilliseconds();


    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clMortonCodes[0], CL_TRUE, 0, m_nElements * sizeof(cl_uint), mortoncodessort.data(), 0, NULL, NULL),
                "Error reading data from device!");
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_clSortPermutation[0], CL_TRUE, 0, m_nElements * sizeof(cl_uint), permutation.data(), 0, NULL, NULL),
                "Error reading data from device!");

    // ASSERT that morton codes and permutation are correct,
    res = memcmp(mortoncodessort.data(), mortoncodessort_cpu.data(), mortoncodessort.size() * sizeof(cl_uint)) == 0;
    // ASSERT that the permutation permutes to the actual reorderd morton codes.
    for (size_t i = 0; i < m_nElements; ++i) {
      res = mortoncodessort[i] == mortoncodes[permutation[i]];
      if (!res) break;
    }

    print_info("RADIX SORT", timecpu, timegpu, res);
  }
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
void CCreateBVH::RadixSort(cl_context Context, cl_command_queue CommandQueue) {
  // Initialize the first source permutation to the ascending numbers from 0 to m_nElements
  PermutationIdentity(Context, CommandQueue, m_clSortPermutation[0]);

  // For each bit stable sort
  for (size_t i = 0; i < 32; ++i) {
    // Start with least important bit
    cl_uint mask = 1 << i;
    // Indices for source and destination of ping-pong buffers.
    int src = i % 2;
    int dst = (i + 1) % 2;

    // Extract the flags for the current bit.
    SelectBitflag(Context, CommandQueue, m_clRadixZeroBit, m_clRadixOneBit, m_clMortonCodes[src], mask);
    // Scan on each buffer of flags
    Scan(Context, CommandQueue, m_clRadixZeroBit);
    Scan(Context, CommandQueue, m_clRadixOneBit);
    // Reorder
    ReorderKeys(Context, CommandQueue, m_clMortonCodes[dst], m_clSortPermutation[dst], m_clMortonCodes[src], m_clSortPermutation[src], m_clRadixZeroBit,
                m_clRadixOneBit, mask);
  }
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

  glEnable(GL_POINT_SPRITE_ARB);

  glBindBuffer(GL_ARRAY_BUFFER, m_glPosLife[0]);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glColor4f(1.0, 0.0, 0.0, 0.3f);
  glDrawArrays(GL_POINTS, 0, m_nParticles);
  glDisableClientState(GL_VERTEX_ARRAY);

  glDisable(GL_POINT_SPRITE_ARB);
  glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
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
  if (m_KeyboardMask[GLFW_KEY_W]) m_TranslateZ += 2.f * ElapsedTime;
  if (m_KeyboardMask[GLFW_KEY_S]) m_TranslateZ -= 2.f * ElapsedTime;

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
  gluPerspective(60.0, (GLfloat)Width / (GLfloat)Height, 0.1, 10.0);
}


///////////////////////////////////////////////////////////////////////////////
