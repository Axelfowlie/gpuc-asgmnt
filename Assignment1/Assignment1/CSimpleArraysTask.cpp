/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CSimpleArraysTask.h"

#include "../Common/CLUtil.h"

#include <string.h>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CSimpleArraysTask

CSimpleArraysTask::CSimpleArraysTask(size_t ArraySize) : m_ArraySize(ArraySize) {
}

CSimpleArraysTask::~CSimpleArraysTask() {
  ReleaseResources();
}

bool CSimpleArraysTask::InitResources(cl_device_id Device, cl_context Context) {
  // CPU resources
  m_hA = new int[m_ArraySize];
  m_hB = new int[m_ArraySize];
  m_hC = new int[m_ArraySize];
  m_hGPUResult = new int[m_ArraySize];

  // fill A and B with random integers
  for (unsigned int i = 0; i < m_ArraySize; i++) {
    m_hA[i] = rand() % 1024;
    m_hB[i] = rand() % 1024;
  }

  // device resources

  /////////////////////////////////////////
  // Sect. 4.5
  // Create a OpenCL buffer resource for each input and output array
  cl_int clError;
  m_dA = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create buffer A.");

  m_dB = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create buffer B.");

  m_dC = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create buffer C.");


  /////////////////////////////////////////
  // Sect. 4.6.
  // Load the kernel source and build a program (and kernel)
  string programCode;
  if (!CLUtil::LoadProgramSourceToMemory("VectorAdd.cl", programCode)) return false;

  m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
  if (m_Program == nullptr) return false;

  m_Kernel = clCreateKernel(m_Program, "VecAdd", &clError);
  V_RETURN_FALSE_CL(clError, "Failed to create kernel for \"VecAdd\".");

  // Set the input parameters of the kernel
  clError = clSetKernelArg(m_Kernel, 0, sizeof(cl_mem), (void*)&m_dA);
  clError |= clSetKernelArg(m_Kernel, 1, sizeof(cl_mem), (void*)&m_dB);
  clError |= clSetKernelArg(m_Kernel, 2, sizeof(cl_mem), (void*)&m_dC);
  clError |= clSetKernelArg(m_Kernel, 3, sizeof(cl_int), (void*)&m_ArraySize);
  V_RETURN_FALSE_CL(clError, "Failed to bind kernel arguments.");

  return true;
}

void CSimpleArraysTask::ReleaseResources() {
  // CPU resources
  SAFE_DELETE_ARRAY(m_hA);
  SAFE_DELETE_ARRAY(m_hB);
  SAFE_DELETE_ARRAY(m_hC);
  SAFE_DELETE_ARRAY(m_hGPUResult);

  /////////////////////////////////////////////////
  // Sect. 4.5., 4.6.
  SAFE_RELEASE_MEMOBJECT(m_dA);
  SAFE_RELEASE_MEMOBJECT(m_dB);
  SAFE_RELEASE_MEMOBJECT(m_dC);

  if (m_Kernel != nullptr) {
    clReleaseKernel(m_Kernel);
    m_Kernel = nullptr;
  }
  if (m_Program != nullptr) {
    clReleaseProgram(m_Program);
    m_Program = nullptr;
  }

  // TO DO: free resources on the GPU
}

void CSimpleArraysTask::ComputeCPU() {
  for (unsigned int i = 0; i < m_ArraySize; i++) {
    m_hC[i] = m_hA[i] + m_hB[m_ArraySize - i - 1];
  }
}

void CSimpleArraysTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]) {
  /////////////////////////////////////////////////
  // Sect. 4.5
  // Copy input data into CL buffers
  cl_int clError;
  clError = clEnqueueWriteBuffer(CommandQueue, m_dA, CL_FALSE, 0, m_ArraySize * sizeof(int), m_hA, 0, NULL, NULL);
  V_RETURN_CL(clError, "Failed to enqueue buffer write operation for A.");

  clError = clEnqueueWriteBuffer(CommandQueue, m_dB, CL_FALSE, 0, m_ArraySize * sizeof(int), m_hB, 0, NULL, NULL);
  V_RETURN_CL(clError, "Failed to enqueue buffer write operation for B.");

  /////////////////////////////////////////
  // Sect. 4.6.
  // Calculate the global work size
  size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_ArraySize, LocalWorkSize[0]);
  size_t nGroups = globalWorkSize / LocalWorkSize[0];
  cout << "\n\tExecuting " << globalWorkSize << " threads in " << nGroups << " groups of size " << LocalWorkSize[0] << " ...";

  // clError = clEnqueueNDRangeKernel(CommandQueue, m_Kernel, 1, NULL, &globalWorkSize, LocalWorkSize, 0, NULL, NULL);
  // V_RETURN_CL(clError, "Failed to execute kernel.");

  // Sect. 4.7.: rewrite the kernel call to use our ProfileKernel()
  //				utility function to measure execution time.
  //				Also print out the execution time.

  // Execute the kernel n times
  int NIterations = 100;
  double ms = CLUtil::ProfileKernel(CommandQueue, m_Kernel, 1, &globalWorkSize, LocalWorkSize, NIterations);
  cout << cout.precision(10) << "\n\tAveraging " << ms << " milliseconds over " << NIterations << " iterations ...";
  cout << "\n\t" << m_ArraySize / ms / 1000000.0 << " million elements per millisecond ...";

  // cout << endl
  //     << endl
  //     << "(" << m_ArraySize / ms / 1000000.0 << "," << m_ArraySize << ")" << endl;

  // This command has to be blocking, since we need the data
  clError = clEnqueueReadBuffer(CommandQueue, m_dC, CL_TRUE, 0, m_ArraySize * sizeof(int), m_hGPUResult, 0, NULL, NULL);
  V_RETURN_CL(clError, "Failed to enqueue buffer read operation for C.");
}

bool CSimpleArraysTask::ValidateResults() {
  return (memcmp(m_hC, m_hGPUResult, m_ArraySize * sizeof(float)) == 0);
}

///////////////////////////////////////////////////////////////////////////////
