/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CLUtil.h"
#include "CTimer.h"

#include <iostream>
#include <fstream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CLUtil

size_t CLUtil::GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize) {
  size_t r = DataElemCount % LocalWorkSize;
  return (r == 0) ? DataElemCount : DataElemCount + LocalWorkSize - r;
}

bool CLUtil::LoadProgramSourceToMemory(const std::string& Path, std::string& SourceCode) {
  ifstream sourceFile;

  sourceFile.open(Path.c_str());
  if (!sourceFile.is_open()) {
    cerr << "Failed to open file '" << Path << "'." << endl;
    return false;
  }

  // read the entire file into a string
  sourceFile.seekg(0, ios::end);
  ifstream::pos_type fileSize = sourceFile.tellg();
  sourceFile.seekg(0, ios::beg);

  SourceCode.clear();
  SourceCode.resize((size_t)fileSize);
  sourceFile.read(&SourceCode[0], fileSize);

  return true;
}

cl_program CLUtil::BuildCLProgramFromMemory(cl_device_id Device, cl_context Context, const std::string& SourceCode, const std::string& CompileOptions) {
  // Ignore the last parameter CompileOptions in assignment 1
  // This may be used later to pass flags and macro definitions to the OpenCL compiler

  const char* src = SourceCode.c_str();
  size_t len = SourceCode.length();
  cl_int clError;

  // Create a program to hold the source
  cl_program prog = clCreateProgramWithSource(Context, 1, &src, &len, &clError);
  if (clError != CL_SUCCESS) {
    cout << "Failed to create CL program from source." << endl;
    return nullptr;
  }

  // Compile and link
  clError = clBuildProgram(prog, 1, &Device, NULL, NULL, NULL);
  if (clError != CL_SUCCESS) {
    // Only in case of error print the build log
    PrintBuildLog(prog, Device);
    cout << "Failed to build CL program." << endl;
    SAFE_RELEASE_PROGRAM(prog);
    return nullptr;
  }

  return prog;
}

void CLUtil::PrintBuildLog(cl_program Program, cl_device_id Device) {
  cl_build_status buildStatus;
  clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);

  // let's print out possible warnings even if the kernel compiled..
  // if(buildStatus == CL_SUCCESS)
  //	return;

  // there were some errors.
  size_t logSize;
  clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
  string buildLog(logSize, ' ');

  clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
  buildLog[logSize] = '\0';

  if (buildStatus != CL_SUCCESS) cout << "There were build errors!" << endl;
  cout << "Build log:" << endl;
  cout << buildLog << endl;
}

double CLUtil::ProfileKernel(cl_command_queue CommandQueue, cl_kernel Kernel, cl_uint Dimensions, const size_t* pGlobalWorkSize, const size_t* pLocalWorkSize,
                             int NIterations) {
  CTimer timer;
  cl_int clError;

  // Finish pending cl commands in the queue
  clError = clFinish(CommandQueue);

  // Start time measurement and que up the kernels
  timer.Start();
  for (int i = 0; i < NIterations; ++i) {
    clError |= clEnqueueNDRangeKernel(CommandQueue, Kernel, Dimensions, NULL, pGlobalWorkSize, pLocalWorkSize, 0, NULL, NULL);
  }

  // Wait for kernels to finish and stop the timer
  clError = clFinish(CommandQueue);
  timer.Stop();

  if (clError != CL_SUCCESS) {
    cout << "Failed executing kernel." << endl;
    return -1;
  }

  // Average time over all iterations
  return timer.GetElapsedMilliseconds() / ((double)NIterations);
}

#define CL_ERROR(x) \
  case (x): return #x;

const char* CLUtil::GetCLErrorString(cl_int CLErrorCode) {
  switch (CLErrorCode) {
    CL_ERROR(CL_SUCCESS);
    CL_ERROR(CL_DEVICE_NOT_FOUND);
    CL_ERROR(CL_DEVICE_NOT_AVAILABLE);
    CL_ERROR(CL_COMPILER_NOT_AVAILABLE);
    CL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    CL_ERROR(CL_OUT_OF_RESOURCES);
    CL_ERROR(CL_OUT_OF_HOST_MEMORY);
    CL_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE);
    CL_ERROR(CL_MEM_COPY_OVERLAP);
    CL_ERROR(CL_IMAGE_FORMAT_MISMATCH);
    CL_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    CL_ERROR(CL_BUILD_PROGRAM_FAILURE);
    CL_ERROR(CL_MAP_FAILURE);
    CL_ERROR(CL_INVALID_VALUE);
    CL_ERROR(CL_INVALID_DEVICE_TYPE);
    CL_ERROR(CL_INVALID_PLATFORM);
    CL_ERROR(CL_INVALID_DEVICE);
    CL_ERROR(CL_INVALID_CONTEXT);
    CL_ERROR(CL_INVALID_QUEUE_PROPERTIES);
    CL_ERROR(CL_INVALID_COMMAND_QUEUE);
    CL_ERROR(CL_INVALID_HOST_PTR);
    CL_ERROR(CL_INVALID_MEM_OBJECT);
    CL_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    CL_ERROR(CL_INVALID_IMAGE_SIZE);
    CL_ERROR(CL_INVALID_SAMPLER);
    CL_ERROR(CL_INVALID_BINARY);
    CL_ERROR(CL_INVALID_BUILD_OPTIONS);
    CL_ERROR(CL_INVALID_PROGRAM);
    CL_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
    CL_ERROR(CL_INVALID_KERNEL_NAME);
    CL_ERROR(CL_INVALID_KERNEL_DEFINITION);
    CL_ERROR(CL_INVALID_KERNEL);
    CL_ERROR(CL_INVALID_ARG_INDEX);
    CL_ERROR(CL_INVALID_ARG_VALUE);
    CL_ERROR(CL_INVALID_ARG_SIZE);
    CL_ERROR(CL_INVALID_KERNEL_ARGS);
    CL_ERROR(CL_INVALID_WORK_DIMENSION);
    CL_ERROR(CL_INVALID_WORK_GROUP_SIZE);
    CL_ERROR(CL_INVALID_WORK_ITEM_SIZE);
    CL_ERROR(CL_INVALID_GLOBAL_OFFSET);
    CL_ERROR(CL_INVALID_EVENT_WAIT_LIST);
    CL_ERROR(CL_INVALID_EVENT);
    CL_ERROR(CL_INVALID_OPERATION);
    CL_ERROR(CL_INVALID_GL_OBJECT);
    CL_ERROR(CL_INVALID_BUFFER_SIZE);
    CL_ERROR(CL_INVALID_MIP_LEVEL);
    default: return "Unknown error code";
  }
}

///////////////////////////////////////////////////////////////////////////////
