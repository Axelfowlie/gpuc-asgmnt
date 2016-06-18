/******************************************************************************
                         .88888.   888888ba  dP     dP
                        d8'   `88  88    `8b 88     88
                        88        a88aaaa8P' 88     88
                        88   YP88  88        88     88
                        Y8.   .88  88        Y8.   .8P
                         `88888'   dP        `Y88888P'


   a88888b.                                         dP   oo
  d8'   `88                                         88
  88        .d8888b. 88d8b.d8b. 88d888b. dP    dP d8888P dP 88d888b. .d8888b.
  88        88'  `88 88'`88'`88 88'  `88 88    88   88   88 88'  `88 88'  `88
  Y8.   .88 88.  .88 88  88  88 88.  .88 88.  .88   88   88 88    88 88.  .88
   Y88888P' `88888P' dP  dP  dP 88Y888P' `88888P'   dP   dP dP    dP `8888P88
                                88                                        .88
                                dP                                    d8888P
******************************************************************************/

#ifndef CPARTICLESYSTEMTASK_H_80VXUHBT
#define CPARTICLESYSTEMTASK_H_80VXUHBT


#include "../Common/IGUIEnabledComputeTask.h"

#include "CTriMesh.h"
#include "CGLTexture.h"

#include <string>

//! A4 / T1 Particle system
/*!
        Note that this task does not implement a CPU reference result, all simulation is
        running on the GPU. The methods related to the evaluation of the correctness of the
        kernel are therefore meaningless (not implemented).
*/
class CCreateBVH : public IGUIEnabledComputeTask {
public:
  CCreateBVH(const std::string& CollisionMeshPath, size_t NElems, size_t scanWorksize, size_t LocalWorkSize[3]);

  virtual ~CCreateBVH();

  // IComputeTask
  virtual bool InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue);
  bool InitGL();
  bool InitCL(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue);

  virtual void ReleaseResources();

  virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);

  virtual void TestPerformance(cl_context Context, cl_command_queue CommandQueue);


  void AdvancePositions(cl_context Context, cl_command_queue CommandQueue);
  void CreateLeafAABBs(cl_context Context, cl_command_queue CommandQueue);
  void ReduceAABB(cl_context Context, cl_command_queue CommandQueue, cl_mem aabbs, cl_kernel Kernel);
  void MortonCodeAABB(cl_context Context, cl_command_queue CommandQueue);
  void MortonCodes(cl_context Context, cl_command_queue CommandQueue, cl_mem codes, cl_mem positions, cl_mem aabb);

  void PermutationIdentity(cl_context Context, cl_command_queue CommandQueue, cl_mem permutation);
  void SelectBitflag(cl_context Context, cl_command_queue CommandQueue, cl_mem flagnotset, cl_mem flagset, cl_mem keys, cl_uint mask);
  void Scan(cl_context Context, cl_command_queue CommandQueue, cl_mem inoutbuffer);
  void ReorderKeys(cl_context Context, cl_command_queue CommandQueue, cl_mem keysout, cl_mem permutationout, cl_mem keysin, cl_mem permutationin,
                   cl_mem indexzerobits, cl_mem indexonebits, cl_uint mask);
  void RadixSort(cl_context Context, cl_command_queue CommandQueue);


  // Not implemented!
  virtual void ComputeCPU(){};
  virtual bool ValidateResults() {
    return false;
  };

  // IGUIEnabledComputeTask
  virtual void Render();

  virtual void OnKeyboard(int Key, int Action);

  virtual void OnMouse(int Button, int State);

  virtual void OnMouseMove(int X, int Y);

  virtual void OnIdle(double Time, float ElapsedTime);

  virtual void OnWindowResized(int Width, int Height);


protected:
  // Number of leaf nodes in the BVH
  size_t m_nElements = 0;
  size_t m_ScanLocalWorkSize[3];


  // BVH data
  // Center positions and radius of the AABBs + velocity
  cl_mem m_clPositions = nullptr;
  cl_mem m_clVelocities = nullptr;
  // AABBs for leaf and inner nodes
  // Each one as two float4 arrays, [0] for the minimum and [1] for the maximum
  cl_mem m_clAABBLeaves[2] = {nullptr, nullptr};
  cl_mem m_clAABBNodes[2] = {nullptr, nullptr};
  // Buffer to hold the AABB to calculate the Morton codes
  cl_mem m_clMortonAABB = nullptr;
  // AABBs shall be displayed using wireframe boxes
  GLuint m_glAABBLeafBuf[2] = { 0, 0 };
  GLuint m_glAABBLeafTB[2] = { 0, 0 };
  GLuint m_glAABBNodeBuf[2] = { 0, 0 };
  GLuint m_glAABBNodeTB[2] = { 0, 0 };
  
  cl_program m_PrepAABBsProgram = nullptr;
  cl_kernel m_AdvancePositionsKernel = nullptr;
  cl_kernel m_CreateLeafAABBsKernel = nullptr;

  cl_program m_MortonCodesProgram = nullptr;
  cl_kernel m_ReduceAABBminKernel = nullptr;
  cl_kernel m_ReduceAABBmaxKernel = nullptr;
  cl_kernel m_MortonCodeAABBKernel = nullptr;
  cl_kernel m_MortonCodesKernel = nullptr;



  // RADIX SORT
  // Ping-pong buffers for morton codes (= actual sorted keys) and the permutation
  // Ping-pongin needed in the reorder kernel
  cl_mem m_clMortonCodes[2] = {nullptr, nullptr};
  cl_mem m_clSortPermutation[2] = {nullptr, nullptr};
  // Buffers to hold the flags for the current sorted radix
  cl_mem m_clRadixZeroBit = nullptr;
  cl_mem m_clRadixOneBit = nullptr;

  cl_program m_RadixSortProgram = nullptr;
  cl_kernel m_SelectBitflagKernel = nullptr;
  cl_kernel m_ReorderKeysKernel = nullptr;
  cl_kernel m_PermutationIdentityKernel = nullptr;
  
  // PARALLEL SCAN FOR RADIX SORT
  // Arrays for each level of the work-efficient scan
  std::vector<std::pair<cl_mem, size_t>> m_clScanLevels;

  cl_program m_ScanProgram = nullptr;
  cl_kernel m_ScanKernel = nullptr;
  cl_kernel m_ScanAddKernel = nullptr;


  //
  //
  //
  //
  //
  //
  //



  std::string m_CollisionMeshPath;

  CTriMesh* m_pMesh = nullptr;


  GLhandleARB m_VSParticles = 0;
  GLhandleARB m_PSParticles = 0;
  GLhandleARB m_ProgRenderParticles = 0;

  GLhandleARB m_VSMesh = 0;
  GLhandleARB m_PSMesh = 0;
  GLhandleARB m_ProgRenderMesh = 0;


  bool m_KeyboardMask[255] /*= { false }*/;

  // mouse
  int m_Buttons = 0;
  int m_PrevX = 0;
  int m_PrevY = 0;

  // for camera handling
  float m_RotateX = 0.0f;
  float m_RotateY = 0.0f;
  float m_TranslateZ = 0.0f;
};

#endif /* end of include guard: CPARTICLESYSTEMTASK_H_80VXUHBT */
