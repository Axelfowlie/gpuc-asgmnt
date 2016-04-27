/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment1.h"

#include "CSimpleArraysTask.h"
#include "CMatrixRotateTask.h"

#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment1

bool CAssignment1::DoCompute() {
  // Task 1: simple array addition.
  cout << "Running vector addition example..." << endl
       << endl;


  //size_t locsize[] = {16, 32, 64, 128, 256, 512, 1024};
  //size_t vecsize[] = {10000, 100000, 1000000, 10000000, 100000000};

  //for (int i = 0; i < 7; ++i) {
  //  for (int j = 0; j < 5; ++j) {
  //    size_t LocalWorkSize[3] = {locsize[i], 1, 1};
  //    CSimpleArraysTask task(vecsize[j]);
  //    RunComputeTask(task, LocalWorkSize);
  //  }
  //}

  {
    size_t LocalWorkSize[3] = {256, 1, 1};
    CSimpleArraysTask task(1564320);
    RunComputeTask(task, LocalWorkSize);
  }


	// Task 2: matrix rotation.
	std::cout << "Running matrix rotation example..." << std::endl << std::endl;
	{
		size_t LocalWorkSize[3] = {16, 16, 1};
		CMatrixRotateTask task(2048, 1025);
		RunComputeTask(task, LocalWorkSize);
	}
	{
		size_t LocalWorkSize[3] = {32, 16, 1};
		CMatrixRotateTask task(2048, 1025);
		RunComputeTask(task, LocalWorkSize);
	}
	{
		size_t LocalWorkSize[3] = {32, 32, 1};
		CMatrixRotateTask task(2048, 1025);
		RunComputeTask(task, LocalWorkSize);
	}
	{
		size_t LocalWorkSize[3] = {30, 20, 1};
		CMatrixRotateTask task(6001, 4000);
		RunComputeTask(task, LocalWorkSize);
	}

	return true;
}

