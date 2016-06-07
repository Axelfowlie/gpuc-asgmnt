/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment5.h"

#include <iostream>

using namespace std;


int main(int argc, char** argv)
{
	CAssignment5* pAssignment = CAssignment5::GetSingleton();

	if(pAssignment)
	{
		pAssignment->EnterMainLoop(argc, argv);
		delete pAssignment;
	}
	
#ifdef _MSC_VER
	cout<<"Press any key..."<<endl;
	cin.get();
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


