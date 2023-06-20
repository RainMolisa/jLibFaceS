#pragma once
#include"input_output_info.h"
#include"ptrResize.h"
template<typename A,typename B>
void ptrPrepss(A* inPtr,int iRows,int iCols,
	B* otPtr,int oRows,int oCols);
template<typename A,typename B>
void ptrPrepss100160(A* inPtr,B* otPtr);
template<typename A, typename B>
void ptrPrepss(A* inPtr, int iRows, int iCols,
	B* otPtr, int oRows, int oCols) {
	int iN = iRows * iCols;
	int oN = oRows * oCols;
	A* bfPtr = new A[iN];
	for (int i = 0; i < iN; i++) {
		bfPtr[i] = inPtr[i]>>4;
#if MODEL_INPUT__1_FRAC!=0
		bfPtr[i] = bfPtr[i] << MODEL_INPUT__1_FRAC;
#endif
	}
	ptrResize<A, B>(bfPtr, iRows, iCols, otPtr, oRows, oCols);
	delete[] bfPtr;
}
template<typename A, typename B>
void ptrPrepss100160(A* inPtr, B* otPtr) {
	ptrPrepss<A, B>(inPtr, 100, 100, otPtr, 160, 160);
}