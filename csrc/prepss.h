#pragma once
#include"input_output_info.h"
#include"ptrResize.h"
class ptrPrepss100160 {
private:
	const int iRows = 100;
	const int iCols = 100;
	const int oRows = 160;
	const int oCols = 160;
	int iN = iRows * iCols;
	int oN = oRows * oCols;
	resize100160 rsz;
	uint8_t* bfPtr;
public:
	void make(uint16_t* inPtr, uint8_t* otPtr);
public:
	ptrPrepss100160();
	~ptrPrepss100160();
private:
	void intial();
};
void ptrPrepss100160::make(uint16_t* inPtr, uint8_t* otPtr) {
	for (int i = 0; i < iN; i++) {
		bfPtr[i] = inPtr[i] >> 4;
#if MODEL_INPUT__1_FRAC!=0
		bfPtr[i] = bfPtr[i] << MODEL_INPUT__1_FRAC;
#endif
	}
	rsz.make(bfPtr, otPtr);
}
ptrPrepss100160::ptrPrepss100160() {
	intial();
}
ptrPrepss100160::~ptrPrepss100160() {
	delete[] bfPtr;
}
void ptrPrepss100160::intial() {
	bfPtr = new uint8_t[iN];
}

#ifndef useSimd
template<typename A, typename B>
void ptrPrepss(A* inPtr, int iRows, int iCols,
	B* otPtr, int oRows, int oCols);
template<typename A, typename B>
void ptrPrepss100160(A* inPtr, B* otPtr);
template<typename A, typename B>
void ptrPrepss(A* inPtr, int iRows, int iCols,
	B* otPtr, int oRows, int oCols) {
	int iN = iRows * iCols;
	int oN = oRows * oCols;
	A* bfPtr = new A[iN];
	for (int i = 0; i < iN; i++) {
		bfPtr[i] = inPtr[i] >> 4;
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
#else
//void ptrPrepss100160(uint16_t* inPtr, uint8_t* otPtr) {
//	const int iRows = 100;
//	const int iCols = 100;
//	const int oRows = 160;
//	const int oCols = 160;
//	int iN = iRows * iCols;
//	int oN = oRows * oCols;
//	uint8_t* bfPtr = new uint8_t[iN];
//	for (int i = 0; i < iN; i++) {
//		bfPtr[i] = inPtr[i] >> 4;
//#if MODEL_INPUT__1_FRAC!=0
//		bfPtr[i] = bfPtr[i] << MODEL_INPUT__1_FRAC;
//#endif
//	}
//	ptr100resize160(bfPtr, otPtr);
//	delete[] bfPtr;
//}
#endif
