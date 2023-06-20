#pragma once
#include"NCHW.h"
#include"input_output_info.h"
namespace fdpp{
	struct cfg{
		int w=160;
		int h=160;
		float varience[2]={0.100000,0.200000};
		float confThres=0.650000;
		int topK=25;
		float nmsThres=0.500000;
	}_cfg;
	const int layNum=3;
	struct minSizes{
		int lys[layNum]={3,2,2};
		int ls_[layNum]={0,3,5};
		int dat[7]={10,16,24,32,48,64,96};
	public:
		inline int operator()(int i, int j) {
			return dat[ls_[i] + j];
		}
	}_minSizes;
	int steps[layNum]={8,16,32};
	std::vector<bx::faceBox> faceDPostProc(NCHW::nchw<float> *ts);
	std::vector<bx::faceBox> faceDPostProc(uint8_t* ftData){
		int8_t *n0=(int8_t*)(ftData+MODEL_204_ADDR_OFFSET);
		int8_t *n1=(int8_t*)(ftData+MODEL_210_ADDR_OFFSET);
		int8_t *n2=(int8_t*)(ftData+MODEL_216_ADDR_OFFSET);
//#define iotLog
#ifdef iotLog
	iot_printf("\n---------n0----------\n");
	for(int i=0;i<32640;i++)
		iot_printf("%d\n",n0[i]);
	iot_printf("\n---------n1----------\n");
	for(int i=0;i<10880;i++)
		iot_printf("%d\n",n1[i]);
	iot_printf("\n---------n2----------\n");
	for(int i=0;i<5440;i++)
		iot_printf("%d\n",n2[i]);
#endif
		NCHW::nchw<float> ts[3];
		ts[0].create(1,51,20,20,n0,MODEL_204_FRAC);  
		ts[1].create(1,34,10,10,n1,MODEL_210_FRAC);  
		ts[2].create(1,34,5,5,n2,MODEL_216_FRAC);  
		return faceDPostProc(ts);
	}
#ifndef noStream
	std::vector<bx::faceBox> faceDPostProc(){
		NCHW::nchw<float> ts[3];
		ts[0].create<int8_t>(1,51,20,20,MODEL_204_FRAC,"D:/jhh/code/jLibFaceS/out/npu_204");
		ts[1].create<int8_t>(1,34,10,10,MODEL_210_FRAC,"D:/jhh/code/jLibFaceS/out/npu_210");
		ts[2].create<int8_t>(1,34,5,5,MODEL_216_FRAC,"D:/jhh/code/jLibFaceS/out/npu_216");
		return faceDPostProc(ts);
	}
#endif
};
