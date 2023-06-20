import sys
import os
def fDPPItf(w,h,cfg,nwOtLs,nwOSz):
 from evConfig import confidenceThreshold,\
  topK,nmsThreshold,outFolder
 pwd=os.getcwd()
 pwd=pwd.replace('\\','/')
 fileName=pwd+'/csrc/fDPPItf.h'
 resStr=''
 resStr=resStr+'#pragma once\n'
 resStr=resStr+'#include"NCHW.h"\n'
 resStr=resStr+'#include"input_output_info.h"\n'
 resStr=resStr+'namespace fdpp{\n'
 resStr=resStr+'\tstruct cfg{\n'
 resStr=resStr+'\t\tint w=%d;\n'%w
 resStr=resStr+'\t\tint h=%d;\n'%h
 resStr=resStr+'\t\tfloat varience[%d]={'%len(cfg['variance'])
 for e in cfg['variance']:
  resStr=resStr+'%f,'%e
 resStr=resStr[:-1]
 resStr=resStr+'};\n'
 resStr=resStr+'\t\tfloat confThres=%f;\n'%confidenceThreshold
 resStr=resStr+'\t\tint topK=%d;\n'%topK
 resStr=resStr+'\t\tfloat nmsThres=%f;\n'%nmsThreshold
 resStr=resStr+'\t}_cfg;\n'
 resStr=resStr+'\tconst int layNum=%d;\n'%len(cfg['min_sizes'])
 resStr=resStr+'\tstruct minSizes{\n'
 resStr=resStr+'\t\tint lys[layNum]={'
 for e in cfg['min_sizes']:
  resStr=resStr+'%d,'%len(e)
 resStr=resStr[:-1]
 resStr=resStr+'};\n'
 resStr=resStr+'\t\tint ls_[layNum]={'
 n=0
 for i in range(len(cfg['min_sizes'])):
  resStr=resStr+'%d,'%n
  n=n+len(cfg['min_sizes'][i])
 resStr=resStr[:-1]
 resStr=resStr+'};\n'
 #print(n)
 resStr=resStr+'\t\tint dat[%d]={'%n
 for i in cfg['min_sizes']:
  for e in i:
   resStr=resStr+str(e)+','
 resStr=resStr[:-1]
 resStr=resStr+'};\n'
 resStr=resStr+'\tpublic:\n'
 resStr=resStr+'\t\tinline int operator()(int i, int j) {\n'
 resStr=resStr+'\t\t\treturn dat[ls_[i] + j];\n'
 resStr=resStr+'\t\t}\n'
 resStr=resStr+'\t}_minSizes;\n'
 resStr=resStr+'\tint steps[layNum]={'
 for e in cfg['steps']:
  resStr=resStr+'%d,'%e
 resStr=resStr[:-1]
 resStr=resStr+'};\n'
 resStr=resStr+'\tstd::vector<bx::faceBox> faceDPostProc(NCHW::nchw<float> *ts);\n'
 resStr=resStr+'\tstd::vector<bx::faceBox> faceDPostProc(uint8_t* ftData){\n'
 for i,e in enumerate(nwOtLs):
  resStr=resStr+'\t\tint8_t *n%d=(int8_t*)(ftData+MODEL_%s_ADDR_OFFSET);\n'%(i,e)
 resStr=resStr+'//#define iotLog\n#ifdef iotLog\n'
 for i,e in enumerate(nwOtLs):
  resStr=resStr+'\tiot_printf("\\n---------n%d----------\\n");\n'%i
  resStr=resStr+'\tfor(int i=0;i<%d;i++)\n'%(nwOSz[i][0]*nwOSz[i][1]*nwOSz[i][2]*32)
  resStr=resStr+'\t\tiot_printf("%d\\n",'
  resStr=resStr+'n%d[i]);\n'%i
 resStr=resStr+'#endif\n'
 resStr=resStr+'\t\tNCHW::nchw<float> ts[%d];\n'%(len(nwOtLs))
 for i,e in enumerate(nwOtLs):
  resStr=resStr+('\t\tts[%d].create(%d,%d,%d,%d,n%d,MODEL_%s_FRAC);\
  \n')%(i,nwOSz[i][0],nwOSz[i][1],nwOSz[i][2],nwOSz[i][3],i,e)
 resStr=resStr+'\t\treturn faceDPostProc(ts);\n'
 resStr=resStr+'\t}\n';
 
 resStr=resStr+'#ifndef noStream\n'
 resStr=resStr+'\tstd::vector<bx::faceBox> faceDPostProc(){\n'
 resStr=resStr+'\t\tNCHW::nchw<float> ts[%d];\n'%(len(nwOtLs))
 for i,e in enumerate(nwOtLs):
  #resStr=resStr+('\t\tts[%d].create("'%i)+pwd+'/'+outFolder+'/npu_'+e+'");\n'
  resStr=resStr+('\t\tts[%d].create<int8_t>('%i);
  resStr=resStr+'%d,%d,%d,%d,MODEL_%s_FRAC'%(nwOSz[i][0],nwOSz[i][1],nwOSz[i][2],nwOSz[i][3],e)
  resStr=resStr+',"'+pwd+'/'+outFolder+'/npu_'+e+'");\n'
 resStr=resStr+'\t\treturn faceDPostProc(ts);\n'
 resStr=resStr+'\t}\n';
 resStr=resStr+'#endif\n'
 
 resStr=resStr+'};\n'
 with open(fileName,'w') as f:
  f.write(resStr)
  f.close()
 return