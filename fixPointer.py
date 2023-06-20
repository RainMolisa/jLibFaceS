#!/usr/bin/python3
from __future__ import print_function

import os
import shutil
import sys
import torch
import torch.backends.cudnn as cudnn
import argparse
import cv2
import numpy as np
from collections import OrderedDict

import onnx
import onnxoptimizer as onpz

from evConfig import ndkPath
sys.path.append(ndkPath)
import onnx_interface as oxi
import quantize as qtz
import modelpack as mdp
import quant_tools.numpy_net as npn
import layers

from faceDNet import YuFaceDetectNet,\
 netPostProc
from fcResize import fcResize
import dataGen as dg
from detPostProc import detPostProc

imgDim=160
if(__name__=='__main__'):
 from evConfig import \
  detectModel,fNameOnnx,dtBufStr,testImag,fxPDataP,\
  bitWidth,modelFolder,modelName
 mdBinFolder=modelFolder+'/'+modelName.replace('.pth','.bin')
 fixModPath=modelFolder+'/'+modelName.replace('.pth','.fix')
 if os.path.exists(dtBufStr):
  shutil.rmtree(dtBufStr)
 os.makedirs(dtBufStr)
 if(os.path.exists(mdBinFolder)):
  shutil.rmtree(mdBinFolder)
 os.makedirs(mdBinFolder)
 if(os.path.exists(fixModPath)):
  shutil.rmtree(fixModPath)
 os.makedirs(fixModPath)
 device="cuda" if torch.cuda.is_available() else "cpu"
 print("using {} device".format(device))
 net=YuFaceDetectNet('test',imgDim)
 state_dict = torch.load(detectModel)
 from collections import OrderedDict
 new_state_dict = OrderedDict()
 for k, v in state_dict.items():
  head = k[:7]
  if head == 'module.':
   name = k[7:] # remove `module.`
  else:
   name = k
  new_state_dict[name] = v
 net.load_state_dict(new_state_dict)
 net=net.to(device)
 net.eval()
 print('Finished loading model')
 imgRaw=cv2.imread(testImag,cv2.IMREAD_GRAYSCALE)
 inPp=dg.imgPerProc(imgRaw,imgDim,imgDim)
 inpt=torch.from_numpy(inPp).to(device)
 torch.onnx.export(net,inpt,fNameOnnx)
 print('save onnx')
 layerList, paramDict = oxi.load_from_onnx(fNameOnnx)
 print('onnx load success\n')
 dataSize=dg.genLen0(fxPDataP)
 _dg=dg.dataGen0(fxPDataP,imgDim,imgDim,False)
 quantLayerList,quantParamDict=qtz.quantize_model\
  (layerList,paramDict,bitWidth,_dg,num_step=dataSize)
 print('quant finish')
 mdp.modelpack(bitWidth,quantLayerList,quantParamDict,\
  mdBinFolder,use_machine_code=True)
 qllFilePath=fixModPath+'/qtll'
 qpdFilePath=fixModPath+'/qtpd'
 mdp.save_to_file(quantLayerList,qllFilePath,\
  quantParamDict,qpdFilePath)