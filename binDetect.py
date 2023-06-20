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
#from collections import OrderDict

from evConfig import ndkPath
sys.path.append(ndkPath)
import onnx_interface as oxi
import quantize as qtz
import modelpack as mdp
import quant_tools.numpy_net as npn
import layers

from writeNCHW import writeNCHW,writeNPUType
from faceDNet import cfg,netPostProc
from fcResize import fcResize
import dataGen as dg
from detPostProc import detPostProc
from mkIntfc import fDPPItf

imgDim=160
if(__name__=='__main__'):
 from evConfig import visThres,\
  confidenceThreshold,topK,nmsThreshold,\
  keepTopK,modelFolder,modelFolder,bitWidth,\
  testImag,outFolder,modelName
 mdBinFolder=modelFolder+'/'+modelName.replace('.pth','.bin')
 fixModPath=modelFolder+'/'+modelName.replace('.pth','.fix')
 qllFilePath=fixModPath+'/qtll.prototxt'
 qpdFilePath=fixModPath+'/qtpd.npz'
 device="cuda" if torch.cuda.is_available() else "cpu"
 layerList,paramDict=mdp.load_from_file(qllFilePath,qpdFilePath)
 print('load finish')
 nwOutList=layers.get_network_output(layerList)
 if(os.path.exists(outFolder)):
  shutil.rmtree(outFolder)
 os.makedirs(outFolder)
 imgRaw=cv2.imread(testImag,cv2.IMREAD_GRAYSCALE)
 imgShw=cv2.imread(testImag,cv2.IMREAD_COLOR)
 imgRsz=fcResize(imgRaw,imgDim,imgDim)
 imgShw=fcResize(imgShw,imgDim,imgDim)
 inPp=dg.imgPerProc(imgRaw,imgDim,imgDim)
 writeNCHW(inPp,outFolder+'/inpt')
 hd3=npn.run_layers(inPp,layerList,nwOutList,param_dict=paramDict,\
  bitwidth=bitWidth,quant=True)
 nwOutList.sort()
 nwOutSize=list()
 for ole in nwOutList:
  print(ole,'=',hd3[ole].shape)
  nwOutSize.append(hd3[ole].shape)
 headData3=list()
 for ole in nwOutList:
  headData3.append(torch.from_numpy(hd3[ole]).to(device))
  writeNCHW(hd3[ole],outFolder+'/'+ole)
  writeNPUType(hd3[ole],paramDict[ole+'_frac'],bitWidth,outFolder+'/npu_'+ole)
 loc,conf,iou=netPostProc(headData3,2)
 print('loc=',loc.shape)
 print('conf=',conf.shape)
 print('iou=',iou.shape)
 dets=detPostProc(loc, conf, iou,imgDim,imgDim,confidenceThreshold\
  ,topK,keepTopK,nmsThreshold)
 fDPPItf(imgDim,imgDim,cfg,nwOutList,nwOutSize)
 #np.savetxt(dtSpcStr+'/'+dtSpcBuf+'/dets2.txt',dets)
 face_cc = 0
 #exit()
 for k in range(dets.shape[0]):
  if dets[k, 14] < visThres:
   continue
  xmin = dets[k, 0]
  ymin = dets[k, 1]
  xmax = dets[k, 2]
  ymax = dets[k, 3]
  score = dets[k, 14]
  wr = xmax - xmin + 1
  hr = ymax - ymin + 1
  print('{}: {:.3f} {:.3f} {:.3f} {:.3f} {:.10f}'.format(face_cc, xmin, ymin, wr, hr, score))
  face_cc =  face_cc + 1
 #cv2.imwrite(outPath+'/img.bmp',imgRsz);
 #img_rsz=cv2.imread(outPath+'/img.bmp',cv2.IMREAD_COLOR)
 cv2.imwrite(outFolder+'/imgOrg.png',imgShw)
 # show image
 for b in dets:
  if b[14] < visThres:
   continue
  text = "{:.4f}".format(b[14])
  b = list(map(int, b))
  cv2.rectangle(imgShw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

  cv2.circle(imgShw, (b[4], b[4 + 1]), 2, (255, 0, 0), 2)
  cv2.circle(imgShw, (b[4 + 2], b[4 + 3]), 2, (0, 0, 255), 2)
  cv2.circle(imgShw, (b[4 + 4], b[4 + 5]), 2, (0, 255, 255), 2)
  cv2.circle(imgShw, (b[4 + 6], b[4 + 7]), 2, (255, 255, 0), 2)
  cv2.circle(imgShw, (b[4 + 8], b[4 + 9]), 2, (0, 255, 0), 2)

  cx = b[0]
  cy = b[1] + 12
  cv2.putText(imgShw, text, (cx, cy),
    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
 cv2.namedWindow('res', cv2.WINDOW_NORMAL )
 cv2.imshow('res', imgShw)
 cv2.imwrite(outFolder+'/imgShw.jpg',imgShw)
 cv2.resizeWindow('res', imgDim, imgDim)
 cv2.waitKey(0)
 
 
 
 
 
 
 