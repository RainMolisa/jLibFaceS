#!/usr/bin/python3
from __future__ import print_function
import os
import shutil
import sys
import cv2
import numpy as np
import torch

from evConfig import ndkPath
sys.path.append(ndkPath)
import quantize as qtz
import modelpack as mdp
import quant_tools.numpy_net as npn
import layers

from writeNCHW import writeNCHW,writeNPUType
from faceDNet import netPostProc
from fcResize import fcResize
import dataGen as dg
from detPostProc import detPostProc

imgDim=160
if(__name__=='__main__'):
 device="cuda" if torch.cuda.is_available() else "cpu"
 from evConfig import visThres,\
  confidenceThreshold,topK,nmsThreshold,\
  keepTopK,modelFolder,modelFolder,bitWidth,\
  testImgFlr,tstImg,outFolder,modelName
 mdBinFolder=modelFolder+'/'+modelName.replace('.pth','.bin')
 fixModPath=modelFolder+'/'+modelName.replace('.pth','.fix')
 qllFilePath=fixModPath+'/qtll.prototxt'
 qpdFilePath=fixModPath+'/qtpd.npz'
 layerList,paramDict=mdp.load_from_file(qllFilePath,qpdFilePath)
 print('load finish')
 nwOutList=layers.get_network_output(layerList)
 nwOutList.sort()
 if(os.path.exists(outFolder)):
  shutil.rmtree(outFolder)
 os.makedirs(outFolder)
 for imgId,strImg in enumerate(tstImg):
  testImag=testImgFlr+strImg
  imgRaw=cv2.imread(testImag,cv2.IMREAD_GRAYSCALE)
  imgShw=cv2.imread(testImag,cv2.IMREAD_COLOR)
  imgRsz=fcResize(imgRaw,imgDim,imgDim)
  imgShw=fcResize(imgShw,imgDim,imgDim)
  inPp=dg.imgPerProc(imgRaw,imgDim,imgDim)
  hd3=npn.run_layers(inPp,layerList,nwOutList,param_dict=paramDict,\
   bitwidth=bitWidth,quant=True)
  headData3=list()
  for ole in nwOutList:
   headData3.append(torch.from_numpy(hd3[ole]).to(device))
  loc,conf,iou=netPostProc(headData3,2)
  dets=detPostProc(loc, conf, iou,imgDim,imgDim,confidenceThreshold\
   ,topK,keepTopK,nmsThreshold)
  face_cc = 0
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
  cv2.imwrite(outFolder+'/imgShw%03d.jpg'%imgId,imgShw)