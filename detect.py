#!/usr/bin/python3
from __future__ import print_function

import os
import sys
import torch
import numpy as np
import cv2

from faceDNet import YuFaceDetectNet,\
 netPostProc
from fcResize import fcResize
import dataGen as dg
from detPostProc import detPostProc


imgDim=160
if(__name__=='__main__'):
 from evConfig import visThres,\
  confidenceThreshold,topK,nmsThreshold,keepTopK,\
  testImag,detectModel
 device="cuda" if  torch.cuda.is_available() else "cpu"
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
 imgShw=cv2.imread(testImag,cv2.IMREAD_COLOR)
 imgRsz=fcResize(imgRaw,imgDim,imgDim)
 imgShw=fcResize(imgShw,imgDim,imgDim)
 inPp=dg.imgPerProc(imgRaw,imgDim,imgDim)
 inpt=torch.from_numpy(inPp).to(device)
 headData2=net(inpt)
 for e in headData2:
  print(e.shape)
 loc, conf, iou=netPostProc(headData2,net.num_classes)
 dets=detPostProc(loc, conf, iou,imgDim,imgDim,confidenceThreshold\
  ,topK,keepTopK,nmsThreshold)
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
 cv2.resizeWindow('res', imgDim, imgDim)
 cv2.waitKey(0)