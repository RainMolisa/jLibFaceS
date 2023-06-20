import sys
import os
import shutil
import numpy as np
import cv2
import copy
lbPath='D:/jhh/code/libfacedetection.train/data/HFace01/label.txt'
igRoot='D:/jhh/code/libfacedetection.train/data/HFace01'
shPath='D:/jhh/code/libfacedetection.train/data/HFace01/show'
def readData(labelPath,imgRoot):
 ret=list()
 for lblLine in open(labelPath,"r"):
  lblLine2=lblLine.strip()
  lblE=lblLine2.split(' ')
  fN=imgRoot+'/'+lblE[0]
  #print(fN)
  N=int(lblE[1])
  lbl=np.zeros([N,15],dtype=float)
  for i in range(N):
   for j in range(15):
    k=float(lblE[2+i*15+j])
    lbl[i,j]=k
  ret.append([fN,lbl])
 return ret
def drawPoint(img_rsz,dets):
 retImg=copy.deepcopy(img_rsz)
 for b in dets:
  b = list(map(int, b))
  cv2.rectangle(retImg, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
  cv2.circle(retImg, (b[4], b[4 + 1]), 2, (255, 0, 0), 2)
  cv2.circle(retImg, (b[4 + 2], b[4 + 3]), 2, (0, 0, 255), 2)
  cv2.circle(retImg, (b[4 + 4], b[4 + 5]), 2, (0, 255, 255), 2)
  cv2.circle(retImg, (b[4 + 6], b[4 + 7]), 2, (255, 255, 0), 2)
  cv2.circle(retImg, (b[4 + 8], b[4 + 9]), 2, (0, 255, 0), 2)
 return retImg
if(__name__=='__main__'):
 lblC=readData(lbPath,igRoot)
 if(os.path.exists(shPath)):
  shutil.rmtree(shPath)
 os.makedirs(shPath)
 for i,e in enumerate(lblC):
  imgRaw = cv2.imread(e[0], cv2.IMREAD_COLOR)
  imgSh=drawPoint(imgRaw,e[1])
  cv2.imwrite(shPath+'/%03d.png'%(i),imgSh)
 #print(lblC)