import sys
import os
import shutil
import numpy as np
import cv2
import copy
#import torch
from readData import readData
class myDataSet(object):
 def __init__(self,imgsRoot,labelPath,w,h,bz):
  super().__init__()
  self.dtList=readData(labelPath,imgsRoot)
  self.dtIter=iter(self.dtList)
  self.size=(w,h,bz)
 def __next__(self):
  w,h,bz=self.size
  lblRets=list()
  imgRets=np.zeros([bz,3,w,h],dtype='float')
  for k in range(bz):
   dte=next(self.dtIter)
   imgRaw=cv2.imread(dte[0],cv2.IMREAD_COLOR)
   imgRet=cv2.resize(imgRaw,(w,h),interpolation=cv2.INTER_AREA)
   lblRet=np.zeros(dte[1].shape,dtype='float')
   wR=1.0/float(imgRaw.shape[1])#float(w)
   hR=1.0/float(imgRaw.shape[0])#float(h)
   for i,e1 in enumerate(dte[1]):
    for j in range(7):
     lblRet[i,2*j+0]=e1[2*j+0]*wR
     lblRet[i,2*j+1]=e1[2*j+1]*hR
    lblRet[i,14]=e1[14]
   lblRets.append(lblRet)
   imgRet=np.float32(imgRet)
   imgRet=imgRet.transpose(2,0,1)
   imgRets[k,:,:,:]=imgRet
  return imgRets,lblRets
 def __iter__(self):
  return self
 def __len__(self):
  dtSz=len(self.dtList)
  bz=self.size[2]
  a=int(dtSz/bz)
  if(dtSz%bz!=0):
   a=a+1
  return a
def get_train_loader(imgsRoot,labelPath,w,h):
 return
def scPoint(p,w,h):
 for e in p:
  for i in range(7):
   e[i*2+0]=e[i*2+0]*w
   e[i*2+1]=e[i*2+1]*h
 return

if(__name__=='__main__'):
 lbPath='D:/jhh/code/libfacedetection.train/data/HFace01/label2.txt'
 igRoot='D:/jhh/code/libfacedetection.train/data/HFace01'
 shPath='D:/jhh/code/libfacedetection.train/data/HFace01/show'
 w,h,bz=320,320,16
 '''
 a=myDataSet(igRoot,lbPath,320,320,16)
 for i,e in enumerate(a):
  print("%03d:"%i)
  for j in e:
   print("-",j)
 '''
 if(os.path.exists(shPath)):
  shutil.rmtree(shPath)
 os.makedirs(shPath)
 from readData import drawPoint
 for k in range(4):
  a=myDataSet(igRoot,lbPath,w,h,bz)
  print(len(a))
  for i,e in enumerate(a):
   images,targets=e
   for j,t in enumerate(targets):
    img=np.zeros([3,w,h],dtype='uint8')
    img=images[j,:,:,:]
    img=img.transpose(1,2,0)
    cv2.imwrite(shPath+'/k%03di%03dj%03d.png'%(k,i,j),img)
    img=cv2.imread(shPath+'/k%03di%03dj%03d.png'%(k,i,j),cv2.IMREAD_COLOR)
    scPoint(t,w,h)
    igSh=drawPoint(img,t)
    cv2.imwrite(shPath+'/k%03di%03dj%03d.png'%(k,i,j),igSh)
    #print('k=',k,'i=',i)