import os
import numpy as np
import cv2
from fcResize import fcResize
def imgPerProc(img_raw,h,w):
 img_rsz = fcResize(img_raw,h,w)
 img_f32=np.float32(img_rsz)
 if(img_f32.ndim==3):
  ig2=img_f32.transpose(2,0,1)
 else:
  ig2=np.expand_dims(img_f32,axis=0)
 inpt=np.expand_dims(ig2,axis=0)
 return inpt
def dataGen0(fPath,h,w,isColor):
 listDir=os.listdir(fPath)
 for i in listDir:
  iPath=fPath+'\\'+i
  #print(iPath)
  flag=cv2.IMREAD_COLOR if(isColor) else cv2.IMREAD_GRAYSCALE
  img_raw = cv2.imread(iPath, flag)
  x=imgPerProc(img_raw,h,w)
  yield {'input': x}
def genLen0(fPath):
 listDir=os.listdir(fPath)
 return len(listDir)
if __name__=='__main__':
 fPath='D:\\jhh\\data\\widerFace\\WIDER_test\\images\\36--Football'
 #dg=dataGen0(fPath,240,320)
 #print(sum(1 for _ in dg))
 dg=dataGen0(fPath,240,320,True)
 print(genLen0(fPath))
 for data_batch in dg:
  #batch_size,_,_,_=data_batch['input'].shape
  print(data_batch['input'].shape)
  #print(data_batch['input'])