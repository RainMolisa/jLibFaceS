import os
import shutil
import sys
import numpy as np
def writeNCHW_folder(nchw,fName):
 if os.path.exists(fName):
  shutil.rmtree(fName)
 os.makedirs(fName)
 n=0
 for chw in nchw:
  fN=fName+'/%02d'%n
  #print(fN)
  os.makedirs(fN)
  c=0
  for hw in chw:
   fNC=fN+'/%02d.txt'%c
   #print(fNC)
   np.savetxt(fNC,hw);
   c=c+1
  n=n+1
 return
def writeNCHW(nchw,fName):
 fo=open(fName,"w")
 fo.write(str(nchw.shape[0])+'\n')
 fo.write(str(nchw.shape[1])+'\n')
 fo.write(str(nchw.shape[2])+'\n')
 fo.write(str(nchw.shape[3])+'\n')
 for chw in nchw:
  for hw in chw:
   for w in hw:
    for e in w:
     fo.write(str(e)+'\n')
 fo.close()
 return
def writeNPUType(nchw,frac,bit,fName):
 fo=open(fName,"w")
 a=1<<frac
 b=32 if(bit<=8) else bit
 fillB=b-(nchw.shape[3]%b)
 for chw in nchw:
  for hw in chw:
   for w in hw:
    for e in w:
     v=e*a
     v=int(v)
     fo.write(str(v)+'\n')
    for i in range(fillB):
     fo.write('0\n')
 fo.close()
 return
if __name__=='__main__':
 img=np.zeros((11,12,3),dtype='float32')
 ig2=img.transpose(2,0,1)
 inpt=np.expand_dims(ig2,axis=0)
 for chw in inpt:
  for hw in inpt:
   for w in hw:
    for i in range(w.shape[0]):
     w[i]=i
 print(inpt)
 writeNCHW(inpt,'writeNCHWTest')
 #print(inpt)