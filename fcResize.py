import os
import sys
import cv2
import numpy as np
def fcResize(ig1,h,w):
 k1=h/w;
 ig1_h, ig1_w = ig1.shape[0],ig1.shape[1]
 k2=ig1_h/ig1_w;
 if(ig1.ndim==3):
  ret=np.zeros((h,w,3),np.uint8)
 else:
  ret=np.zeros((h,w),np.uint8)
 if k1<k2:
  h2=h
  w2=h2/k2
  w2=int(w2)
  img2=cv2.resize(ig1,(w2,h2),interpolation=cv2.INTER_AREA)
  #cv2.imshow("img2",img2)
  #cv2.waitKey(0)
  d=int((w-w2)/2)
  #print(img2.shape)
  ret[0:h2,d:(d+w2)]=img2
 else:
  w2=w
  h2=w2*k2
  h2=int(h2)
  img2=cv2.resize(ig1,(w2,h2),interpolation=cv2.INTER_AREA)
  d=int((h-h2)/2)
  ret[d:(d+h2),0:w2]=img2
 return ret
if __name__ == '__main__':
 img_raw = cv2.imread('./dtSpc/tsImage/01/005.jpg', cv2.IMREAD_COLOR)# IMREAD_GRAYSCALE
 print(img_raw.ndim)
 img_out = fcResize(img_raw,240,320)
 print(img_out.shape)
 cv2.imshow("img2",img_out)
 cv2.waitKey(0)
 cv2.destroyAllWindows()