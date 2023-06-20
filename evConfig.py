import sys
import os
visThres=0.5
confidenceThreshold=0.65
topK=25
nmsThreshold=0.5
keepTopK=750
tstImg=['001.jpg','002.jpg','003.jpg','004.jpg',\
 '005.jpg','006.jpg','007.jpg','008.png','009.png',\
 '010.png','011.png','012.png','013.png','014.png',\
 '015.jpg','016.png','b.png']

dtBufStr='dtBuf'
testImgRes='testIg'
modelFolder='model'

testImgFlr='D:/jhh/code/NdkOnnxTs/t03/dtSpc/tsImage/01/'
testImag=testImgFlr+tstImg[17-1]#11 5

#lbPath='D:/jhh/code/libfacedetection.train/data/HFace01/label2.txt'
#igRoot='D:/jhh/code/libfacedetection.train/data/HFace01'
lbPath='D:/jhh/code/mkHFaceFCOCO/data/label.txt'
igRoot='D:/jhh/code/mkHFaceFCOCO/data'


#modelName='jhhNetFinS1.pth'
modelName='yunetF5.pth'
detectModel=modelFolder+'/'+modelName

useTensorboard=True
lambda_bbox_eiou=10
lambda_iouhead_smoothl1=1
lambda_lm_smoothl1=9#1
lambda_cls_ce=1

imgDim=160
numClasses=2
gpuIds='0'
gpuIds =  [int(item) for item in gpuIds.split(',')]
momentum=0.9
weightDecay=5e-4
gamma=0.1
maxEpoch=140#3
lr=1e-4#2
batchSize=128#16
resumeEpoch=0

ndkPath='D:/jhh/src/OPNOUS_WQ5007_ISP_SDK_V1.0/sdk/zip_file_pack/zip_file_pack/ndk'
fNameOnnx=dtBufStr+'/jhhTs.onnx'
fxPDataP='D:/jhh/data/ptFixImgs/fix001'
bitWidth=8

outFolder='out'
