import os
import shutil
from shutil import copyfile
from evConfig import modelFolder,\
 modelName
relPath='release'
if(os.path.exists(relPath)):
 shutil.rmtree(relPath)
os.makedirs(relPath)
pwd=os.getcwd()
pwd=pwd.replace('\\','/')
print(pwd)
mdBinFolder=modelFolder+'/'+modelName.replace('.pth','.bin')

srcFile=pwd+'/'+mdBinFolder+'/machine_code_file/input_output_info.h'
trgFile=pwd+'/'+relPath+'/input_output_info.h'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)
srcFile=pwd+'/'+mdBinFolder+'/for_flash_in_other_running_method/code.bin'
trgFile=pwd+'/'+relPath+'/code.bin'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)
srcFile=pwd+'/'+mdBinFolder+'/for_flash_in_other_running_method/w.bin'
trgFile=pwd+'/'+relPath+'/w.bin'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)

srcFile=pwd+'/csrc/faceBox.h'
trgFile=pwd+'/'+relPath+'/faceBox.h'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)
srcFile=pwd+'/csrc/faceDPostProc.h'
trgFile=pwd+'/'+relPath+'/faceDPostProc.h'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)
srcFile=pwd+'/csrc/fDPPItf.h'
trgFile=pwd+'/'+relPath+'/fDPPItf.h'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)
srcFile=pwd+'/csrc/NCHW.h'
trgFile=pwd+'/'+relPath+'/NCHW.h'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)
srcFile=pwd+'/csrc/prepss.h'
trgFile=pwd+'/'+relPath+'/prepss.h'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)
srcFile=pwd+'/csrc/ptrResize.h'
trgFile=pwd+'/'+relPath+'/ptrResize.h'
print(srcFile,trgFile)
copyfile(srcFile,trgFile)
