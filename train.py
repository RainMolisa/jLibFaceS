#!/usr/bin/python3
import os
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from faceDNet import cfg,YuFaceDetectNet
from multibox_loss import MultiBoxLoss
from prior_box import PriorBox
from data import myDataSet
from evConfig import lbPath,igRoot,\
 useTensorboard,lambda_bbox_eiou,lambda_iouhead_smoothl1,\
 lambda_lm_smoothl1,lambda_cls_ce,imgDim,numClasses,\
 gpuIds,momentum,weightDecay,gamma,maxEpoch,lr,\
 batchSize,resumeEpoch
if(useTensorboard):
 from torch.utils.tensorboard import SummaryWriter
 prefix='yunet'
 logTag='{prefix}-bbox_{l_bbox}-iouhead_{l_iouhead}-lm_{l_lm}-cls_{l_cls}'.format(
  prefix=prefix,
  l_bbox=lambda_bbox_eiou,
  l_iouhead=lambda_iouhead_smoothl1,
  l_lm=lambda_lm_smoothl1,
  l_cls=lambda_cls_ce
 )
 logger=SummaryWriter(os.path.join('./tb_logs',logTag))

device="cuda" if torch.cuda.is_available() else "cpu"
print("using {} device".format(device))
net=YuFaceDetectNet('train',imgDim)

if True:
    print('Loading resume network...')
    state_dict = torch.load('model/yunetF5.pth')
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

optimizer=optim.SGD(net.parameters(),lr=lr,momentum=momentum,weight_decay=weightDecay)
criterion=MultiBoxLoss(numClasses,0.35,True,0,True,3,0.35,False,False)

prBox=PriorBox(cfg,image_size=(imgDim,imgDim))
with torch.no_grad():
 priors=prBox.forward()
 priors=priors.to(device)

def train():
 net.train()
 print('start train......')
 for epoch in range(resumeEpoch,maxEpoch):
  lrAdj=adjust_learning_rate_poly(optimizer,lr,epoch,maxEpoch)
  lossBboxEpoch=[]
  lossIouheadEpoch=[]
  lossLmEpoch=[]
  lossClsEpoch=[]
  lossEpoch=[]
  #loadT0=time.time()
  tLder=myDataSet(igRoot,lbPath,imgDim,imgDim,batchSize)
  numIterInEpoch=len(tLder)
  for tLderI,tLderE in enumerate(tLder):
   image,target=tLderE
   #img=np.float32(image)
   #img=img.transpose(2,0,1)
   #img = torch.from_numpy(img).unsqueeze(0)
   img=torch.from_numpy(image)
   img=img.type(torch.cuda.FloatTensor)
   img = img.to(device)
   out=net(img)
   target = [torch.from_numpy(anno).to(device) for anno in target]
   #print(len(target))
   lossBboxEiou,lossIouheadSmoothl1,lossLmSmoothl1,lossClsCe=criterion(out,priors,target)
   loss=lambda_bbox_eiou*lossBboxEiou+\
    lambda_iouhead_smoothl1*lossIouheadSmoothl1+\
    lambda_lm_smoothl1*lossLmSmoothl1+\
    lambda_cls_ce*lossClsCe
   #print(lossBboxEiou.item(),',',lossIouheadSmoothl1.item())
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   #
   lossBboxEpoch.append(lossBboxEiou.item())
   lossIouheadEpoch.append(lossIouheadSmoothl1.item())
   lossLmEpoch.append(lossLmSmoothl1.item())
   lossClsEpoch.append(lossClsCe.item())
   lossEpoch.append(loss.item())
   if(useTensorboard):
    logger.add_scalar(tag='Iter/loss_bbox',
     scalar_value=lossBboxEiou.item(),
     global_step=tLderI+epoch*numIterInEpoch
    )
    logger.add_scalar(
     tag='Iter/loss_iou',
     scalar_value=lossIouheadSmoothl1.item(),
     global_step=tLderI+epoch*numIterInEpoch
    )
    logger.add_scalar(
     tag='Iter/loss_landmark',
     scalar_value=lossLmSmoothl1.item(),
     global_step=tLderI+epoch*numIterInEpoch
    )
    logger.add_scalar(
     tag='Iter/loss_cls',
     scalar_value=lossClsCe.item(),
     global_step=tLderI+epoch*numIterInEpoch
    )
   if (tLderI % 5 == 0 or tLderI == numIterInEpoch - 1):
    print('Epoch:{:03d}/{} || iter: {:03d}/{} || L: {:.2f}({:.2f}) IOU: {:.2f}({:.2f}) LM: {:.2f}({:.2f}) C: {:.2f}({:.2f}) All: {:.2f}({:.2f}) || LR: {:.8f}'.format(
     epoch, maxEpoch, tLderI, numIterInEpoch, 
     lossBboxEiou.item(), np.mean(lossBboxEpoch),
     lossIouheadSmoothl1.item(), np.mean(lossIouheadEpoch),
     lossLmSmoothl1.item(), np.mean(lossLmEpoch),
     lossClsCe.item(), np.mean(lossClsEpoch),
     loss.item(),  np.mean(lossEpoch), lrAdj))
  if(useTensorboard):
   logger.add_scalar(
    tag='Epoch/loss_bbox',
    scalar_value=np.mean(lossBboxEpoch),
    global_step=epoch
   )
   logger.add_scalar(
    tag='Epoch/loss_iouhead',
    scalar_value=np.mean(lossIouheadEpoch),
    global_step=epoch
   )
   logger.add_scalar(
    tag='Epoch/loss_landmark',
    scalar_value=np.mean(lossLmEpoch),
    global_step=epoch
   )
   logger.add_scalar(
    tag='Epoch/loss_cls',
    scalar_value=np.mean(lossClsEpoch),
    global_step=epoch
   )
  
 netNw='./model/'
 torch.save(net.state_dict(), netNw+'jhhNetFinS1.pth')
 return



def adjust_learning_rate_poly(optimizer, initial_lr, iteration, max_iter):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * ( 1 - (iteration / max_iter)) * ( 1 - (iteration / max_iter))
    if ( lr < 1.0e-7 ):
      lr = 1.0e-7
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if(__name__=='__main__'):
 train()



