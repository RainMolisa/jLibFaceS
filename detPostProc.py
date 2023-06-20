import torch
import numpy as np
from faceDNet import cfg
from prior_box import PriorBox
from libFcUtils import decode
from nms import nms
def detPostProc(loc, conf, iou,h,w,confidence_threshold\
  ,top_k,keep_top_k,nms_threshold):
 if(False):
  from evConfig import pbCfg_H
  pbCfg_H(cfg,'./faceDPostProc/')
 device="cuda" if torch.cuda.is_available() else "cpu"
 scale = torch.Tensor([w, h, w, h,
                       w, h, w, h,
                       w, h, w, h,
                       w, h ])
 scale = scale.to(device)
 priorbox = PriorBox(cfg, image_size=(h, w))
 priors = priorbox.forward()
 priors = priors.to(device)
 prior_data = priors.data
 if(False):
  from evConfig import dtSpcStr,dtSpcBuf
  np.savetxt(dtSpcStr+'/'+dtSpcBuf+'/prior_data.txt',prior_data.cpu().numpy())
 boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
 boxes = boxes * scale
 boxes = boxes.cpu().numpy()
 cls_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
 iou_scores = iou.squeeze(0).data.cpu().numpy()[:, 0]
 # clamp here for the compatibility for ONNX
 _idx = np.where(iou_scores < 0.)
 iou_scores[_idx] = 0.
 _idx = np.where(iou_scores > 1.)
 iou_scores[_idx] = 1.
 scores = np.sqrt(cls_scores * iou_scores)

 # ignore low scores
 inds = np.where(scores > confidence_threshold)[0]
 boxes = boxes[inds]
 scores = scores[inds]

 # keep top-K before NMS
 order = scores.argsort()[::-1][:top_k]
 boxes = boxes[order]
 scores = scores[order]

 print('there are', len(boxes), 'candidates')

 #for ss in scores:
 #    print('score', ss)
 #for bb in boxes:
 #    print('box', bb, bb[2]-bb[0], bb[3]-bb[1])

 # do NMS
 dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
 selected_idx = np.array([0,1,2,3,14])
 keep = nms(dets[:,selected_idx], nms_threshold)
 dets = dets[keep, :]

 # keep top-K faster NMS
 dets = dets[:keep_top_k, :]
 return dets