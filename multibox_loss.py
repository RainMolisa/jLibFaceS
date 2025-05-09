import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from libFcUtils import match, log_sum_exp
from eiou import eiou_loss
import numpy as np
GPU = True

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label,
                 neg_mining, neg_pos, neg_overlap, encode_target, rect_only):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.rect_only = rect_only
        self.smooth_point = 0.2

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,14)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,15] (last idx is the label).
        """

        loc_data, conf_data, iou_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 14)
        conf_t = torch.LongTensor(num, num_priors)
        iou_t = torch.Tensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, 0:14].data
            labels = targets[idx][:, -1].data
            #print(truths,labels)
            defaults = priors.data
            iou_t[idx] = match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)#error!!!!
        iou_t = iou_t.view(num, num_priors, 1)
        #print('conf_t=',np.sum(conf_t.numpy()>0))
        if GPU:
            device = priors.get_device()
            loc_t = loc_t.cuda(device)
            conf_t = conf_t.cuda(device)
            iou_t = iou_t.cuda(device)
        
        
        pos = conf_t > 0
        #print(pos)
        # Localization Loss
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        #print('pos_idx=',pos_idx)
        loc_p = loc_data[pos_idx].view(-1, 14)
        loc_t = loc_t[pos_idx].view(-1, 14)
        #print(loc_p)error!
        #print(loc_t)
        loss_bbox_eiou = eiou_loss(loc_p[:, 0:4], loc_t[:, 0:4], variance=self.variance, smooth_point=self.smooth_point, reduction='sum')
        loss_lm_smoothl1 = F.smooth_l1_loss(loc_p[:, 4:14], loc_t[:, 4:14], reduction='sum')

        # IoU diff
        pos_idx_ = pos.unsqueeze(pos.dim()).expand_as(iou_data)
        iou_p = iou_data[pos_idx_].view(-1, 1)
        iou_t = iou_t[pos_idx_].view(-1, 1)
        loss_iouhead_smoothl1 = F.smooth_l1_loss(iou_p, iou_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_cls_ce = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_cls_ce[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_cls_ce = loss_cls_ce.view(num, -1)
        _, loss_idx = loss_cls_ce.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_cls_ce = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses
        N = max(num_pos.data.sum().float(), 1)
        loss_bbox_eiou /= N
        loss_iouhead_smoothl1 /= N
        loss_lm_smoothl1 /= (N*5)
        loss_cls_ce /= N

        return loss_bbox_eiou, loss_iouhead_smoothl1, loss_lm_smoothl1, loss_cls_ce
