import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms

#print("faceDetecNet5")
cfg = {
    'name': 'faceDetecNet5',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96]],
    'steps': [8,16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'fmth':0b1110
}

def netPostProc(headData2,num_classes):
 print('**********netPostProc start**********')
 sfm=nn.Softmax(dim=-1)
 head_data = list()
 loc_data = list()
 conf_data = list()
 iou_data = list()
 for x in headData2:
  head_data.append(x.permute(0, 2, 3, 1).contiguous())
 
 head_data = torch.cat([o.view(o.size(0), -1) for o in head_data], 1)
 print('head_data=',head_data.shape)
 head_data = head_data.view(head_data.size(0), -1, 17)
 print('head_data=',head_data.shape)

 loc_data = head_data[:, :, 0:14]
 conf_data = head_data[:, :, 14:16]
 iou_data = head_data[:,:, 16:17]
 #output = (loc_data, conf_data, iou_data)
 print ('loc_data=', loc_data.shape)
 loc_data = loc_data.view(-1, 14)
 conf_data = sfm(conf_data.view(-1, num_classes))
 iou_data = iou_data.view(-1, 1)
 output = (loc_data, conf_data, iou_data)
 print('**********netPostProc end **********')
 return output

def combine_conv_bn(conv, bn):
    conv_result = nn.Conv2d(conv.in_channels, conv.out_channels, 
                            kernel_size=conv.kernel_size, stride=conv.stride, 
                            padding=conv.padding, groups = conv.groups, bias=True)
    
    scales = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv_result.bias[:] = (conv.bias - bn.running_mean) * scales + bn.bias
    for ch in range(conv.out_channels):
        conv_result.weight[ch, :, :, :] = conv.weight[ch, :, :, :] * scales[ch]

    return conv_result

def convert_param2string(conv, name, is_depthwise=False, isfirst3x3x3=False, precision='.3g'):
    '''
    Convert the weights to strings
    '''
    (out_channels, in_channels, width, height) = conv.weight.size()

    if (isfirst3x3x3):
        w = conv.weight.detach().numpy().reshape((-1,27))
        w_zeros = np.zeros((out_channels ,5))
        w = np.hstack((w, w_zeros))
        w = w.reshape(-1)
    elif (is_depthwise):
        w = conv.weight.detach().numpy().reshape((-1,9)).transpose().reshape(-1)
    else:
        w = conv.weight.detach().numpy().reshape(-1)

    b = conv.bias.detach().numpy().reshape(-1)

    if (isfirst3x3x3):
        lengthstr_w = str(out_channels) + '* 32 * 1 * 1'
        # print(conv.in_channels, conv.out_channels, conv.kernel_size)
    else:
        lengthstr_w = str(out_channels) + '*' + str(in_channels) + '*' + str(width) + '*' + str(height)
    resultstr = 'float ' + name + '_weight[' + lengthstr_w + '] = {'

    for idx in range(w.size - 1):
        resultstr += (format(w[idx], precision) + ',')
    resultstr += (format(w[-1], precision))
    resultstr += '};\n'

    resultstr += 'float ' + name + '_bias[' + str(out_channels) + '] = {'
    for idx in range(b.size - 1):
        resultstr += (format(b[idx], precision) + ',')
    resultstr += (format(b[-1], precision))
    resultstr += '};\n'

    return resultstr
    

class ConvDPUnit(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super(ConvDPUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = F.relu(x, inplace=True)
        return x
    def convert_to_cppstring(self, varname):
        rs1 = convert_param2string(self.conv1, varname+'_1', False)
        if self.withBNRelu:
            rs2 = convert_param2string(combine_conv_bn(self.conv2, self.bn), varname+'_2', True)
        else:
            rs2 = convert_param2string(self.conv2, varname+'_2', True)
        return rs1 + rs2

def ConvDPUnitModOut(cdp,idx,N):
    out_channels=int(cdp.out_channels/N);
    ret=ConvDPUnit(cdp.in_channels,out_channels,cdp.withBNRelu)
    ret.conv1 = nn.Conv2d(cdp.in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
    ret.conv1.weight.requires_grad=False
    ret.conv1.bias.requires_grad=False
    ret.conv1.weight[:,:,:,:]=cdp.conv1.weight[idx:cdp.out_channels:N,:,:,:]
    ret.conv1.bias[:]=cdp.conv1.bias[idx:cdp.out_channels:N]
    #ret.conv1=cdp.conv1
    ret.conv2=nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)
    ret.conv2.weight.requires_grad=False
    ret.conv2.bias.requires_grad=False
    ret.conv2.weight[:,:,:,:]=cdp.conv2.weight[idx:cdp.out_channels:N,:,:,:]
    ret.conv2.bias[:]=cdp.conv2.bias[idx:cdp.out_channels:N]
    return ret

class Conv_head(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Conv_head, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(mid_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)

        return x

    def convert_to_cppstring(self, varname):
       rs1 = convert_param2string(self.conv1, varname+'_0', False, True)
       rs2 = self.conv2.convert_to_cppstring(varname)
       return rs1 + rs2

class Conv4layerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super(Conv4layerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def Conv4layerBlockModOut(cv4lyb,idx,N):
    out_channels=cv4lyb.out_channels/N
    out_channels=int(out_channels)
    #print(out_channels)
    ret=Conv4layerBlock(cv4lyb.in_channels,out_channels)
    ret.conv1=cv4lyb.conv1
    print('yeah~')
    ret.conv2=ConvDPUnitModOut(cv4lyb.conv2,idx,N)
    return ret

class YuFaceDetectNet(nn.Module):

    def __init__(self, phase, size):
        super(YuFaceDetectNet, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.size = size
        self.c0Toc3= nn.Conv2d(1, 3, 1, 1, 0, bias=False, groups=1)
        self.c0Toc3.weight.requires_grad=False
        self.c0Toc3.weight[:,:,:,:]=1
        self.model0 = Conv_head(3, 16, 16)
        self.model1 = Conv4layerBlock(16, 64)
        self.model2 = Conv4layerBlock(64, 64)
        self.model3 = Conv4layerBlock(64, 64)
        self.model4 = Conv4layerBlock(64, 64)
        self.model5 = Conv4layerBlock(64, 64)

        self.head = self.multibox(self.num_classes)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def multibox(self, num_classes):
        head_layers = []
        head_layers += [Conv4layerBlock(self.model3.out_channels, 3 * (14 + 2 + 1), False)]
        head_layers += [Conv4layerBlock(self.model4.out_channels, 2 * (14 + 2 + 1), False)]
        head_layers += [Conv4layerBlock(self.model5.out_channels, 2 * (14 + 2 + 1), False)]
        return nn.Sequential(*head_layers)
    def forward(self, x):
        
        detection_sources = list()
        if(self.phase=='test'):
            headData2=list()
            x = self.c0Toc3(x)
        else:
            head_data = list()
            loc_data = list()
            conf_data = list()
            iou_data = list()
        x = self.model0(x)
        x = F.max_pool2d(x, 2)
        x = self.model1(x)
        x = self.model2(x)
        x = F.max_pool2d(x, 2)
        x = self.model3(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model4(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model5(x)
        detection_sources.append(x)
        if(self.phase=='test'):
            for (x,h) in zip(detection_sources,self.head):
                headData2.append(h(x))
            return headData2
        else:
            for (x, h) in zip(detection_sources, self.head):
                head_data.append(h(x).permute(0, 2, 3, 1).contiguous())
            head_data = torch.cat([o.view(o.size(0), -1) for o in head_data], 1)
            head_data = head_data.view(head_data.size(0), -1, 17)

            loc_data = head_data[:, :, 0:14]
            conf_data = head_data[:, :, 14:16]
            iou_data = head_data[:,:, 16:17]
            output = (loc_data, conf_data, iou_data)
            # print ('output size', head_data[0].size())
            output = (loc_data.view(loc_data.size(0), -1, 14),
                    conf_data.view(conf_data.size(0), -1, self.num_classes),
                    iou_data.view(iou_data.size(0), -1, 1))
            return output

