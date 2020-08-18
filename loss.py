import torch
import torch.nn as nn


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

global side
side = 64
class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.resnet = False
    def forward(self, input, gt):
        loss_dict = {}
        target = gt
        xgt = target[:,:,0]
        ygt = target[:,:,1]
        assert xgt.shape == (gt.shape[0],1000)
        assert ygt.shape == (gt.shape[0],1000)
        gt_img = torch.zeros(gt.shape[0],3,side,side)
        pred_img = torch.zeros(gt.shape[0],3,side,side)
        
        if self.resnet:
            try:
                xpred = input[0][:,:1000]
                ypred = input[0][:,-1000:]
            except:
                print(input.shape)
                xpred = input[:,:1000]
                ypred = input[:,-1000:]
        else:
            xpred = input[:,:1000]
            ypred = input[:,-1000:]
        
        
        for i in range gt.shape[0]:
            gt_img[i,:,xgt[i,:],ygt[i,:]] = 1.0 
            pred_img[i,:,xgt[i,:],ygt[i,:]] = 1.0 
            
        if output.shape[1] == 3:
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))

        #loss_dict['tv'] = total_variation_loss(output_comp)
        loss = loss_dict['style']+loss_dict['prc']
        return loss
