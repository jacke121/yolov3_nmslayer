import time
import torch
import torch.nn as nn
import numpy as np
import math
from common.utils import bbox_iou


class YOLOLayer(nn.Module):
    def __init__(self, batch_size,layer_num, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.8
        self.lambda_xy = 4
        self.lambda_wh = 4
        self.lambda_conf = 2
        self.lambda_cls = 2

        cuda = True if torch.cuda.is_available() else False
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
        g_dim = 11
        bs = batch_size
        x = [batch_size, 3, g_dim, g_dim]
        if layer_num == 1:
            g_dim = g_dim*2
            x = [batch_size, 3, g_dim, g_dim]

        elif layer_num == 2:
            g_dim = g_dim*4
            x = [batch_size, 3, g_dim, g_dim]

        self.g_dim = g_dim
        self.batch_size=batch_size
        self.stride = self.img_size[0] / g_dim
        self.scaled_anchors = [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        if cuda:
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()



    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        batch_size=bs
        g_dim=in_h

        # stride_h = self.img_size[1] / in_h
        # stride_w = self.img_size[0] / in_w
        # scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        grid_x = torch.linspace(0, self.g_dim - 1, self.g_dim).repeat(self.g_dim, 1).repeat(bs * self.num_anchors, 1, 1).view(x.size()).type(self.FloatTensor)
        grid_y = torch.linspace(0, self.g_dim - 1, self.g_dim).repeat(self.g_dim, 1).t().repeat(bs * self.num_anchors, 1, 1).view(x.shape).type(self.FloatTensor)

        anchor_w = self.FloatTensor(self.scaled_anchors).index_select(1, self.LongTensor([0]))
        anchor_h = self.FloatTensor(self.scaled_anchors).index_select(1, self.LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, self.g_dim * self.g_dim).view(x.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, self.g_dim * self.g_dim).view(x.shape)

        pred_boxes = self.FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if targets is not None:
            conf_mask = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim, requires_grad=False).cuda()
            noobj_mask = torch.ones(batch_size, self.num_anchors, g_dim, g_dim, requires_grad=False).cuda()

            tx = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim, requires_grad=False).cuda()
            ty = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim, requires_grad=False).cuda()
            tw = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim, requires_grad=False).cuda()
            th = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim, requires_grad=False).cuda()
            tconf = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim, requires_grad=False).cuda()
            tcls = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim, self.num_classes, requires_grad=False).cuda()
            #  build target
            nGT, nCorrect = self.get_target(pred_boxes,targets, self.scaled_anchors,
                                                                           in_w, in_h,
                                                                           self.ignore_threshold,   conf_mask, noobj_mask , tx, ty, tw, th, tconf,tcls)
            # print("targets time", time.time() - start)
            recall = float(nCorrect / nGT) if nGT else 1

            loss_x = self.bce_loss(x * conf_mask, tx * conf_mask)
            loss_y = self.bce_loss(y * conf_mask, ty * conf_mask)
            loss_w = self.mse_loss(w * conf_mask, tw * conf_mask)
            loss_h = self.mse_loss(h * conf_mask, th * conf_mask)
            loss_conf = self.bce_loss(conf * conf_mask, conf_mask) + 1 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            # loss_conf = self.bce_loss(conf * conf_mask, conf_mask) + 0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            loss_cls =self.bce_loss(pred_cls[conf_mask == 1], tcls[conf_mask == 1])
            # loss_cls=loss_conf
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
            return loss, loss_x.item()* self.lambda_xy, loss_y.item()* self.lambda_xy, loss_w.item()* self.lambda_wh,\
                loss_h.item()* self.lambda_wh, loss_conf.item()* self.lambda_conf, loss_cls.item()* self.lambda_cls,recall
        else:

            # Add offset and scale with anchors

            # Results
            _scale = torch.Tensor([self.stride, self.stride] * 2).type(self.FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self,pred_boxes, target, anchors, in_w, in_h, ignore_threshold,conf_mask, noobj_mask , tx, ty, tw, th, tconf,tcls):
        bs = target.size(0)
        # tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        nGT = 0
        nCorrect = 0

        gx_ = target[:, :, 1:5]* in_w

        gt_box = target[:, :, 1:5]* in_w
        gt_box[:, :, 0:2] = 0
        targetbox = torch.FloatTensor(
            np.concatenate((np.zeros((1, self.num_anchors, 2)), np.array([anchors])), 2)).repeat(self.batch_size, 1, 1)
        batch_anchor=torch.FloatTensor(anchors).repeat(self.batch_size, 1,1).cuda()
        for t in range(target.shape[1]):
            if target[:, t].sum() == 0:
                continue
            gi= np.array(gx_[:, t, 0].int())#.type(torch.uint8)
            gj= np.array(gx_[:, t, 1].int())#.type(torch.uint8)
            r_gt_box= gt_box[:, t, :]
            anch_ious= bbox_iou(r_gt_box.view(self.batch_size,1,4),targetbox.cuda())
            #80 3,11,11

            noobj_mask[anch_ious > ignore_threshold] = 0

            values = torch.max(anch_ious, 1, keepdim=True)[0]

            anch_ious[anch_ious == 0] = 1e+16
            c = anch_ious - values

            best_n=(c==0)#(c == 0).type(torch.uint8)
            conf_mask[best_n, gj, gi]=1
            # Coordinates
            tx[best_n,gj, gi] = (gx_[:, t, 0] - gx_[:, t, 0].int().float())#.cpu()
            ty[best_n,gj, gi] = (gx_[:, t, 1] - gx_[:, t, 1].int().float())#.cpu()

            tw[best_n,gj, gi] =torch.log(gx_[:,t, 2]/batch_anchor[best_n][:,0]+1e-16)#.cpu()
            th[best_n,gj, gi] =torch.log(gx_[:,t, 3]/batch_anchor[best_n][:,1]+1e-16)#.cpu()
            # object
            tconf[best_n,gj, gi] = 1
            # One-hot encoding of label
            tcls[best_n, gj, gi, np.array(target[:, t, 0])] = 1
            r_gt_box[:,0:2]=gx_[:, t, 0:2]
            pred_box = pred_boxes[best_n,gj, gi]
            iou = bbox_iou(r_gt_box.cuda(), pred_box, x1y1x2y2=False)
            nGT=nGT+self.batch_size
            nCorrect=nCorrect+int(sum(iou > 0.8))


        return  nGT, nCorrect
