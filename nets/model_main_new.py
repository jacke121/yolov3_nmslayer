import time
import torch
import torch.nn as nn
from collections import OrderedDict

# from nets.coordConv import AddCoords
from common.utils import bbox_iou
from nets import mobilenet
from nets.layers import DetectionLayer
from nets.shufflenet_yolo import ShuffleNetV2
from nets.yolo_loss import YOLOLayer
from tools.utils import IoU
from .backbone import backbone_fn
import numpy as np

class ModelMain(nn.Module):
    def __init__(self, config, is_training=True):
        super(ModelMain, self).__init__()
        self.config = config
        self.training = is_training
        self.model_params = config["model_params"]
        self.num_classes = config["yolo"]["classes"]
        #  backbone
        # _backbone_fn = backbone_fn[self.model_params["backbone_name"]]
        # self.backbone = _backbone_fn(self.model_params["backbone_pretrained"])
        # self.backbone = mobilenet.mobilenetv2(self.model_params["backbone_pretrained"])
        self.backbone = ShuffleNetV2(scale=1, in_channels=3, c_tag=0.5, num_classes=2, activation=nn.ReLU,
                                     SE=False, residual=False)
        _out_filters = self.backbone.layers_out_filters
        #  embedding0
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

        self.yolo_losses = []
        self.losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls", "recall"]

        for i in range(3):
            yololayer = DetectionLayer(config["yolo"]["anchors"][i],config["yolo"]["classes"], config["img_w"],0.5)

            self.yolo_losses.append(yololayer)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def myloss(self, anchors,y_pred, y_true):
        self.reso=352
        self.anchors=anchors

        loss = dict()

        # 1. Prepare
        # 1.1 re-organize y_pred
        # [bs, (5+nC)*nA, gs, gs] => [bs, num_anchors, gs, gs, 5+nC]
        bs, _, gs, _ = y_pred.size()
        nA = len(self.anchors)
        nC = self.num_classes
        y_pred = y_pred.view(bs, nA, 5 + nC, gs, gs)
        y_pred = y_pred.permute(0, 1, 3, 4, 2)

        # 1.3 prepare anchor boxes
        stride = self.reso // gs
        anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors]
        anchor_bboxes = torch.zeros(3, 4).cuda()
        anchor_bboxes[:, 2:] = torch.Tensor(anchors)

        anchor_bboxes = anchor_bboxes.repeat(bs, 1, 1)


        # 2. Build gt [tx, ty, tw, th] and masks
        # TODO: f1 score implementation
        # total_num = 0
        gt_tx = torch.zeros(bs, nA, gs, gs, requires_grad=False)
        gt_ty = torch.zeros(bs, nA, gs, gs, requires_grad=False)
        gt_tw = torch.zeros(bs, nA, gs, gs, requires_grad=False)
        gt_th = torch.zeros(bs, nA, gs, gs, requires_grad=False)
        obj_mask = torch.zeros(bs, nA, gs, gs, requires_grad=False)
        non_obj_mask = torch.ones(bs, nA, gs, gs, requires_grad=False)
        cls_mask = torch.zeros(bs, nA, gs, gs, nC, requires_grad=False)
        start = time.time()
        # for batch_idx in range(bs):
        #     for box_idx, y_true_one in enumerate(y_true[batch_idx]):
        # total_num += 1
        gt_bbox = y_true[:,:,:4] * gs  # scale bbox relative to feature map
        gt_cls_label = y_true[:,:,4].int()

        # gt_xc, gt_yc, gt_w, gt_h = gt_bbox[:,:,0:4]
        gt_xc = gt_bbox[:,:,0]
        gt_yc = gt_bbox[:,:,1]
        gt_w = gt_bbox[:,:,2]
        gt_h = gt_bbox[:,:,3]
        gt_i, gt_j = gt_xc.int(), gt_yc.int()
        gt_box_shape = y_true[:,:,:4] * gs
        gt_box_shape[:, :, 0:2] = 0
        # gt_box_shape = torch.Tensor([0, 0, gt_w, gt_h]).unsqueeze(0).cuda()
        anch_ious = bbox_iou(gt_box_shape.view(self.batch_size, 1, 4), anchor_bboxes.cuda())
        anchor_ious = IoU(gt_box_shape, anchor_bboxes, format='center')
        best_anchor = np.argmax(anchor_ious)
        anchor_w, anchor_h = anchors[best_anchor]

        gt_tw[:, best_anchor, gt_i, gt_j] = torch.log(gt_w / anchor_w + 1e-16)
        gt_th[:, best_anchor, gt_i, gt_j] = torch.log(gt_h / anchor_h + 1e-16)
        gt_tx[:, best_anchor, gt_i, gt_j] = gt_xc - gt_i
        gt_ty[:, best_anchor, gt_i, gt_j] = gt_yc - gt_j

        obj_mask[:, best_anchor, gt_i, gt_j] = 1
        non_obj_mask[:, anchor_ious > 0.5] = 0  # FIXME: 0.5 as variable
        cls_mask[:, best_anchor, gt_i, gt_j, gt_cls_label] = 1

        # 3. activate raw y_pred
        end = time.time()
        print("yolo_losses",bs,len(y_true) ,end - start)
        pred_tx = torch.sigmoid(y_pred[..., 0])  # gt tx/ty are not deactivated
        pred_ty = torch.sigmoid(y_pred[..., 1])
        pred_tw = y_pred[..., 2]
        pred_th = y_pred[..., 3]
        pred_conf = y_pred[..., 4]
        pred_cls = y_pred[..., 5:]

        # 4. Compute loss
        obj_mask = obj_mask.cuda()
        non_obj_mask = non_obj_mask.cuda()
        cls_mask = cls_mask.cuda()
        gt_tx, gt_ty = gt_tx.cuda(), gt_ty.cuda()
        gt_tw, gt_th = gt_tw.cuda(), gt_th.cuda()

        # average over batch
        MSELoss = nn.MSELoss()
        BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        BCELoss = nn.BCELoss()
        CrossEntropyLoss = nn.CrossEntropyLoss()

        loss['x'] = MSELoss(pred_tx[obj_mask == 1], gt_tx[obj_mask == 1])
        loss['y'] = MSELoss(pred_ty[obj_mask == 1], gt_ty[obj_mask == 1])
        loss['w'] = MSELoss(pred_tw[obj_mask == 1], gt_tw[obj_mask == 1])
        loss['h'] = MSELoss(pred_th[obj_mask == 1], gt_th[obj_mask == 1])
        loss['cls'] = CrossEntropyLoss(pred_cls[obj_mask == 1], torch.argmax(cls_mask[obj_mask == 1], 1))
        loss['conf'] = BCEWithLogitsLoss(pred_conf[obj_mask == 1], obj_mask[obj_mask == 1])
        loss['non_conf'] = BCEWithLogitsLoss(pred_conf[non_obj_mask == 1], non_obj_mask[non_obj_mask == 1])
        loss['total_loss']=loss['x']+loss['y']+loss['w']+loss['h']+ loss['cls']+ loss['conf']+ loss['non_conf']
        #["total_loss", "x", "y", "w", "h", "conf", "cls", "recall"]
        return loss['total_loss'],loss['x'], loss['y'],loss['w'],loss['h'], loss['cls'],loss['conf'], loss['non_conf']
    def forward(self, x, targets=None):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x.cuda())
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)

        outputs=[]
        outputs.append(out0)
        outputs.append(out1)
        outputs.append(out2)
        losses = torch.zeros(len(self.losses_name), requires_grad=True).cuda()
        detections=torch.Tensor().cuda()
        for i in range(3):


            x= self.yolo_losses[i](outputs[i])

            detections = x if len(detections.size()) == 1 else torch.cat((detections, x), 1)
            _loss_item= self.myloss(self.config["yolo"]["anchors"][i],outputs[i], targets)



            # for (j,l) in _loss_item.items():
            for j, l in enumerate(_loss_item):
                losses[j] += l

        return losses,detections

    def load_darknet_weights(self, weights_path):
        import numpy as np
        #Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values
        # Needed to write header when saving weights
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        print ("total len weights = ", weights.shape)
        fp.close()

        ptr = 0
        all_dict = self.state_dict()
        all_keys = self.state_dict().keys()
        print (all_keys)
        last_bn_weight = None
        last_conv = None
        for i, (k, v) in enumerate(all_dict.items()):
            if 'bn' in k:
                if 'weight' in k:
                    last_bn_weight = v
                elif 'bias' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_bias: ", ptr, num_b, k)
                    ptr += num_b
                    # weight
                    v = last_bn_weight
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_weight: ", ptr, num_b, k)
                    ptr += num_b
                    last_bn_weight = None
                elif 'running_mean' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_mean: ", ptr, num_b, k)
                    ptr += num_b
                elif 'running_var' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_var: ", ptr, num_b, k)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv wight: ", ptr, num_b, k)
                    ptr += num_b
                    last_conv = None
                else:
                    raise Exception("Error for bn")
            elif 'conv' in k:
                if 'weight' in k:
                    last_conv = v
                else:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv bias: ", ptr, num_b, k)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv wight: ", ptr, num_b, k)
                    ptr += num_b
                    last_conv = None
        print("Total ptr = ", ptr)
        print("real size = ", weights.shape)


if __name__ == "__main__":
    config = {"model_params": {"backbone_name": "darknet_53"}}
    m = ModelMain(config)
    x = torch.randn(1, 3, 416, 416)
    y0, y1, y2 = m(x)
    print(y0.size())
    print(y1.size())
    print(y2.size())

